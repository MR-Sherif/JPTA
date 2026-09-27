from tqdm import tqdm

import torch
import torch.nn.functional as F
import os
import clip
from prompt_banks import get_template_bank


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    target = target.to(pred.device)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().item()
    acc = 100 * acc / target.shape[0]
    return acc


def compute_zero_shot_logits(query_features, clip_prototypes):
    if isinstance(clip_prototypes, dict):
        if "legacy" in clip_prototypes:
            clip_prototypes = clip_prototypes["legacy"]
        else:
            clip_prototypes = next(iter(clip_prototypes.values()))
    if clip_prototypes.dim() == 3:
        clip_prototypes = clip_prototypes.mean(dim=0)
        clip_prototypes = F.normalize(clip_prototypes, dim=0)
    clip_logits = 100 * query_features.to(clip_prototypes.device) @ clip_prototypes
    return clip_logits.squeeze()


def _as_probabilities(output):
    if output.numel() == 0:
        return output
    row_sums = output.sum(dim=1)
    is_prob_like = (
        torch.all(output >= 0)
        and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    )
    if is_prob_like:
        return output.clamp_min(1e-12)
    return F.softmax(output, dim=1).clamp_min(1e-12)


def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            gap = accuracies[mask].float().mean() - confidences[mask].mean()
            ece += gap.abs() * mask.float().mean()
    return float(ece.item())


def adaptive_expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()
    sorted_conf, order = torch.sort(confidences)
    sorted_acc = accuracies[order]
    chunks_conf = torch.chunk(sorted_conf, n_bins)
    chunks_acc = torch.chunk(sorted_acc, n_bins)
    total = probs.size(0)
    ada_ece = 0.0

    for conf_chunk, acc_chunk in zip(chunks_conf, chunks_acc):
        if conf_chunk.numel() == 0:
            continue
        weight = conf_chunk.numel() / total
        ada_ece += abs(float(acc_chunk.mean().item() - conf_chunk.mean().item())) * weight
    return ada_ece


def compute_calibration_metrics(output, labels):
    probs = _as_probabilities(output)
    labels = labels.to(probs.device)
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    log_probs = probs.log()

    nll = float(F.nll_loss(log_probs, labels, reduction='mean').item())
    brier = float(((probs - one_hot) ** 2).sum(dim=1).mean().item())

    return {
        'ece': expected_calibration_error(probs, labels),
        'ada_ece': adaptive_expected_calibration_error(probs, labels),
        'brier': brier,
        'nll': nll,
    }


def clip_classifier(classnames, template, clip_model, reduce='mean', gpt=False, wordnet_dict=None):
    with torch.no_grad():
        clip_weights = []
        class_specific_templates = isinstance(template, dict)
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:

                    # Tokenize the prompts
                    name = name.replace('_', ' ')

                    texts = [t.format(name) for t in template]
                    texts = clip.tokenize(texts).cuda()

                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce=='mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i+=1
                indices.append(i)

            return clip_weights, indices
        else:

            for classname in classnames:

                # Tokenize the prompts
                raw_classname = classname
                classname = classname.replace('_', ' ')

                if gpt or class_specific_templates:
                    texts = [t.format(classname) for t in template[raw_classname]]
                else:
                    texts = [t.format(classname)  for t in template]
                texts = clip.tokenize(texts).cuda()

                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if reduce=='mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)

            clip_weights = torch.stack(clip_weights, dim=-1).cuda()
    return clip_weights


def get_all_features(args, test_loader, dataset, clip_model):
    use_auto_prompt_bank = (
        getattr(args, "method", None) in {"CTA", "JPTA"}
        and bool(getattr(args, "template_bank", False))
        and getattr(args, "prompt_bank_strategy", "legacy") == "auto"
    )
    if use_auto_prompt_bank:
        clip_prototypes = {}
        for strategy in ("legacy", "published", "hybrid"):
            template_bank = get_template_bank(args.dataset, dataset=dataset, strategy=strategy)
            if template_bank:
                clip_prototypes[strategy] = clip_classifier(
                    dataset.classnames,
                    template_bank,
                    clip_model,
                    reduce=None,
                )
    else:
        clip_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model, reduce=None)
    test_features, test_labels = pre_load_features(args, "test", clip_model, test_loader)

    return test_features, test_labels, clip_prototypes


def build_cache_model(cfg, clip_model, train_loader_cache, n_views=0, reduce=None):
    print('... for shot samples from train split:')

    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []
        if n_views == 0:
            n_epochs =1
        else:
            n_epochs = n_views
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(n_epochs):
                train_features = []
                train_labels = []
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))



        if n_views == 1:
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            #cache_keys = cache_keys.permute(1, 0)
        else:
            cache_keys = torch.cat(cache_keys, dim=0) # [n_views, n_classes, n_features]
            if reduce == 'mean':
                cache_keys = cache_keys.mean(0, keepdim=True)

            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys.permute(0, 2, 1)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values






def pre_load_features(args, split, clip_model, loader, n_views=1):
    os.makedirs(args.cache_dir, exist_ok=True)

    should_compute = not args.load
    if args.load:
        try:
            features = torch.load(args.cache_dir + "/" + split + "_f.pt")
            labels = torch.load(args.cache_dir + "/" + split + "_l.pt")
            should_compute = False
        except FileNotFoundError:
            print("Cache not found, extracting features...")
            should_compute = True

    if should_compute:
        features, labels = [], []

        with torch.no_grad():

            for view in range(n_views):
                length = 0
                for i, (images, target) in enumerate(tqdm(loader)):
                    if n_views == 1:

                        images, target = images.cuda(), target.cuda()


                        image_features = clip_model.encode_image(images)

                        image_features /= image_features.norm(dim=-1, keepdim=True)


                        features.append(image_features.cpu())
                        labels.append(target.cpu())
                    else:
                        images, target = images.cuda(), target.cuda()
                        image_features = clip_model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        if view == 0:
                            labels.append(target.cpu())
                            if i ==0:
                                mean_features = image_features
                            else:
                                mean_features = torch.cat((mean_features, image_features))
                        else:
                            mean_features[length:length+image_features.size(0)] += image_features
                            length += image_features.size(0)

        if n_views > 1:
            mean_features = mean_features / n_views
            features = mean_features / mean_features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels)

        elif n_views==1:
            features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, args.cache_dir + "/" + split + "_f.pt")
        torch.save(labels, args.cache_dir + "/" + split + "_l.pt")

    return features, labels
