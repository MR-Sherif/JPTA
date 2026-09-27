import math

import torch
import torch.nn.functional as F

from .StatA import (
    Gaussian,
    StatA_solver,
    build_affinity_matrix,
    get_zero_shot_logits,
    init_cov,
    update_cov,
    update_mu,
)


EPS = 1e-12


def maybe_move_to_cpu(tensor, return_cpu=True):
    return tensor.cpu() if return_cpu else tensor


def canonicalize_clip_prototypes(clip_prototypes):
    if clip_prototypes.dim() == 2:
        clip_prototypes = clip_prototypes.unsqueeze(0)
    if clip_prototypes.dim() != 3:
        raise ValueError(
            f"Expected clip_prototypes to have 2 or 3 dims, got shape {tuple(clip_prototypes.shape)}."
        )
    if clip_prototypes.size(0) > 1:
        clip_prototypes = clip_prototypes.mean(dim=0, keepdim=True)
    return F.normalize(clip_prototypes, dim=1)


def compute_prompt_bank_coverage(anchor_probs):
    num_classes = max(anchor_probs.size(1), 1)
    batch_marginal = anchor_probs.mean(dim=0).clamp_min(EPS)
    entropy = -(batch_marginal * batch_marginal.log()).sum()
    return float((torch.exp(entropy) / num_classes).item())


def select_auto_prompt_bank(clip_prototypes, query_features, template_bank_adaptation):
    if not isinstance(clip_prototypes, dict):
        return clip_prototypes, template_bank_adaptation

    legacy_bank = clip_prototypes.get("legacy")
    if legacy_bank is None:
        legacy_bank = next(iter(clip_prototypes.values()))

    legacy_mean = canonicalize_clip_prototypes(legacy_bank.float()).squeeze(0)
    legacy_logits = 100.0 * query_features.to(legacy_mean.device).float() @ legacy_mean
    legacy_probs = F.softmax(legacy_logits, dim=1)
    coverage = compute_prompt_bank_coverage(legacy_probs)

    if coverage <= 0.50:
        selected_bank = clip_prototypes.get("published")
        if selected_bank is None:
            selected_bank = clip_prototypes.get("hybrid")
        if selected_bank is None:
            selected_bank = legacy_bank
        return selected_bank, False
    if coverage >= 0.82:
        return legacy_bank, template_bank_adaptation

    selected_bank = clip_prototypes.get("hybrid")
    if selected_bank is None:
        selected_bank = legacy_bank
    return selected_bank, template_bank_adaptation


def adapt_prompt_bank(
    clip_prototypes,
    query_features,
    topk_per_class=64,
    temperature=0.07,
):
    if clip_prototypes.dim() != 3 or clip_prototypes.size(0) <= 1:
        return clip_prototypes

    prompt_bank = F.normalize(clip_prototypes, dim=1)
    mean_prototypes = canonicalize_clip_prototypes(prompt_bank).squeeze(0)
    mean_logits = 100.0 * query_features @ mean_prototypes
    mean_probs = F.softmax(mean_logits, dim=1)

    num_prompts, feature_dim, num_classes = prompt_bank.shape
    topk_per_class = max(1, min(int(topk_per_class), query_features.size(0)))

    topk_scores, topk_indices = mean_probs.topk(topk_per_class, dim=0)
    expanded_features = query_features[topk_indices.reshape(-1)].view(topk_per_class, num_classes, feature_dim)
    selected_features = expanded_features.permute(1, 0, 2).contiguous()
    selected_scores = topk_scores.transpose(0, 1).contiguous().clamp_min(EPS)

    prompt_vectors = prompt_bank.permute(2, 0, 1).contiguous()
    prompt_sims = torch.einsum("ctd,cpd->ctp", selected_features, prompt_vectors)
    weighted_prompt_scores = (prompt_sims * selected_scores.unsqueeze(-1)).sum(dim=1)
    weighted_prompt_scores = weighted_prompt_scores / selected_scores.sum(dim=1, keepdim=True).clamp_min(EPS)

    if num_prompts > 6:
        batch_marginal = mean_probs.mean(dim=0).clamp_min(EPS)
        batch_entropy = -(batch_marginal * batch_marginal.log()).sum()
        coverage = float((torch.exp(batch_entropy) / max(num_classes, 1)).item())
        base_fraction = 0.15 + 0.85 * (1.0 - coverage)

        class_support = topk_scores.mean(dim=0)
        support_norm = class_support / class_support.max().clamp_min(EPS)
        min_fraction = 2.0 / num_prompts
        max_fraction = min(1.0, 0.4 + 0.6 * base_fraction)
        keep_fraction = (min_fraction + support_norm * (max_fraction - min_fraction)).clamp(
            min=min_fraction,
            max=1.0,
        )
        keep_counts = torch.round(num_prompts * keep_fraction).long().clamp(min=2, max=num_prompts)

        adapted = []
        for class_idx in range(num_classes):
            keep_count = int(keep_counts[class_idx].item())
            selected_prompt_ids = weighted_prompt_scores[class_idx].topk(keep_count).indices
            class_prompt_vectors = prompt_vectors[class_idx, selected_prompt_ids, :]
            class_prototype = class_prompt_vectors.mean(dim=0)
            adapted.append(F.normalize(class_prototype, dim=0))

        adapted = torch.stack(adapted, dim=0)
        return adapted.transpose(0, 1).unsqueeze(0)

    prompt_weights = F.softmax(weighted_prompt_scores / max(float(temperature), EPS), dim=1)
    adapted = torch.einsum("cp,cpd->cd", prompt_weights, prompt_vectors)
    adapted = F.normalize(adapted, dim=1)
    return adapted.transpose(0, 1).unsqueeze(0)


def class_posterior_mean(z, prior_strength):
    counts = z.sum(dim=0)
    num_classes = z.size(1)
    pi = counts + float(prior_strength) / num_classes
    pi = pi / pi.sum().clamp_min(EPS)
    return pi.clamp_min(EPS), counts


def effective_num_classes(pi):
    entropy = -(pi * pi.clamp_min(EPS).log()).sum()
    return float(torch.exp(entropy).item())


def compute_soft_class_means(query_features, z):
    counts = z.sum(dim=0).clamp_min(EPS)
    means = torch.einsum("nk,nd->kd", z, query_features)
    means = means / counts.unsqueeze(1)
    means = F.normalize(means, dim=1)
    return means, counts


def compute_transport_weights(z):
    counts = z.sum(dim=0).clamp_min(EPS)
    class_confidence = (z.pow(2).sum(dim=0) / counts).clamp(0.0, 1.0)
    return counts * class_confidence


def estimate_orthogonal_transport(text_prototypes, image_prototypes, class_weights):
    valid = class_weights > EPS
    feature_dim = text_prototypes.size(1)
    identity = torch.eye(feature_dim, device=text_prototypes.device, dtype=text_prototypes.dtype)
    if valid.sum() < 2:
        return identity, 0.0

    weights = class_weights[valid].sqrt().unsqueeze(1)
    text_valid = F.normalize(text_prototypes[valid], dim=1)
    image_valid = F.normalize(image_prototypes[valid], dim=1)

    cross_covariance = (weights * text_valid).transpose(0, 1) @ (weights * image_valid)
    try:
        U, singular_values, Vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    except RuntimeError:
        return identity, 0.0

    transport = U @ Vh
    if torch.det(transport) < 0:
        correction = identity.clone()
        correction[-1, -1] = -1
        transport = U @ correction @ Vh
    alignment_score = float(singular_values.mean().item() / max(weights.sum().item(), EPS))
    alignment_score = max(0.0, min(alignment_score, 1.0))
    return transport, alignment_score


def apply_transport(text_prototypes, transport):
    transported = text_prototypes @ transport
    return F.normalize(transported, dim=1)


def shrink_transport(text_prototypes, transported_prototypes, support, alignment_score):
    feature_dim = text_prototypes.size(1)
    shrinkage = float(support) / (float(support) + feature_dim)
    shrinkage *= float(alignment_score)
    blended = (1.0 - shrinkage) * text_prototypes + shrinkage * transported_prototypes
    return F.normalize(blended, dim=1)


def build_anchor_probs(anchor_logits, counts, prior_strength):
    del counts, prior_strength
    return F.softmax(anchor_logits, dim=1).clamp_min(EPS)


def compute_batch_marginal(anchor_probs, smoothing=1.0):
    num_classes = anchor_probs.size(1)
    marginal = anchor_probs.mean(dim=0)
    marginal = marginal + float(smoothing) / num_classes
    marginal = marginal / marginal.sum().clamp_min(EPS)
    return marginal.clamp_min(EPS)


def compute_batch_margin(anchor_probs):
    topk = anchor_probs.topk(min(2, anchor_probs.size(1)), dim=1).values
    if topk.size(1) == 1:
        return 1.0
    return float((topk[:, 0] - topk[:, 1]).mean().item())


def compute_batch_evidence(
    anchor_probs,
    W,
    margin_center=0.20,
    margin_scale=0.05,
    agreement_center=0.55,
    agreement_scale=0.08,
    npc_center=12.0,
    npc_scale=4.0,
):
    topk = anchor_probs.topk(min(2, anchor_probs.size(1)), dim=1).values
    if topk.size(1) == 1:
        mean_margin = anchor_probs.new_tensor(1.0)
    else:
        mean_margin = (topk[:, 0] - topk[:, 1]).mean()

    predicted_classes = anchor_probs.argmax(dim=1)
    unique_classes = max(predicted_classes.unique().numel(), 1)
    samples_per_predicted_class = anchor_probs.new_tensor(anchor_probs.size(0) / unique_classes)

    coalesced = W.coalesce()
    row_idx, col_idx = coalesced.indices()
    if row_idx.numel() == 0:
        neighbor_agreement = anchor_probs.new_tensor(0.0)
    else:
        neighbor_agreement = (predicted_classes[row_idx] == predicted_classes[col_idx]).float().mean()

    margin_score = torch.sigmoid((mean_margin - float(margin_center)) / max(float(margin_scale), EPS))
    agreement_score = torch.sigmoid((neighbor_agreement - float(agreement_center)) / max(float(agreement_scale), EPS))
    npc_score = torch.sigmoid((samples_per_predicted_class - float(npc_center)) / max(float(npc_scale), EPS))
    return float((margin_score * agreement_score * npc_score).clamp(0.0, 1.0).item())


def compute_samples_per_predicted_class(anchor_probs):
    predicted_classes = anchor_probs.argmax(dim=1)
    unique_classes = max(int(predicted_classes.unique().numel()), 1)
    return float(anchor_probs.size(0)) / unique_classes


def compute_sinkhorn_selector(
    anchor_probs,
    W,
    evidence_margin_center=0.20,
    evidence_margin_scale=0.05,
    evidence_agreement_center=0.55,
    evidence_agreement_scale=0.08,
    evidence_npc_center=12.0,
    evidence_npc_scale=4.0,
    sinkhorn_margin_center=0.45,
    sinkhorn_margin_scale=0.08,
):
    evidence = compute_batch_evidence(
        anchor_probs,
        W,
        margin_center=evidence_margin_center,
        margin_scale=evidence_margin_scale,
        agreement_center=evidence_agreement_center,
        agreement_scale=evidence_agreement_scale,
        npc_center=evidence_npc_center,
        npc_scale=evidence_npc_scale,
    )
    mean_margin = compute_batch_margin(anchor_probs)
    ambiguity = 1.0 / (
        1.0 + math.exp(
            (mean_margin - float(sinkhorn_margin_center)) / max(float(sinkhorn_margin_scale), EPS)
        )
    )
    return evidence * ambiguity


def sinkhorn_assignments(joint_logits, column_marginal, max_iter=25):
    num_samples, num_classes = joint_logits.shape
    log_kernel = joint_logits - joint_logits.max()

    column_targets = column_marginal.clamp_min(EPS)
    column_targets = column_targets / column_targets.sum().clamp_min(EPS)
    column_targets = column_targets * float(num_samples)
    log_column_targets = column_targets.log()

    log_u = joint_logits.new_zeros(num_samples)
    log_v = joint_logits.new_zeros(num_classes)

    for _ in range(max_iter):
        log_u = -torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_column_targets - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)

    assignments = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(EPS)
    return assignments.clamp_min(EPS)


def compute_marginal_regularization_strength(column_marginal, num_samples):
    effective_classes = effective_num_classes(column_marginal)
    num_classes = max(column_marginal.numel(), 1)
    sample_confidence = float(num_samples) / (float(num_samples) + num_classes)
    return float(effective_classes / num_classes) * sample_confidence


def compute_transport_trust(column_marginal):
    effective_classes = effective_num_classes(column_marginal)
    num_classes = max(column_marginal.numel(), 1)
    trust = 1.0 - (effective_classes / num_classes)
    return max(0.0, min(float(trust), 1.0))


def compute_anchor_weight(lambda_y_hat, column_marginal):
    effective_classes = effective_num_classes(column_marginal)
    num_classes = max(column_marginal.numel(), 1)
    coverage = effective_classes / num_classes
    return float(lambda_y_hat) * (1.0 + coverage)


def geometric_anchor_blend(base_anchor_probs, transported_anchor_probs, transport_trust):
    blend_weight = float(transport_trust)
    if blend_weight <= 0.0:
        return base_anchor_probs.clamp_min(EPS)
    if blend_weight >= 1.0:
        return transported_anchor_probs.clamp_min(EPS)

    log_base = base_anchor_probs.clamp_min(EPS).log()
    log_transported = transported_anchor_probs.clamp_min(EPS).log()
    blended = torch.exp((1.0 - blend_weight) * log_base + blend_weight * log_transported)
    blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(EPS)
    return blended.clamp_min(EPS)


def compute_sample_anchor_weights(
    base_anchor_probs,
    anchor_weight,
    anchor_ambiguity_gamma,
    margin_center=0.20,
    margin_scale=0.05,
):
    if anchor_ambiguity_gamma <= 0:
        return float(anchor_weight)

    topk = base_anchor_probs.topk(min(2, base_anchor_probs.size(1)), dim=1).values
    if topk.size(1) == 1:
        margin = torch.ones(base_anchor_probs.size(0), device=base_anchor_probs.device, dtype=base_anchor_probs.dtype)
    else:
        margin = topk[:, 0] - topk[:, 1]
    ambiguity = torch.sigmoid((float(margin_center) - margin) / max(float(margin_scale), EPS))
    return float(anchor_weight) * (1.0 + float(anchor_ambiguity_gamma) * ambiguity)


def compute_effective_max_iter(
    base_anchor_probs,
    W,
    requested_max_iter,
    coverage_threshold=0.90,
    shortcircuit_evidence=0.05,
    iter_npc_center=100.0,
    iter_npc_scale=30.0,
    evidence_margin_center=0.20,
    evidence_margin_scale=0.05,
    evidence_agreement_center=0.55,
    evidence_agreement_scale=0.08,
    evidence_npc_center=12.0,
    evidence_npc_scale=4.0,
):
    if requested_max_iter <= 0:
        return 0, False

    num_samples, num_classes = base_anchor_probs.shape
    predicted_classes = base_anchor_probs.argmax(dim=1)
    num_predicted_classes = max(int(predicted_classes.unique().numel()), 1)
    coverage = float(num_predicted_classes) / max(float(num_classes), 1.0)
    if coverage < float(coverage_threshold):
        return int(requested_max_iter), False

    evidence = compute_batch_evidence(
        base_anchor_probs,
        W,
        margin_center=evidence_margin_center,
        margin_scale=evidence_margin_scale,
        agreement_center=evidence_agreement_center,
        agreement_scale=evidence_agreement_scale,
        npc_center=evidence_npc_center,
        npc_scale=evidence_npc_scale,
    )
    samples_per_predicted_class = float(num_samples) / num_predicted_classes
    npc_gate = 1.0 / (
        1.0 + math.exp(
            -(samples_per_predicted_class - float(iter_npc_center)) / max(float(iter_npc_scale), EPS)
        )
    )
    iteration_fraction = evidence * npc_gate
    effective = int(round(requested_max_iter * max(0.0, min(iteration_fraction, 1.0))))
    shortcircuit_zero_shot = effective == 0 and evidence < float(shortcircuit_evidence)
    return max(0, min(int(requested_max_iter), effective)), shortcircuit_zero_shot


def update_assignments(
    likelihoods,
    anchor_probs,
    z,
    W,
    anchor_weight,
    lambda_laplacian,
    n_neighbors,
    sigma,
    column_marginal=None,
    use_exact_marginal=False,
    max_iter=5,
):
    log_anchor = anchor_probs.clamp_min(EPS).log()
    if isinstance(anchor_weight, torch.Tensor):
        anchor_weight_term = anchor_weight.to(log_anchor.device, dtype=log_anchor.dtype).unsqueeze(1)
    else:
        anchor_weight_term = float(anchor_weight)
    log_column_marginal = None
    marginal_strength = 0.0
    if column_marginal is not None:
        normalized_marginal = column_marginal.clamp_min(EPS)
        normalized_marginal = normalized_marginal / normalized_marginal.sum().clamp_min(EPS)
        log_column_marginal = normalized_marginal.log()
        marginal_strength = compute_marginal_regularization_strength(normalized_marginal, z.size(0))

    for _ in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian * (50 / (n_neighbors * 2)) * (W.T @ z + W @ z)

        sigma_log_sum = sigma.clamp_min(EPS).log().sum(dim=1)
        intermediate -= 0.5 * sigma_log_sum.unsqueeze(0)

        joint_logits = intermediate / 50.0 + anchor_weight_term * log_anchor
        if column_marginal is None:
            z = F.softmax(joint_logits, dim=1)
        elif use_exact_marginal:
            z = sinkhorn_assignments(joint_logits, normalized_marginal)
        else:
            current_marginal = z.mean(dim=0).clamp_min(EPS)
            marginal_bias = marginal_strength * (log_column_marginal - current_marginal.log())
            z = F.softmax(joint_logits + marginal_bias.unsqueeze(0), dim=1)

    return z


def compute_anchor_pseudocount(num_samples, pi, alpha):
    effective_classes = max(effective_num_classes(pi), 1.0)
    total_classes = max(float(pi.numel()), 1.0)
    effective_support = max(num_samples / effective_classes, 1.0)
    coverage = effective_classes / total_classes
    return float(alpha) * math.sqrt(effective_support) * (1.0 + coverage)


def JPTA_solver(
    query_features,
    query_labels,
    clip_prototypes,
    alpha=1.0,
    lambda_y_hat=1.0,
    lambda_laplacian=1.0,
    n_neighbors=3,
    max_iter=10,
    prior_strength=1.0,
    anchor_ambiguity_gamma=0.0,
    gate_margin_center=0.20,
    gate_margin_scale=0.05,
    template_bank_adaptation=False,
    prompt_bank_topk=64,
    prompt_bank_temperature=0.07,

    evidence_margin_center=0.20,
    evidence_margin_scale=0.05,
    evidence_agreement_center=0.55,
    evidence_agreement_scale=0.08,
    evidence_npc_center=12.0,
    evidence_npc_scale=4.0,

    sinkhorn_margin_center=0.45,
    sinkhorn_margin_scale=0.08,

    fallback_coverage=0.90,
    fallback_evidence=0.60,
    fallback_sinkhorn_max=0.35,
    fallback_npc_min=20.0,

    preserve_evidence=0.90,
    preserve_sinkhorn_max=0.05,
    preserve_npc_min=80.0,

    single_proj_evidence=0.70,
    single_proj_sinkhorn_min=0.50,
    single_proj_npc_min=400.0,

    iter_coverage_threshold=0.90,
    iter_shortcircuit_evidence=0.05,
    iter_npc_center=100.0,
    iter_npc_scale=30.0,

    return_cpu=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else query_features.device)

    query_labels = query_labels.to(device).float()
    query_features = query_features.to(device).float()
    clip_prototypes, template_bank_adaptation = select_auto_prompt_bank(
        clip_prototypes,
        query_features,
        template_bank_adaptation,
    )
    clip_prototypes = clip_prototypes.to(device).float()
    if template_bank_adaptation:
        clip_prototypes = adapt_prompt_bank(
            clip_prototypes,
            query_features,
            topk_per_class=prompt_bank_topk,
            temperature=prompt_bank_temperature,
        )
    clip_prototypes = canonicalize_clip_prototypes(clip_prototypes)

    text_prototypes = clip_prototypes.squeeze(0).transpose(0, 1)
    num_samples = query_features.size(0)
    num_classes = text_prototypes.size(0)

    pi = query_features.new_full((num_classes,), 1.0 / num_classes)
    anchor_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    base_anchor_probs = F.softmax(anchor_logits, dim=1)
    zero_shot_marginal = compute_batch_marginal(base_anchor_probs, smoothing=prior_strength)
    transport_trust = compute_transport_trust(zero_shot_marginal)
    anchor_weight = compute_anchor_weight(lambda_y_hat, zero_shot_marginal)
    sample_anchor_weight = compute_sample_anchor_weights(
        base_anchor_probs,
        anchor_weight=anchor_weight,
        anchor_ambiguity_gamma=anchor_ambiguity_gamma,
        margin_center=gate_margin_center,
        margin_scale=gate_margin_scale,
    )
    z = base_anchor_probs.clone()

    W = build_affinity_matrix(query_features.float(), n_neighbors)
    evidence = compute_batch_evidence(
        base_anchor_probs,
        W,
        margin_center=evidence_margin_center,
        margin_scale=evidence_margin_scale,
        agreement_center=evidence_agreement_center,
        agreement_scale=evidence_agreement_scale,
        npc_center=evidence_npc_center,
        npc_scale=evidence_npc_scale,
    )

    sinkhorn_selector = compute_sinkhorn_selector(
        base_anchor_probs,
        W,
        evidence_margin_center=evidence_margin_center,
        evidence_margin_scale=evidence_margin_scale,
        evidence_agreement_center=evidence_agreement_center,
        evidence_agreement_scale=evidence_agreement_scale,
        evidence_npc_center=evidence_npc_center,
        evidence_npc_scale=evidence_npc_scale,
        sinkhorn_margin_center=sinkhorn_margin_center,
        sinkhorn_margin_scale=sinkhorn_margin_scale,
    )
    samples_per_predicted_class = compute_samples_per_predicted_class(base_anchor_probs)
    predicted_coverage = float(base_anchor_probs.argmax(dim=1).unique().numel()) / max(float(num_classes), 1.0)
    stata_fallback_mode = (
        predicted_coverage >= float(fallback_coverage)
        and evidence >= float(fallback_evidence)
        and sinkhorn_selector <= float(fallback_sinkhorn_max)
        and samples_per_predicted_class >= float(fallback_npc_min)
    )
    if stata_fallback_mode:
        return StatA_solver(
            query_features,
            query_labels,
            clip_prototypes,
            alpha=1.0,
            soft_beta=False,
            lambda_y_hat=lambda_y_hat,
            lambda_laplacian=lambda_laplacian,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            return_cpu=return_cpu,
        )
    anchor_relaxation = 1.0 - 0.75 * sinkhorn_selector
    use_exact_marginal = sinkhorn_selector >= 0.5
    effective_max_iter, shortcircuit_zero_shot = compute_effective_max_iter(
        base_anchor_probs,
        W,
        max_iter,
        coverage_threshold=iter_coverage_threshold,
        shortcircuit_evidence=iter_shortcircuit_evidence,
        iter_npc_center=iter_npc_center,
        iter_npc_scale=iter_npc_scale,
        evidence_margin_center=evidence_margin_center,
        evidence_margin_scale=evidence_margin_scale,
        evidence_agreement_center=evidence_agreement_center,
        evidence_agreement_scale=evidence_agreement_scale,
        evidence_npc_center=evidence_npc_center,
        evidence_npc_scale=evidence_npc_scale,
    )
    adapter = None
    last_anchor_probs = base_anchor_probs
    preserve_anchor_output = (
        evidence >= float(preserve_evidence)
        and sinkhorn_selector <= float(preserve_sinkhorn_max)
        and samples_per_predicted_class >= float(preserve_npc_min)
    )

    single_projection_mode = (
        evidence >= float(single_proj_evidence)
        and sinkhorn_selector >= float(single_proj_sinkhorn_min)
        and samples_per_predicted_class >= float(single_proj_npc_min)
    )

    if preserve_anchor_output:
        return maybe_move_to_cpu(base_anchor_probs, return_cpu=return_cpu), maybe_move_to_cpu(base_anchor_probs, return_cpu=return_cpu)

    if shortcircuit_zero_shot:
        return maybe_move_to_cpu(base_anchor_probs, return_cpu=return_cpu), maybe_move_to_cpu(base_anchor_probs, return_cpu=return_cpu)

    if single_projection_mode:
        effective_max_iter = 1

    for iteration in range(effective_max_iter + 1):
        pi, counts = class_posterior_mean(z, prior_strength=prior_strength)
        image_prototypes, _ = compute_soft_class_means(query_features, z)
        transport_weights = compute_transport_weights(z)
        transport, alignment_score = estimate_orthogonal_transport(text_prototypes, image_prototypes, transport_weights)
        transported_matrix = apply_transport(text_prototypes, transport)
        transported_matrix = shrink_transport(
            text_prototypes,
            transported_matrix,
            transport_weights.sum(),
            alignment_score,
        )
        transported_prototypes = transported_matrix.transpose(0, 1).unsqueeze(0)

        anchor_logits = get_zero_shot_logits(query_features, query_labels, transported_prototypes)
        transported_anchor_probs = build_anchor_probs(anchor_logits, counts, prior_strength)
        anchor_probs = geometric_anchor_blend(base_anchor_probs, transported_anchor_probs, transport_trust)
        last_anchor_probs = anchor_probs

        init_prototypes = transported_prototypes.permute(2, 0, 1)
        init_covariance = init_cov(init_prototypes, query_features, anchor_probs)
        init_covariance = init_covariance.unsqueeze(0).repeat(num_classes, 1)

        if adapter is None:
            adapter = Gaussian(mu=init_prototypes.clone(), cov=init_covariance.clone()).to(device)

        likelihoods = adapter(query_features, no_exp=True)
        z = update_assignments(
            likelihoods,
            anchor_probs,
            z,
            W,
            anchor_weight=sample_anchor_weight * anchor_relaxation,
            lambda_laplacian=lambda_laplacian,
            n_neighbors=n_neighbors,
            sigma=adapter.cov,
            column_marginal=zero_shot_marginal,
            use_exact_marginal=use_exact_marginal,
        )

        if iteration == effective_max_iter:
            break

        pi, counts = class_posterior_mean(z, prior_strength=prior_strength)
        anchor_pseudocount = compute_anchor_pseudocount(num_samples, pi, alpha) * anchor_relaxation
        beta = counts / (counts + anchor_pseudocount + EPS)
        mu = update_mu(adapter, query_features, z, beta, init_prototypes)
        adapter.set_mu(mu)
        cov = update_cov(adapter, query_features, z, beta, init_prototypes, init_covariance)
        adapter.set_cov(cov)

    return maybe_move_to_cpu(last_anchor_probs, return_cpu=return_cpu), maybe_move_to_cpu(z, return_cpu=return_cpu)
