"""JPTA evaluation entry point for the 11 benchmark datasets.

The solver, feature extraction, samplers, and averaging follow the experiment
implementation in JPTA-Optimized. Only the JPTA evaluation path is exposed.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import clip
from datasets import dataset_list, get_all_dataloaders
from prompt_banks import get_template_bank
from sampler import BatchSampler, OnlineSampler
from solvers import JPTA_solver
from utils import (
    cls_acc,
    compute_calibration_metrics,
    compute_zero_shot_logits,
    get_all_features,
)


BACKBONES = {
    "rn50": "RN50",
    "rn101": "RN101",
    "vit_b16": "ViT-B/16",
    "vit_b32": "ViT-B/32",
    "vit_l14": "ViT-L/14",
}


def get_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate JPTA on one dataset and regime")
    parser.add_argument("--root_path", required=True, help="Parent folder of the 11 datasets")
    parser.add_argument("--dataset", required=True, choices=sorted(dataset_list))
    parser.add_argument("--backbone", default="vit_b16", choices=sorted(BACKBONES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_tasks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--num_class_eff", type=int, default=None)
    parser.add_argument("--num_class_eff_min", type=int, default=None)
    parser.add_argument("--num_class_eff_max", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--report-calibration", action="store_true")

    # Defaults and flag names match the JPTA branch of the experiment code.
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-y-hat", dest="lambda_y_hat", type=float, default=1.0)
    parser.add_argument("--lambda_laplacian", type=float, default=1.0)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--anchor-ambiguity-gamma", type=float, default=0.0)
    parser.add_argument("--gate-margin-center", type=float, default=0.20)
    parser.add_argument("--gate-margin-scale", type=float, default=0.05)
    parser.add_argument("--template-bank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt-bank-strategy", default="legacy", choices=["legacy", "published", "hybrid", "auto"])
    parser.add_argument("--prompt-bank-topk", type=int, default=64)
    parser.add_argument("--prompt-bank-temperature", type=float, default=0.07)
    parser.add_argument("--jpta-n-neighbors", type=int, default=3)
    parser.add_argument("--jpta-max-iter", type=int, default=10)
    parser.add_argument("--jpta-evidence-margin-center", type=float, default=0.20)
    parser.add_argument("--jpta-evidence-margin-scale", type=float, default=0.05)
    parser.add_argument("--jpta-evidence-agreement-center", type=float, default=0.55)
    parser.add_argument("--jpta-evidence-agreement-scale", type=float, default=0.08)
    parser.add_argument("--jpta-evidence-npc-center", type=float, default=12.0)
    parser.add_argument("--jpta-evidence-npc-scale", type=float, default=4.0)
    parser.add_argument("--jpta-sinkhorn-margin-center", type=float, default=0.45)
    parser.add_argument("--jpta-sinkhorn-margin-scale", type=float, default=0.08)
    parser.add_argument("--jpta-fallback-coverage", type=float, default=0.90)
    parser.add_argument("--jpta-fallback-evidence", type=float, default=0.60)
    parser.add_argument("--jpta-fallback-sinkhorn-max", type=float, default=0.35)
    parser.add_argument("--jpta-fallback-npc-min", type=float, default=20.0)
    parser.add_argument("--jpta-preserve-evidence", type=float, default=0.90)
    parser.add_argument("--jpta-preserve-sinkhorn-max", type=float, default=0.05)
    parser.add_argument("--jpta-preserve-npc-min", type=float, default=80.0)
    parser.add_argument("--jpta-single-proj-evidence", type=float, default=0.70)
    parser.add_argument("--jpta-single-proj-sinkhorn-min", type=float, default=0.50)
    parser.add_argument("--jpta-single-proj-npc-min", type=float, default=400.0)
    parser.add_argument("--jpta-iter-coverage-threshold", type=float, default=0.90)
    parser.add_argument("--jpta-iter-shortcircuit-evidence", type=float, default=0.05)
    parser.add_argument("--jpta-iter-npc-center", type=float, default=100.0)
    parser.add_argument("--jpta-iter-npc-scale", type=float, default=30.0)
    return parser.parse_args(argv)


def jpta_hyperparameters(args):
    return {
        "alpha": args.alpha,
        "lambda_y_hat": args.lambda_y_hat,
        "lambda_laplacian": args.lambda_laplacian,
        "n_neighbors": args.jpta_n_neighbors,
        "max_iter": args.jpta_max_iter,
        "prior_strength": args.prior_strength,
        "anchor_ambiguity_gamma": args.anchor_ambiguity_gamma,
        "gate_margin_center": args.gate_margin_center,
        "gate_margin_scale": args.gate_margin_scale,
        "template_bank_adaptation": False if args.template_bank is None else args.template_bank,
        "prompt_bank_topk": args.prompt_bank_topk,
        "prompt_bank_temperature": args.prompt_bank_temperature,
        "evidence_margin_center": args.jpta_evidence_margin_center,
        "evidence_margin_scale": args.jpta_evidence_margin_scale,
        "evidence_agreement_center": args.jpta_evidence_agreement_center,
        "evidence_agreement_scale": args.jpta_evidence_agreement_scale,
        "evidence_npc_center": args.jpta_evidence_npc_center,
        "evidence_npc_scale": args.jpta_evidence_npc_scale,
        "sinkhorn_margin_center": args.jpta_sinkhorn_margin_center,
        "sinkhorn_margin_scale": args.jpta_sinkhorn_margin_scale,
        "fallback_coverage": args.jpta_fallback_coverage,
        "fallback_evidence": args.jpta_fallback_evidence,
        "fallback_sinkhorn_max": args.jpta_fallback_sinkhorn_max,
        "fallback_npc_min": args.jpta_fallback_npc_min,
        "preserve_evidence": args.jpta_preserve_evidence,
        "preserve_sinkhorn_max": args.jpta_preserve_sinkhorn_max,
        "preserve_npc_min": args.jpta_preserve_npc_min,
        "single_proj_evidence": args.jpta_single_proj_evidence,
        "single_proj_sinkhorn_min": args.jpta_single_proj_sinkhorn_min,
        "single_proj_npc_min": args.jpta_single_proj_npc_min,
        "iter_coverage_threshold": args.jpta_iter_coverage_threshold,
        "iter_shortcircuit_evidence": args.jpta_iter_shortcircuit_evidence,
        "iter_npc_center": args.jpta_iter_npc_center,
        "iter_npc_scale": args.jpta_iter_npc_scale,
    }


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_apply_template_bank(args, dataset):
    if not args.template_bank or args.prompt_bank_strategy == "auto":
        return
    if args.dataset.startswith("imagenet") and hasattr(dataset, "custom_templates"):
        dataset.template = dataset.custom_templates
        return
    template_bank = get_template_bank(args.dataset, dataset=dataset, strategy=args.prompt_bank_strategy)
    if template_bank:
        dataset.template = template_bank


def average_metrics(totals, count):
    return {key: value / count for key, value in totals.items()}


def add_metrics(totals, output, labels):
    for key, value in compute_calibration_metrics(output, labels).items():
        totals[key] += value


def empty_metrics():
    return {"ece": 0.0, "ada_ece": 0.0, "brier": 0.0, "nll": 0.0}


def evaluate(args):
    set_random_seed(args.seed)
    args.method = "JPTA"
    if not args.cache_dir:
        args.cache_dir = os.path.join("./caches", args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)

    clip_model, preprocess = clip.load(BACKBONES[args.backbone])
    clip_model.eval()
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess)
    maybe_apply_template_bank(args, dataset)
    features, labels, prototypes = get_all_features(args, test_loader, dataset, clip_model)
    clip_model.to("cpu")

    hp = jpta_hyperparameters(args)
    final_accuracy = 0.0
    zero_accuracy = 0.0
    final_metrics = empty_metrics()
    zero_metrics = empty_metrics()
    tasks_run = 0

    if not args.online:
        sampler = BatchSampler(features, labels, args.batch_size, args.num_class_eff,
                               args.num_class_eff_min, args.num_class_eff_max)
        for _ in tqdm(range(args.n_tasks)):
            indices = sampler.generate_indices()
            if indices is None:
                break
            _, preds = JPTA_solver(features[indices, :], labels[indices], prototypes, **hp)
            preds_zs = compute_zero_shot_logits(features[indices, :], prototypes)
            final_accuracy += cls_acc(preds, labels[indices])
            zero_accuracy += cls_acc(preds_zs, labels[indices])
            if args.report_calibration:
                add_metrics(final_metrics, preds, labels[indices])
                add_metrics(zero_metrics, preds_zs, labels[indices])
            tasks_run += 1
    else:
        for _ in tqdm(range(args.n_tasks)):
            num_batch = features.shape[0] // args.batch_size
            num_slots = min(num_batch, len(torch.unique(labels)))
            sampler = OnlineSampler(features, labels, args.gamma, num_slots, args.batch_size)
            indices = sampler.generate_indices()
            task_accuracy = 0.0
            task_zero_accuracy = 0.0
            task_final_metrics = empty_metrics()
            task_zero_metrics = empty_metrics()
            batch_count = 0
            while indices is not None:
                _, preds = JPTA_solver(features[indices, :], labels[indices], prototypes, **hp)
                preds_zs = compute_zero_shot_logits(features[indices, :], prototypes)
                task_accuracy += cls_acc(preds, labels[indices])
                task_zero_accuracy += cls_acc(preds_zs, labels[indices])
                if args.report_calibration:
                    add_metrics(task_final_metrics, preds, labels[indices])
                    add_metrics(task_zero_metrics, preds_zs, labels[indices])
                batch_count += 1
                indices = sampler.generate_indices()
            if batch_count == 0:
                continue
            final_accuracy += task_accuracy / batch_count
            zero_accuracy += task_zero_accuracy / batch_count
            if args.report_calibration:
                for key in final_metrics:
                    final_metrics[key] += task_final_metrics[key] / batch_count
                    zero_metrics[key] += task_zero_metrics[key] / batch_count
            tasks_run += 1

    if tasks_run == 0:
        raise RuntimeError("The sampler produced no tasks; check the dataset and regime")
    result = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "online": args.online,
        "gamma": args.gamma if args.online else None,
        "num_class_eff_min": args.num_class_eff_min,
        "num_class_eff_max": args.num_class_eff_max,
        "tasks_run": tasks_run,
        "zero_shot_accuracy": zero_accuracy / tasks_run,
        "final_accuracy": final_accuracy / tasks_run,
        "hyperparameters": hp,
    }
    if args.report_calibration:
        result["zero_shot_calibration"] = average_metrics(zero_metrics, tasks_run)
        result["final_calibration"] = average_metrics(final_metrics, tasks_run)
    return result


def main(argv=None):
    args = get_arguments(argv)
    result = evaluate(args)
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
