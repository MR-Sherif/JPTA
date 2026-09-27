"""Run JPTA on the 11 datasets, seven batch regimes, and four online regimes."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "best_hyperparameters.json"
DATASETS = (
    "imagenet", "sun397", "fgvc", "eurosat", "stanford_cars", "food101",
    "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101",
)
REGIMES = {
    "very_low_b64": ("batch", 64, 1, 4),
    "low_b64": ("batch", 64, 2, 10),
    "medium_b64": ("batch", 64, 5, 25),
    "medium_b1000": ("batch", 1000, 5, 25),
    "high_b1000": ("batch", 1000, 25, 50),
    "very_high_b1000": ("batch", 1000, 50, 100),
    "all_classes": ("batch", -1, None, None),
    "online_low": ("online", 128, 0.1),
    "online_medium": ("online", 128, 0.01),
    "online_high": ("online", 128, 0.001),
    "online_separate": ("online", 128, -1.0),
}
TUNABLE_FLAGS = frozenset({
    "alpha", "lambda-y-hat", "lambda_laplacian", "prior-strength",
    "anchor-ambiguity-gamma", "template-bank", "prompt-bank-strategy",
    "prompt-bank-topk", "prompt-bank-temperature", "jpta-n-neighbors",
    "jpta-max-iter", "jpta-evidence-margin-center",
    "jpta-evidence-margin-scale", "jpta-evidence-agreement-center",
    "jpta-evidence-agreement-scale", "jpta-evidence-npc-center",
    "jpta-evidence-npc-scale", "jpta-sinkhorn-margin-center",
    "jpta-sinkhorn-margin-scale", "jpta-fallback-coverage",
    "jpta-fallback-evidence", "jpta-fallback-sinkhorn-max",
    "jpta-fallback-npc-min", "jpta-preserve-evidence",
    "jpta-preserve-sinkhorn-max", "jpta-preserve-npc-min",
    "jpta-single-proj-evidence", "jpta-single-proj-sinkhorn-min",
    "jpta-single-proj-npc-min", "jpta-iter-coverage-threshold",
    "jpta-iter-shortcircuit-evidence", "jpta-iter-npc-center",
    "jpta-iter-npc-scale",
})
INTEGER_FLAGS = frozenset({"prompt-bank-topk", "jpta-n-neighbors", "jpta-max-iter"})


def load_best_configs(path=CONFIG_PATH):
    """Load the pinned settings and fail before launching if any run is uncovered."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark config schema: {path}")
    families = data.get("regime_families", {})
    datasets = data.get("datasets", {})
    if set(families) != set(REGIMES) or set(datasets) != set(DATASETS):
        raise ValueError("Benchmark config must cover every dataset and regime")
    for dataset in DATASETS:
        for regime in REGIMES:
            family = families[regime]
            entry = datasets[dataset].get(family)
            if not isinstance(entry, dict) or not entry.get("hyperparameters"):
                raise ValueError(f"Missing benchmark settings: {dataset} / {regime}")
            for flag, value in entry["hyperparameters"].items():
                if flag not in TUNABLE_FLAGS:
                    raise ValueError(f"Unknown benchmark setting: {flag}")
                if flag == "template-bank":
                    valid = isinstance(value, bool)
                elif flag == "prompt-bank-strategy":
                    valid = value in {"legacy", "published", "hybrid", "auto"}
                elif flag in INTEGER_FLAGS:
                    minimum = 0 if flag == "jpta-max-iter" else 1
                    valid = isinstance(value, int) and not isinstance(value, bool) and value >= minimum
                else:
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                if not valid:
                    raise ValueError(f"Invalid benchmark setting: {dataset} / {family} / {flag}")
    return data


def tuned_arguments(configs, dataset, regime):
    family = configs["regime_families"][regime]
    settings = configs["datasets"][dataset][family]["hyperparameters"]
    arguments = []
    for flag, value in settings.items():
        if flag == "template-bank":
            arguments.append("--template-bank" if value else "--no-template-bank")
        else:
            arguments.extend((f"--{flag}", str(value)))
    return arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=tuple(REGIMES))
    parser.add_argument("--backbone", default="vit_b16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-tasks", type=int, default=1000)
    parser.add_argument("--online-tasks", type=int, default=100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_command(args, dataset, regime, output_file, configs):
    protocol = REGIMES[regime]
    command = [
        args.python, str(ROOT / "main.py"), "--root_path", str(args.data_root),
        "--dataset", dataset, "--backbone", args.backbone,
        "--seed", str(args.seed), "--batch_size", str(protocol[1]),
        "--report-calibration",
        "--output-json", str(output_file),
    ]
    command += tuned_arguments(configs, dataset, regime)
    if protocol[0] == "batch":
        command += ["--n_tasks", str(1 if regime == "all_classes" else args.batch_tasks)]
        if protocol[2] is not None:
            command += ["--num_class_eff_min", str(protocol[2]),
                        "--num_class_eff_max", str(protocol[3])]
    else:
        command += ["--online", "--gamma", str(protocol[2]),
                    "--n_tasks", str(args.online_tasks)]
    return command


def write_summary(results_dir, datasets, regimes):
    rows = []
    for dataset in datasets:
        for regime in regimes:
            result_file = results_dir / dataset / f"{regime}.json"
            if not result_file.exists():
                continue
            record = json.loads(result_file.read_text(encoding="utf-8"))
            rows.append({
                "dataset": dataset,
                "regime": regime,
                "tasks_run": record["tasks_run"],
                "zero_shot_accuracy": record["zero_shot_accuracy"],
                "jpta_accuracy": record["final_accuracy"],
                "gain": record["final_accuracy"] - record["zero_shot_accuracy"],
            })
    if rows:
        summary_file = results_dir / "summary.csv"
        with summary_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary: {summary_file}")


def main(argv=None):
    args = parse_args(argv)
    configs = load_best_configs()
    results_dir = args.results_dir.resolve()
    if not args.dry_run and not args.data_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {args.data_root}")
    for dataset in args.datasets:
        for regime in args.regimes:
            output_file = results_dir / dataset / f"{regime}.json"
            command = build_command(args, dataset, regime, output_file, configs)
            print(f"[{dataset} / {regime}] {' '.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        write_summary(results_dir, args.datasets, args.regimes)


if __name__ == "__main__":
    main()
