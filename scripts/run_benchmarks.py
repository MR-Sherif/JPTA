"""Run JPTA on the 11 datasets, seven batch regimes, and four online regimes."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def build_command(args, dataset, regime, output_file):
    protocol = REGIMES[regime]
    command = [
        args.python, str(ROOT / "main.py"), "--root_path", str(args.data_root),
        "--dataset", dataset, "--backbone", args.backbone,
        "--seed", str(args.seed), "--batch_size", str(protocol[1]),
        "--alpha", "10", "--template-bank", "--report-calibration",
        "--output-json", str(output_file),
    ]
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
    results_dir = args.results_dir.resolve()
    if not args.dry_run and not args.data_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {args.data_root}")
    for dataset in args.datasets:
        for regime in args.regimes:
            output_file = results_dir / dataset / f"{regime}.json"
            command = build_command(args, dataset, regime, output_file)
            print(f"[{dataset} / {regime}] {' '.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        write_summary(results_dir, args.datasets, args.regimes)


if __name__ == "__main__":
    main()
