# JPTA: benchmark code (NeurIPS 2026)

This repository contains the JPTA evaluation code for the 11 datasets in the
paper. It covers the seven batch regimes and four online regimes. The release
contains the JPTA solver and the StatA routines used by its fallback, the CLIP
feature pipeline, dataset loaders, and a benchmark launcher. It does not
include the experimental sweeps, ablations, cached features, or datasets.

## Setup

Use Python 3.10 and a CUDA-capable GPU. Install PyTorch for the CUDA version on
your machine, then install the remaining packages from `requirements.txt`.
The experiment environment used PyTorch 2.0.1 and torchvision 0.15.2; those
versions are listed in the requirements file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the datasets as described in [DATASETS.md](DATASETS.md) and place
their folders under one parent directory, here called `$DATA_ROOT`. The 11
dataset identifiers are `imagenet`, `sun397`, `fgvc`, `eurosat`,
`stanford_cars`, `food101`, `oxford_pets`, `oxford_flowers`, `caltech101`,
`dtd`, and `ucf101`. On first use, the CLIP loader downloads ViT-B/16 weights
and the code computes image features. Feature caches are written under
`caches/` and reused on subsequent runs.

## Reproduce the benchmark grid

```bash
python scripts/run_benchmarks.py --data-root "$DATA_ROOT"
```

The launcher evaluates JPTA on every dataset with the paper's ViT-B/16
backbone, seed 1, `--alpha 10`, and the template bank. It runs 1,000 sampled
tasks per batch regime, one all-classes pass, and 100 sampled online streams.
The seven batch regimes are:

| Regime | Batch size | Effective classes |
| --- | ---: | ---: |
| Very low | 64 | 1–4 |
| Low | 64 | 2–10 |
| Medium | 64 | 5–25 |
| Medium | 1,000 | 5–25 |
| High | 1,000 | 25–50 |
| Very high | 1,000 | 50–100 |
| All classes | Entire test set | All |

The online regimes use batch size 128 and Dirichlet stream parameters
`gamma=0.1`, `0.01`, `0.001`, and `-1` (classes appear separately).

Each evaluation writes a JSON file under `results/<dataset>/<regime>.json`.
The launcher also writes `results/summary.csv` with JPTA accuracy, CLIP
zero-shot accuracy, and the gain. Results are averages over sampled tasks or
streams, following the original experiment code.

To run a smaller subset, for example:

```bash
python scripts/run_benchmarks.py --data-root "$DATA_ROOT" \
  --datasets eurosat oxford_pets \
  --regimes very_low_b64 high_b1000 all_classes online_low
```

Use `--dry-run` to inspect commands without loading data. A single setting
can also be run through `main.py`, for example:

```bash
python main.py --root_path "$DATA_ROOT" --dataset eurosat \
  --backbone vit_b16 --batch_size 64 --num_class_eff_min 1 \
  --num_class_eff_max 4 --n_tasks 1000 --seed 1 \
  --alpha 10 --template-bank --report-calibration
```

Small numerical differences can arise from CUDA operations or dataset and
dependency versions. The benchmark itself requires the image datasets and a
GPU; `--dry-run` validates the full command grid without either.

## Source and license

The JPTA solver and experiment components were selected from the optimized
JPTA research workspace used for the paper. The dataset and CLIP evaluation
framework builds on the StatA implementation by Zanella et al. The repository
retains the source workspace's AGPL-3.0 license in [LICENSE](LICENSE).
The bundled CLIP code originates from [OpenAI CLIP](https://github.com/openai/CLIP)
and its MIT license is included in [clip/LICENSE](clip/LICENSE).
