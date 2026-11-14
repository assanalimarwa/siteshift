"""
Threshold analysis on the 47 patients: tile-mode re-inference saving uint8 tumour probabilities (evaluate.py --save-prob,
new eval_<tag>_test_tile_prob folders) for the site U-Net CV (5 folds, each on its test patients), the all-public U-Net
and the Karolinska 3,814 U-Net (all 47 patients), then eval/threshold47.py on the CPU; resumable:

    python run_threshold47.py [--dry-run]

Probabilities: $SITESHIFT_PRED/prob47/<run with / as _>/ours47_tile. Tables: results/cv47/threshold_summary.md,
threshold_by_patient.csv, threshold_curve.csv (definitions in eval/threshold47.py).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse

from run_fewshot import DATA, TILE, run

RUNS = [(f"cv47/imagenet/fold{k}", f"cv47/fold{k}.json") for k in range(5)] + \
       [(r, "cv47/all47.json") for r in ["all_public_unet", "karolinska_all_unet"]]


def plan():
    log = P.RUNS / "threshold47.log"
    steps = []
    for r, split in RUNS:
        dst = P.PRED / "prob47" / r.replace("/", "_") / "ours47_tile"
        steps.append((f"{r}: 47-patient tile probabilities", (dst / "done.json").exists(),
                      ["evaluate.py", "--name", r, "--split-file", split, *DATA, *TILE, "--save-prob", str(dst)], log))
    return steps + [(f"threshold47.py {c}", False, ["threshold47.py", c], log) for c in ["sweep", "binarize", "protocol", "report"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(plan(), ap.parse_args().dry_run, "results/cv47/threshold_summary.md")
