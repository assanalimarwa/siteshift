"""
Label-convention control on the 47 patients: the baseline U-Net recipe (train.py defaults + --epoch-tiles 60000
--limit-val 20000, seed 0, the data/cv47 folds) trained on dilated or eroded outlines and tested against the ORIGINAL
outlines; resumable:

    python run_labels47.py [--dry-run]

Steps: eval/labels47.py make (the transformed masks, $SITESHIFT_DATA/labels47/<convention>/<slide>/masks/), then per
convention (dilate50, erode50) and fold: train with --mask-root (train and val masks of the fold transformed), test tile
mode on the fold's test patients with the original masks (protocol-format predictions saved), then eval/labels47.py
protocol and report (results/cv47/labels_summary.md, labels_by_patient.csv). Configs: train/configs/cv47_labels__*.json.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse

from run_fewshot import eval_step, run, train_step

LAB = P.DATA_ROOT / "labels47"
CV_EXTRA = ["--epoch-tiles", "60000", "--limit-val", "20000"]


def plan():
    steps = [("transformed labels dilate50 / erode50 (CPU)", (LAB / "MADE.json").exists(), ["labels47.py", "make"],
              P.RUNS / "labels47.log")]
    for k in range(5):
        for conv in ["dilate50", "erode50"]:
            r = f"cv47_labels/{conv}/fold{k}"
            steps.append(train_step(f"{conv} fold {k}: train", r, ["--split", f"cv47/fold{k}.json", "--mask-root", str(LAB / conv), *CV_EXTRA]))
            steps.append(eval_step(f"{conv} fold {k}: test tile (original labels)", r, f"cv47/fold{k}.json", f"fold{k}"))
    steps += [("protocol (CPU)", False, ["labels47.py", "protocol"], P.RUNS / "labels47.log"),
              ("report", False, ["labels47.py", "report"], P.RUNS / "labels47.log")]
    return steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(plan(), ap.parse_args().dry_run, "results/cv47/labels_summary.md")
