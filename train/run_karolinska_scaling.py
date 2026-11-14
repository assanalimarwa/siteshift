"""
Karolinska scaling curve: the same U-Net recipe trained on 20, 155, 419 and 1,000 Karolinska slides,
next to the 3,814-slide model (runs/karolinska_all_unet); resumable:

    python run_karolinska_scaling.py [--sizes 419 1000] [--dry-run]

Nested slide sets: 20 and 155 are random subsets (seed 0) of the 419 train slides of the 600-slide run,
419 is that set itself, 1,000 is it plus 581 random others of the 3,814. Same schedule as the all-slide
run: epochs of 60,000 tile draws (a small set repeats), early stopping on the same 20,000 tiles of the
546 val slides, at most 40 epochs. (The first 419-slide model, runs/karolinska_unet, had an older
setup: JPEG quality 95 tiles, its own 60 val slides; runs/karolinska_n419_unet replaces it here.)
Each model is scored on our 17 patients (tile and full mode) and on the 1,092 Karolinska test slides
(tile mode). Table: karolinska_all_report.py.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = P.INDEX
KA = ["--data", str(P.DATA_ROOT) + "/karolinska_all", "--index", "karolinska_all_tiles.csv"]
MODES = {"tile": ["--context", "0", "--no-tta"], "ctx256_blend_tta": []}
SIZES = [20, 155, 419, 1000]


def split_file(n):
    """karolinska_n<n>_split.json: karolinska_all_split.json with only n train slides (nested sets)."""
    path = HERE / f"karolinska_n{n}_split.json"
    if not path.exists():
        s = json.loads((HERE / "karolinska_all_split.json").read_text())
        pool = sorted(json.loads((HERE / "karolinska_split.json").read_text())["splits"]["train"])
        if n <= len(pool):
            chosen = np.random.default_rng(0).permutation(pool)[:n].tolist()
            s["scaling"] = f"{n} of the 419 train slides of karolinska_split.json (seed 0, nested)"
        else:
            rest = sorted(set(s["splits"]["train"]) - set(pool))
            chosen = pool + np.random.default_rng(0).permutation(rest)[:n - len(pool)].tolist()
            s["scaling"] = f"the 419 train slides of karolinska_split.json + {n - len(pool)} random others (seed 0)"
        s["splits"] = dict(s["splits"], train=sorted(chosen))
        path.write_text(json.dumps(s, indent=2))
    return path.name


def plan(sizes=SIZES):
    steps = []
    for n in sizes:
        name = f"karolinska_n{n}_unet"
        run, split = P.RUNS / name, split_file(n)
        trained = (run / "TRAINED").exists()
        steps.append((f"train U-Net on {n} Karolinska slides", trained,
                      ["train.py", "--name", name, *KA, "--split", split, "--epoch-tiles", "60000", "--limit-val", "20000",
                       *(["--resume"] if (run / "last.pt").exists() and not trained else [])], run / "train.log"))
        for mode, extra in MODES.items():
            steps.append((f"{n}-slide U-Net on our 17 patients, {mode}", (run / f"eval_all_test_{mode}" / "summary.json").exists(),
                          ["evaluate.py", "--name", name, "--split-file", "cv/all.json", *extra], run / "eval.log"))
        steps.append((f"{n}-slide U-Net on all 1,092 Karolinska test slides, tile",
                      (run / "eval_karolinska_all_split_test_tile" / "summary.json").exists(),
                      ["evaluate.py", "--name", name, *KA, "--split-file", "karolinska_all_split.json", "--split", "test",
                       "--no-pictures", *MODES["tile"]], run / "eval.log"))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=SIZES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = plan(a.sizes)
    if a.dry_run:
        import pandas as pd
        labels = pd.read_csv(str(P.PANDA_RAW / "Karolinska" / "labels.csv")).set_index("image_id")
        for n in a.sizes:
            train = json.loads((HERE / split_file(n)).read_text())["splits"]["train"]
            print(f"{n} slides: ISUP {labels.loc[train, 'isup_grade'].value_counts().sort_index().to_dict()}")
        for label, done, _, _ in steps:
            print(f"{'done' if done else 'todo'}  {label}")
        return
    for label, done, cmd, log in steps:
        if done:
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{time.strftime('%m-%d %H:%M')}] {label}", flush=True)
        t0 = time.time()
        with log.open("a") as f:
            code = subprocess.run([sys.executable, str(P.script(cmd[0])), *cmd[1:]], cwd=P.INDEX, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"{label} failed (exit {code}); see {log}")
        if cmd[0] == "train.py":
            (log.parent / "TRAINED").write_text(time.strftime("%Y-%m-%d %H:%M"))
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done; next: python karolinska_all_report.py")


if __name__ == "__main__":
    main()
