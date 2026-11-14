"""
Public-only training: one model trained on all public sources (SICAPv2 + Karolinska + Radboud, equal tile draws per source: condition A of
run_all_public.py) and tested on all 47 of our patients, tile mode only. This is the job behind the paper's "all-public" models.

    python train/run_public_only.py --config nnunet|segformer|imagenet [--dry-run]

Steps: train.py (sources_A.json, written on first use from $SITESHIFT_DATA), evaluate.py on data/cv47/all.json in tile mode. Resumable.
Afterwards the protocol-format predictions for the attribution: see README.md, "Running each step".
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import argparse
import subprocess
import sys
import time

import siteshift_paths as P
import run_cv
import run_all_public as rap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["imagenet", "segformer", "nnunet"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = a.config
    data_args = ["--data", str(P.DATASET47), "--index", "tiles47.csv"]
    extra = ["--epoch-tiles", "3000", "--limit-val", "2000"] if cfg == "nnunet" else ["--epoch-tiles", "60000", "--limit-val", "20000"]
    src = rap.sources()[0]
    name = f"all_public_{cfg}"
    run = P.RUNS / name
    trained = (run / "TRAINED").exists()
    steps = [(f"{cfg} all-public train", trained,
              ["train.py", "--name", name, "--sources", src, *run_cv.CONFIGS[cfg], *extra,
               *(["--resume"] if (run / "last.pt").exists() and not trained else [])], run / "train.log"),
             (f"{cfg} all-public test all 47 patients tile", (run / "eval_all_test_tile" / "summary.json").exists(),
              ["evaluate.py", "--name", name, "--split-file", "cv47/all.json", *data_args, *run_cv.MODES["tile"]], run / "eval.log")]
    for label, done, cmd, log in steps:
        print(f"{'done' if done else 'todo'}  {label}")
    if a.dry_run:
        return
    for label, done, cmd, log in steps:
        if done:
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with log.open("a") as f:
            code = subprocess.run([sys.executable, str(P.script(cmd[0])), *cmd[1:]], cwd=P.INDEX, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"{label} failed (exit {code}); see {log}")
        if cmd[0] == "train.py":
            (log.parent / "TRAINED").write_text(time.strftime("%Y-%m-%d %H:%M"))
        print(f"  finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
