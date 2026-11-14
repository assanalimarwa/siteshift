"""
Where each model's wrong pixels are (breakdown.py), for every train -> test direction among our data,
Karolinska and SICAPv2, tile mode; resumable:

    python run_breakdown.py [--dry-run]

  test set                  models
  our 17 patients           ours (each patient by the CV fold model that never saw it), SICAPv2, Karolinska
                            (all slides), SICAPv2 relabelled r = 40 / 80 um (if trained)
  Karolinska test (1,092)   Karolinska, ours (all five fold models), SICAPv2
  SICAPv2 held-out          SICAPv2, ours (all five fold models), Karolinska   (val + guard, evaluate_sicap.py)
Table: error_breakdown_report.py.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = P.INDEX
TILE = ["--context", "0", "--no-tta", "--no-pictures", "--breakdown"]
KA = ["--data", str(P.DATA_ROOT / "karolinska_all"), "--index", "karolinska_all_tiles.csv",
      "--split-file", "karolinska_all_split.json", "--split", "test", "--um-per-px", "0.503"]
FOLDS = [f"cv/imagenet/fold{k}" for k in range(5)]


def plan():
    steps = []

    def add(label, run, out, cmd):
        r = P.RUNS / run
        steps.append((label, (r / out / "summary.json").exists(), cmd, r / "eval_breakdown.log"))

    for k, f in enumerate(FOLDS):
        add(f"ours fold {k} -> ours", f, f"eval_fold{k}_test_tile_breakdown", ["evaluate.py", "--name", f, "--split-file", f"cv/fold{k}.json", *TILE])
    for run in ["sicap_pretrain", "karolinska_all_unet", "sicap_r40", "sicap_r80"]:
        if (P.RUNS / run / "best.pt").exists():
            add(f"{run} -> ours", run, "eval_all_test_tile_breakdown", ["evaluate.py", "--name", run, "--split-file", "cv/all.json", *TILE])
    for run in ["karolinska_all_unet", *FOLDS, "sicap_pretrain"]:
        add(f"{run} -> Karolinska test", run, "eval_karolinska_all_split_test_tile_breakdown", ["evaluate.py", "--name", run, *TILE, *KA])
    for run in ["sicap_pretrain", *FOLDS, "karolinska_all_unet"]:
        add(f"{run} -> SICAPv2 held-out", run, "eval_sicap_guard_tile_breakdown", ["evaluate_sicap.py", "--name", run, "--breakdown"])
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = plan()
    for label, done, _, _ in steps:
        print(f"{'done' if done else 'todo'}  {label}")
    if a.dry_run:
        return
    for label, done, cmd, log in steps:
        if done:
            continue
        print(f"\n[{time.strftime('%m-%d %H:%M')}] {label}", flush=True)
        t0 = time.time()
        with log.open("a") as f:
            code = subprocess.run([sys.executable, str(P.script(cmd[0])), *cmd[1:]], cwd=P.INDEX, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"{label} failed (exit {code}); see {log}")
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done; next: python error_breakdown_report.py")


if __name__ == "__main__":
    main()
