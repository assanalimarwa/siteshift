"""
Label-harmonization test: does the SICAPv2 model's under-calling on our patients go away when only its
training labels change (gland-level -> region-style, prepare_sicap_closed.py)? Resumable:

    python run_label_harmonization.py [--dry-run]

Three U-Nets with the recipe of runs/sicap_pretrain (SICAPv2 tiles, --shift 0 --zoom 1.5 2.0, seed 0),
differing only in the masks or the seed:
  sicap_r40              SICAPv2 masks closed with r = 40 um ($SITESHIFT_DATA/sicapv2_r40)
  sicap_r80              r = 80 um ($SITESHIFT_DATA/sicapv2_r80)
  sicap_pretrain_seed1   the original masks, seed 1: how much a rerun moves the numbers by itself
Each scored on all 17 of our patients (tile and full mode) and on SICAPv2's held-out tiles with the
original masks (evaluate_sicap.py). Table: label_harmonization_report.py.
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
RUNS = {"sicap_r40": (str(P.DATA_ROOT / "sicapv2_r40"), 0), "sicap_r80": (str(P.DATA_ROOT / "sicapv2_r80"), 0),
        "sicap_pretrain_seed1": (str(P.DATA_ROOT / "sicapv2"), 1)}
MODES = {"tile": ["--context", "0", "--no-tta"], "ctx256_blend_tta": []}


def plan():
    steps = []
    for name, (data, seed) in RUNS.items():
        run = P.RUNS / name
        trained = (run / "TRAINED").exists()
        steps.append((f"train {name}", trained,
                      ["train.py", "--name", name, "--data", data, "--index", "sicap_tiles.csv", "--split", "sicap_split.json",
                       "--shift", "0", "--zoom", "1.5", "2.0", "--seed", str(seed),
                       *(["--resume"] if (run / "last.pt").exists() and not trained else [])], run / "train.log"))
        for mode, extra in MODES.items():
            steps.append((f"{name} on our 17 patients, {mode}", (run / f"eval_all_test_{mode}" / "summary.json").exists(),
                          ["evaluate.py", "--name", name, "--split-file", "cv/all.json", *extra], run / "eval.log"))
        steps.append((f"{name} on SICAPv2 held-out (original masks)", (run / "eval_sicap_guard_tile" / "summary.json").exists(),
                      ["evaluate_sicap.py", "--name", name], run / "eval.log"))
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
    print("done; next: python label_harmonization_report.py")


if __name__ == "__main__":
    main()
