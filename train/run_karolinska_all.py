"""
U-Net trained on every Karolinska slide, tested on our patients (answers "you did not use all of
Karolinska"); resumable:

    python run_karolinska_all.py [--dry-run]

  1. train on the 3,814 Karolinska train slides (karolinska_all_split.json; resnet34_v2 recipe), each
     epoch a fresh random 60,000 tiles (the epoch size of the 600-slide run), early stopping on a
     fixed random 20,000 tiles of the 546 val slides                             runs/karolinska_all_unet
  2. score it on all 17 of our patients (tile mode and full mode)
  3. scaling check: this model and the 600-slide model (runs/karolinska_unet) on the 1,092 Karolinska
     test slides, tile mode
  4. reverse direction, tile mode: our five cross-validation fold U-Nets (runs/cv/imagenet/fold<k>) and
     the SICAPv2 U-Net on the 1,092 Karolinska test slides
  5. reverse direction, full mode: this model and our five fold U-Nets on the 1,092 test slides
     (~70 min each)
Tables: karolinska_all_report.py.
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
KA = ["--data", str(P.DATA_ROOT / "karolinska_all"), "--index", "karolinska_all_tiles.csv"]
MODES = {"tile": ["--context", "0", "--no-tta"], "ctx256_blend_tta": []}


def plan():
    run = P.RUNS / "karolinska_all_unet"
    trained = (run / "TRAINED").exists()
    steps = [("train U-Net on all Karolinska slides", trained,
              ["train.py", "--name", "karolinska_all_unet", *KA, "--split", "karolinska_all_split.json",
               "--epoch-tiles", "60000", "--limit-val", "20000",
               *(["--resume"] if (run / "last.pt").exists() and not trained else [])], run / "train.log")]
    for mode, extra in MODES.items():
        steps.append((f"all-slide Karolinska U-Net on our 17 patients, {mode}", (run / f"eval_all_test_{mode}" / "summary.json").exists(),
                      ["evaluate.py", "--name", "karolinska_all_unet", "--split-file", "cv/all.json", *extra], run / "eval.log"))
    folds = [f"cv/imagenet/fold{k}" for k in range(5)]
    for name, mode in ([(n, "tile") for n in ["karolinska_all_unet", "karolinska_unet", *folds, "sicap_pretrain"]]
                       + [(n, "ctx256_blend_tta") for n in ["karolinska_all_unet", *folds]]):
        r = P.RUNS / name
        steps.append((f"{name} on all 1,092 Karolinska test slides, {mode}", (r / f"eval_karolinska_all_split_test_{mode}" / "summary.json").exists(),
                      ["evaluate.py", "--name", name, *KA, "--split-file", "karolinska_all_split.json", "--split", "test",
                       "--no-pictures", *MODES[mode]], r / "eval_karolinska_all.log"))
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
    print("done; next: python karolinska_all_report.py")


if __name__ == "__main__":
    main()
