"""
Karolinska (PANDA) vs our data with the baseline U-Net (ResNet-34, resnet34_v2 recipe):

    python run_karolinska.py            # resumable; finished steps are skipped
    python run_karolinska.py --dry-run

  1. train a U-Net on the Karolinska train slides (val slides for early stopping)   runs/karolinska_unet
  2. score it on the Karolinska test slides (in-domain reference)
  3. score it on all 17 of our patients (Karolinska -> ours)
  4. score our five cross-validation fold models (runs/cv/imagenet/fold<k>, trained on ours) on the
     Karolinska test slides (ours -> Karolinska)
  5. score the SICAPv2-only U-Net (runs/sicap_pretrain) on the Karolinska test slides (public -> public)
Every score in tile mode (each tile alone) and full mode (256 px neighbour context, blending, flips).
Tables: karolinska_report.py.
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
K = ["--data", str(P.DATA_ROOT / "karolinska"), "--index", "karolinska_tiles.csv"]
MODES = {"tile": ["--context", "0", "--no-tta"], "ctx256_blend_tta": []}


def plan():
    steps = []
    run = P.RUNS / "karolinska_unet"
    steps.append(("train U-Net on Karolinska", (run / "TRAINED").exists(),
                  ["train.py", "--name", "karolinska_unet", *K, "--split", "karolinska_split.json",
                   *(["--resume"] if (run / "last.pt").exists() and not (run / "TRAINED").exists() else [])], run / "train.log"))
    for mode, extra in MODES.items():
        steps.append((f"Karolinska U-Net on Karolinska test, {mode}", (run / f"eval_karolinska_split_test_{mode}" / "summary.json").exists(),
                      ["evaluate.py", "--name", "karolinska_unet", *K, "--split-file", "karolinska_split.json", "--split", "test",
                       "--no-pictures", *extra], run / "eval.log"))
        steps.append((f"Karolinska U-Net on our 17 patients, {mode}", (run / f"eval_all_test_{mode}" / "summary.json").exists(),
                      ["evaluate.py", "--name", "karolinska_unet", "--split-file", "cv/all.json", *extra], run / "eval.log"))
    for k in range(5):
        fold = P.RUNS / "cv" / "imagenet" / f"fold{k}"
        for mode, extra in MODES.items():
            steps.append((f"our fold-{k} U-Net on Karolinska test, {mode}", (fold / f"eval_karolinska_split_test_{mode}" / "summary.json").exists(),
                          ["evaluate.py", "--name", f"cv/imagenet/fold{k}", *K, "--split-file", "karolinska_split.json", "--split", "test",
                           "--no-pictures", *extra], fold / "eval_karolinska.log"))
    sic = P.RUNS / "sicap_pretrain"
    for mode, extra in MODES.items():
        steps.append((f"SICAPv2 U-Net on Karolinska test, {mode}", (sic / f"eval_karolinska_split_test_{mode}" / "summary.json").exists(),
                      ["evaluate.py", "--name", "sicap_pretrain", *K, "--split-file", "karolinska_split.json", "--split", "test",
                       "--no-pictures", *extra], sic / "eval_karolinska.log"))
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
    print("all Karolinska steps done; next: python karolinska_report.py")


if __name__ == "__main__":
    main()
