"""
Reverse direction: does our dataset help on SICAPv2? Cross-validated over SICAPv2 regions
(make_sicap_folds.py: 5 folds, every one of the 238 regions tested once).

    python run_reverse.py                         # every method, lightest first; resumable
    python run_reverse.py --methods imagenet      # one method (the U-Net first, to check early)
    python run_reverse.py --dry-run

Methods: U-Net (ResNet-34) and the frozen encoders with the shared head (UNI, CONCH, UniMed-CLIP,
DINOv2-L, Virchow2). For SICAPv2 fold k (train on its train regions, early stopping on its val
regions, test on its test regions):
  A  SICAPv2 only: U-Net from ImageNet / head from scratch            runs/reverse/A/<method>/fold<k>
  E  U-Net from random weights (no ImageNet), U-Net only              runs/reverse/E/fold<k>
  C  start from our cross-validation fold-k model of the method (each of our five fold models is
     used once), then trained on SICAPv2 fold k                       runs/reverse/C/<method>/fold<k>
  B  our five fold models as they are, each scored on every SICAPv2 tile (they never saw SICAPv2)
Recipe of A, C and E as in the ranking test: U-Net with train.py defaults (40 epochs, early stop 8),
frozen encoders 15 epochs, early stop 4; SICAPv2 tiles enlarged 1.5-2x to our scale.
Scoring: evaluate_reverse.py (also style-tolerant counts). Tables: reverse_report.py -> cv/reverse.md.
(D, joint training on ours + SICAPv2, is not implemented: it needs a mixed-dataset loader.)
Run with the fmgrid env (timm). Steps run one at a time.
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
FM_RECIPE = ["--epochs", "15", "--patience", "4"]
METHODS = {"imagenet": ("resnet34", []), "fm_uni": ("fm_uni", FM_RECIPE), "fm_conch": ("fm_conch", FM_RECIPE),
           "fm_unimedclip": ("fm_unimedclip", FM_RECIPE), "fm_dinov2": ("fm_dinov2", FM_RECIPE),
           "fm_virchow2": ("fm_virchow2", FM_RECIPE)}
ORDER = list(METHODS)                                                    # lightest first
SICAP = ["--data", str(P.DATA_ROOT / "sicapv2"), "--index", "sicap_tiles.csv", "--shift", "0", "--zoom", "1.5", "2.0"]
FOLDS = range(5)
RANDOM_INIT = P.RUNS / "reverse" / "random_init_resnet34.pt"


def fold_file(k):
    return f"cv/sicap_fold{k}.json"


WORKERS = ["--workers", "12"]                                            # main() lowers it with --workers


def train_step(name, arch, k, recipe, init=None):
    run = P.RUNS / name
    done = (run / "TRAINED").exists()
    resume = ["--resume"] if (run / "last.pt").exists() and not done else []
    cmd = ["train.py", "--name", name, "--arch", arch, *SICAP, "--split", fold_file(k), *recipe, *resume, *WORKERS]
    if init:
        cmd += ["--init", str(init)]
    return (f"{name} train", done, cmd, run / "train.log", init)


def score_step(ckpt, out, split, k=None):
    cmd = ["evaluate_reverse.py", "--ckpt", str(ckpt), "--out", str(out), "--split", split, "--workers", WORKERS[1]]
    if k is not None:
        cmd += ["--split-file", fold_file(k)]
    return (f"{out.relative_to(P.RUNS)} score", (out / "summary.json").exists(), cmd, out.parent / "eval_reverse.log", ckpt)


def plan(methods=ORDER, conditions="ABCE"):
    smoke = P.RUNS / "reverse" / "smoke_cv"
    steps = [("smoke test: score 16 tiles of SICAPv2 fold 0", (smoke / "summary.json").exists(),
              ["evaluate_reverse.py", "--ckpt", str(P.RUNS / "sicap_pretrain" / "best.pt"), "--out", str(smoke),
               "--split", "test", "--split-file", fold_file(0), "--limit", "16"], smoke.parent / "smoke.log", None)]
    for m in methods:
        arch, recipe = METHODS[m]
        for k in FOLDS if "A" in conditions else []:                     # A
            name = f"reverse/A/{m}/fold{k}"
            steps.append(train_step(name, arch, k, recipe))
            steps.append(score_step(P.RUNS / name / "best.pt", P.RUNS / name / "eval_reverse_test", "test", k))
        if m == "imagenet" and "E" in conditions:                        # E
            for k in FOLDS:
                name = f"reverse/E/fold{k}"
                steps.append(train_step(name, arch, k, recipe, init=RANDOM_INIT))
                steps.append(score_step(P.RUNS / name / "best.pt", P.RUNS / name / "eval_reverse_test", "test", k))
        for j in FOLDS if "B" in conditions else []:                     # B
            fold = P.RUNS / "cv" / m / f"fold{j}"
            steps.append(score_step(fold / "best.pt", fold / "eval_reverse_all", "all"))
        for k in FOLDS if "C" in conditions else []:                     # C
            name = f"reverse/C/{m}/fold{k}"
            steps.append(train_step(name, arch, k, recipe, init=P.RUNS / "cv" / m / f"fold{k}" / "best.pt"))
            steps.append(score_step(P.RUNS / name / "best.pt", P.RUNS / name / "eval_reverse_test", "test", k))
    return steps


def ready(step):
    """A step can run once its input exists (B and C need our cross-validation fold models; scores need training)."""
    need = step[4]
    return need is None or Path(need).exists() or Path(need) == RANDOM_INIT


def make_random_init():
    if RANDOM_INIT.exists():
        return
    import torch
    from model import build_model
    torch.manual_seed(0)
    RANDOM_INIT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=build_model("resnet34", pretrained=False).state_dict(), arch="resnet34", epoch=0,
                    val={"dice": float("nan")}), RANDOM_INIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", choices=ORDER, default=ORDER, help="in this order (default: lightest first)")
    ap.add_argument("--conditions", default="ABCE", help="which conditions to run, e.g. A (independent of our patients)")
    ap.add_argument("--workers", type=int, default=12, help="data-loading workers (fewer when memory is tight)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    WORKERS[1] = str(a.workers)
    steps = plan(a.methods, a.conditions)
    for step in steps:
        print(f"{'done' if step[1] else 'todo' if ready(step) else 'wait'}  {step[0]}")
    if a.dry_run:
        return
    make_random_init()
    t_start = time.time()
    for step in steps:
        label, done, cmd, log, _ = step
        if done:
            continue
        if not ready(step):
            print(f"skip {label}: its input does not exist yet", flush=True)
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
    print(f"\nall steps done in {(time.time() - t_start) / 3600:.1f} h; next: python reverse_report.py")


if __name__ == "__main__":
    main()
