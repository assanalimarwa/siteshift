"""
Run the patient cross-validation: train one model per fold and configuration, score its test
patients, and score models that never saw our tiles on every patient. Resumable: finished
steps are skipped, an interrupted training continues from its last epoch.

    python run_cv.py                                   # everything below, in order
    python run_cv.py --configs imagenet --folds 0 1    # a subset
    python run_cv.py --dry-run                         # list the steps and what is done

Needs cv/fold<k>.json and cv/all.json from make_folds.py. Steps run one after another, never in
parallel: two jobs with 12 data-loading workers each exceed this machine's memory limit.

Configurations (every setting of resnet34_v2 except the starting weights):
    imagenet    ImageNet ResNet-34 encoder, as resnet34_v2
    sicap       weights of runs/sicap_pretrain/best.pt (pretrained on SICAPv2)
Zero-shot (no training on our tiles, every patient is a test patient):
    sicap_only  runs/sicap_pretrain itself

Outputs: runs/cv/<config>/fold<k>/ (best.pt, log.csv, train.log, eval.log,
eval_fold<k>_test_tile/, eval_fold<k>_test_ctx256_blend_tta/), and for zero-shot models
runs/<run>/eval_all_test_tile/ and eval_all_test_ctx256_blend_tta/. Summaries: cv_summary.py.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = P.INDEX
CONFIGS = {
    "imagenet": [],
    "sicap": ["--init", "runs/sicap_pretrain/best.pt"],
}
# Frozen foundation-model encoders with the same trainable head (model.FrozenEncoderSeg). Fixed
# before any result: at most 15 epochs, early stop after 4 without a better val Dice, and scored
# with each tile alone ("tile" mode) - context windows would cost hours per fold for ViT-L/H.
FM_BUDGET = ["--epochs", "15", "--patience", "4"]
FM_ENCODERS = ["uni", "virchow2", "conch", "dinov2", "dinov3", "unimedclip"]
FM_CONFIGS = {f"fm_{e}": ["--arch", f"fm_{e}", *FM_BUDGET] for e in FM_ENCODERS}
CONFIGS.update(FM_CONFIGS)
# Architectures compared with the ResNet-34 U-Net (model.py). SegFormer-B2 and TransUNet keep the U-Net settings (train.py defaults,
# epochs of 60,000 tiles); nnU-Net uses its own recipe, with epochs of 3,000 tiles (pass --epoch-tiles 3000 --limit-val 2000).
CONFIGS.update(segformer=["--arch", "segformer"], transunet=["--arch", "transunet"],
               nnunet=["--arch", "nnunet", "--recipe", "nnunet", "--lr", "0.01", "--weight-decay", "3e-5", "--ema", "0",
                       "--epochs", "250", "--batch", "12", "--patience", "0", "--keep", "last", "--warmup", "0"])
DEFAULT_CONFIGS = ["imagenet", "sicap"]
# trained on SICAPv2 only (train_sicap_fm.sh for the encoders), scored on every patient
ZERO_SHOT = {"sicap_only": "sicap_pretrain", **{f"sicap_fm_{e}": f"sicap_fm_{e}" for e in FM_ENCODERS}}
ZERO_SHOT_TILE_ONLY = {f"sicap_fm_{e}" for e in FM_ENCODERS}   # frozen encoders: tile mode, as in the grid
# the SICAPv2-only models again, with our tiles recoloured to SICAPv2's stain (stain.py), tile mode
STAIN = "cv/stain_macenko_sicap.json"
STAINED = {f"{n}+stain": run for n, run in ZERO_SHOT.items()}
# no training: a fold model of a configuration averaged with fixed checkpoints (equal weights)
ENSEMBLES = {"ens_imagenet_sicaponly": dict(base="imagenet", add=["runs/sicap_pretrain/best.pt"], tag="sicaponly")}
MODES = {"tile": ["--context", "0", "--no-tta", "--no-pictures"], "ctx256_blend_tta": []}
# set from the command line: fold folder (also the run-name prefix), dataset arguments for every
# train.py / evaluate.py call, extra train.py arguments
CV, DATA_ARGS, TRAIN_EXTRA = "cv", [], []


def steps(configs, folds, zero_shot, ensembles=(), stain_runs=()):
    """(label, done?, command, log) in the order they run."""
    out = []
    for name in ensembles:
        e = ENSEMBLES[name]
        for k in folds:
            run = P.RUNS / CV / e["base"] / f"fold{k}"
            if not (run / "TRAINED").exists():
                continue                              # its base model is not trained yet
            for mode, extra in MODES.items():
                done = (run / f"eval_fold{k}_test_{mode}_ens-{e['tag']}" / "summary.json").exists()
                out.append((f"{name} fold {k} test {mode}", done,
                            ["evaluate.py", "--name", f"{CV}/{e['base']}/fold{k}", "--split-file", f"{CV}/fold{k}.json",
                             *DATA_ARGS, "--ensemble", *e["add"], "--ensemble-name", e["tag"], "--no-pictures", *extra],
                            run / "eval_ensemble.log"))
    for name in zero_shot:
        run = ZERO_SHOT[name]
        if not (P.RUNS / run / "best.pt").exists():
            continue                                  # not trained yet
        for mode, extra in MODES.items():
            if name in ZERO_SHOT_TILE_ONLY and mode != "tile":
                continue
            done = (P.RUNS / run / f"eval_all_test_{mode}" / "summary.json").exists()
            out.append((f"{name} all patients {mode}", done,
                        ["evaluate.py", "--name", run, "--split-file", f"{CV}/all.json", *DATA_ARGS, *extra],
                        P.RUNS / run / "eval_all.log"))
        stained = f"{name}+stain"
        if stained in STAINED and name in stain_runs:
            done = (P.RUNS / run / f"eval_all_test_tile_stain-{Path(STAIN).stem}" / "summary.json").exists()
            out.append((f"{stained} all patients tile", done,
                        ["evaluate.py", "--name", run, "--split-file", f"{CV}/all.json", *DATA_ARGS, "--stain", STAIN,
                         *MODES["tile"]],
                        P.RUNS / run / "eval_all.log"))
    for k in folds:                                   # fold by fold, so paired results arrive early
        for config in configs:
            name = f"{CV}/{config}/fold{k}"
            run = P.RUNS / name
            trained = (run / "TRAINED").exists()
            resume = ["--resume"] if (run / "last.pt").exists() and not trained else []
            out.append((f"{config} fold {k} train", trained,
                        ["train.py", "--name", name, "--split", f"{CV}/fold{k}.json", *DATA_ARGS, *CONFIGS[config],
                         *TRAIN_EXTRA, *resume],
                        run / "train.log"))
            for mode, extra in MODES.items():
                if config in FM_CONFIGS and mode != "tile":
                    continue
                done = (run / f"eval_fold{k}_test_{mode}" / "summary.json").exists()
                out.append((f"{config} fold {k} test {mode}", done,
                            ["evaluate.py", "--name", name, "--split-file", f"{CV}/fold{k}.json", *DATA_ARGS, *extra],
                            run / "eval.log"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=DEFAULT_CONFIGS,
                    help="fm_* configurations need the fmgrid env (timm) and access to the gated weights")
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--zero-shot", nargs="*", choices=list(ZERO_SHOT), default=list(ZERO_SHOT))
    ap.add_argument("--ensembles", nargs="*", choices=list(ENSEMBLES), default=[],
                    help="ensembles of trained fold models with fixed checkpoints (evaluation only)")
    ap.add_argument("--stain", nargs="*", choices=list(ZERO_SHOT), default=[],
                    help="zero-shot runs to score again on stain-normalised tiles (needs cv/stain_macenko_sicap.json)")
    ap.add_argument("--cv", default="cv", help="fold folder from make_folds.py --out; also the prefix of the run names")
    ap.add_argument("--data", help="dataset root passed to train.py and evaluate.py (default: theirs)")
    ap.add_argument("--index", help="tile index passed to train.py and evaluate.py (default: theirs)")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[],
                    help="further train.py arguments, e.g. --epoch-tiles 60000 --limit-val 20000 (must come last)")
    ap.add_argument("--tile-only", action="store_true", help="skip the context-window evaluations (all reported numbers are tile mode)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global CV, DATA_ARGS, TRAIN_EXTRA
    CV, TRAIN_EXTRA = a.cv, a.train_extra
    DATA_ARGS = (["--data", a.data] if a.data else []) + (["--index", a.index] if a.index else [])

    n_folds = len(json.loads((HERE / CV / "folds.json").read_text())["folds"])
    folds = a.folds if a.folds is not None else list(range(n_folds))
    plan = steps(a.configs, folds, a.zero_shot, a.ensembles, a.stain)
    if a.tile_only:
        plan = [x for x in plan if not x[0].endswith("ctx256_blend_tta")]
    for label, done, cmd, log in plan:
        print(f"{'done' if done else 'todo'}  {label}")
    if a.dry_run:
        return

    t_start = time.time()
    for label, done, cmd, log in plan:
        if done:
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{time.strftime('%H:%M')}] {label}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with log.open("a") as f:
            code = subprocess.run([sys.executable, str(P.script(cmd[0])), *cmd[1:]], cwd=P.INDEX, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"{label} failed (exit {code}); see {log}")
        if cmd[0] == "train.py":
            (log.parent / "TRAINED").write_text(time.strftime("%Y-%m-%d %H:%M"))
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nall steps done in {(time.time() - t_start) / 3600:.1f} h; next: python cv_summary.py")


if __name__ == "__main__":
    main()
