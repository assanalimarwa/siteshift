"""
All public data combined, tested on our patients (user request 2026-10-28); resumable:

    python run_all_public.py --stage unet [--dry-run]      # fmgrid env
    python run_all_public.py --stage uni  [--dry-run]
    python run_all_public.py --stage uni --skip ctx256       # UNI in tile mode only (user, 2026-10-29)

Sources (binary tumour vs not, at our ~0.5 um/px): SICAPv2 (Kaggle copy, ~1 um/px, enlarged x1.5-2.0 as in
every SICAPv2 run; grey >= 25 = tumour), PANDA Karolinska (0.503 um/px; 2 = tumour, 1 = not, 0 = background
counted as not tumour, as in all earlier Karolinska runs), PANDA Radboud (0.486 um/px; 3-5 = tumour, 1-2 = not,
0 = ignored; prepare_radboud.py). RINGS and AGGC22 left out for now (user, 2026-10-28; AGGC22 needs a data
request), PESO excluded.

Conditions (each model scored on all 17 of our patients, none seen in training, tile and full mode, with
boundary-tolerant counts: evaluate.py --tolerant):
  A  all sources, the same number of tile draws from each source every epoch
       sources: SICAPv2 202 train regions, Karolinska 3,814 train slides, Radboud 1,800 train slides
  B  size-matched diversity test, N = 600 slides: 200 SICAPv2 regions + 200 Karolinska + 200 Radboud slides
     (SICAPv2 has only 202 train regions, so 600 is the largest even mix) vs 600 Karolinska slides only
     (run_karolinska_scaling.split_file(600): the 419-slide set + 181 random others)
  C  few-shot from model A on 2, 4, 8 of our patients, the run_fewshot.py protocol (per CV fold, nested
     patients, 100,000 tile draws, last weights, tile mode); the Karolinska and ImageNet starts of
     run_fewshot.py are reused
U-Net (ResNet-34, resnet34_v2 recipe): A and both B arms have the schedule of the Karolinska runs (epochs of
60,000 draws, val on 20,000 tiles shared evenly across the run's sources, early stopping patience 8, at most
40 epochs). UNI (frozen, the FM grid head): the FM grid budget (at most 15 epochs, patience 4) on the same
draws; C from a fresh head instead of ImageNet.
The baselines (our CV models, all-Karolinska, SICAPv2) are re-scored with --tolerant once.
Few-shot runs drop last.pt (resume file, ~0.46 GB) once trained: their best.pt holds the last weights.

Report: all_public_report.py -> cv/all_public.md.
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

from run_fewshot import BUDGET, split_file as fewshot_split
from run_karolinska_scaling import KA, split_file as karolinska_split

HERE = P.INDEX
ROOT = str(P.DATA_ROOT)          # folder holding sicapv2/, karolinska_all/, radboud/
CFG = HERE / "cv" / "all_public"
MODES = {"tile": ["--context", "0", "--no-tta"], "ctx256_blend_tta": []}
SCHEDULE = ["--epoch-tiles", "60000", "--limit-val", "20000"]
FM_SCHEDULE = SCHEDULE + ["--epochs", "15", "--patience", "4"]
FEWSHOT_N = [2, 4, 8]
SICAP = dict(name="SICAPv2", data=f"{ROOT}/sicapv2", index="sicap_tiles.csv", split="sicap_split.json", zoom=[1.5, 2.0])
KAR = dict(name="Karolinska", data=f"{ROOT}/karolinska_all", index="karolinska_all_tiles.csv", split="karolinska_all_split.json")
RAD = dict(name="Radboud", data=f"{ROOT}/radboud", index="radboud_tiles.csv", split="radboud_split.json")
N_MIX = 600


def restricted(src_split, train, name):
    """cv/all_public/<name>.json: src_split with only the given train slides (val / test unchanged)."""
    path = CFG / f"{name}.json"
    if not path.exists():
        s = json.loads((HERE / src_split).read_text())
        s["splits"] = dict(s["splits"], train=sorted(train))
        s["restricted"] = f"{len(train)} train slides of {src_split} (seed 0)"
        path.write_text(json.dumps(s, indent=1))
    return path.relative_to(HERE).as_posix()


def sources():
    """Writes the source lists of A and the mixed arm of B; returns their paths (relative to HERE)."""
    CFG.mkdir(parents=True, exist_ok=True)
    a = CFG / "sources_A.json"
    if not a.exists():
        a.write_text(json.dumps({"sources": [SICAP, KAR, RAD]}, indent=1))
    b = CFG / f"sources_B{N_MIX}.json"
    if not b.exists() and (HERE / RAD["split"]).exists():
        k = N_MIX // 3
        pick = lambda split: np.random.default_rng(0).permutation(sorted(json.loads((HERE / split).read_text())["splits"]["train"]))[:k].tolist()
        mix = [dict(SICAP, split=restricted(SICAP["split"], pick(SICAP["split"]), f"sicap_b{N_MIX}")),
               dict(KAR, split=restricted(KAR["split"], pick(karolinska_split(N_MIX)), f"karolinska_b{N_MIX}")),   # within the 600
               dict(RAD, split=restricted(RAD["split"], pick(RAD["split"]), f"radboud_b{N_MIX}"))]
        b.write_text(json.dumps({"sources": mix}, indent=1))
    return a.relative_to(HERE).as_posix(), b.relative_to(HERE).as_posix()


def train_step(label, name, args, steps):
    run = P.RUNS / name
    trained = (run / "TRAINED").exists()
    steps.append((label, trained, ["train.py", "--name", name, *args,
                                   *(["--resume"] if (run / "last.pt").exists() and not trained else [])], run / "train.log"))


def eval_steps(label, name, split_file, steps, modes=MODES, tag="all", pictures=False):
    run = P.RUNS / name
    for mode, extra in modes.items():
        steps.append((f"{label}: our patients, {mode}", (run / f"eval_{tag}_test_{mode}_tol" / "summary.json").exists(),
                      ["evaluate.py", "--name", name, "--split-file", split_file, "--tolerant", *extra,
                       *([] if pictures and mode != "tile" else ["--no-pictures"])], run / "eval.log"))


def plan(stage):
    a_src, b_src = sources()
    steps = []
    if stage == "unet":
        arch, sched, pre, fresh = [], SCHEDULE, "", {"imagenet": []}
        baselines = [("our CV", [(f"cv/imagenet/fold{k}", f"cv/fold{k}.json", f"fold{k}") for k in range(5)]),
                     ("all-Karolinska", [("karolinska_all_unet", "cv/all.json", "all")]),
                     ("SICAPv2", [("sicap_pretrain", "cv/all.json", "all")])]
    else:
        arch, sched, pre, fresh = ["--arch", "fm_uni"], FM_SCHEDULE, "uni_", {"scratch": ["--arch", "fm_uni"]}
        baselines = [("our CV (UNI)", [(f"cv/fm_uni/fold{k}", f"cv/fold{k}.json", f"fold{k}") for k in range(5)]),
                     ("SICAPv2 (UNI)", [("sicap_fm_uni", "cv/all.json", "all")])]
    suffix = "unet" if stage == "unet" else "uni"

    # order (user's meeting 2026-10-28): baselines in tile mode, A, A scored, baselines in full mode, then C and B
    # baselines, re-scored with boundary-tolerant counts
    for mode in MODES:
        if mode != "tile":
            # A: all sources
            train_step(f"A: {suffix} on all public sources", f"all_public_{suffix}", [*arch, "--sources", a_src, *sched], steps)
            eval_steps(f"A ({suffix})", f"all_public_{suffix}", "cv/all.json", steps, pictures=True)
        for label, runs in baselines:
            for name, split, tag in runs:
                eval_steps(f"baseline {label} {tag}", name, split, steps, modes={mode: MODES[mode]}, tag=tag)
    # C: few-shot from A (and, for UNI, from a fresh head)
    inits = {"public": ["--init", f"runs/all_public_{suffix}/best.pt", *arch]}
    if stage != "unet":
        inits.update(fresh)
    for n in FEWSHOT_N:
        for init, extra in inits.items():
            for k in range(5):
                name, split = f"fewshot/{pre}{init}/n{n}/fold{k}", fewshot_split(k, n)
                train_step(f"C: {init} ({suffix}) n={n} fold {k}", name, [*extra, "--split", split, *BUDGET], steps)
                run = P.RUNS / name
                steps.append((f"C: {init} ({suffix}) n={n} fold {k}: test patients, tile",
                              (run / f"eval_fold{k}_n{n}_test_tile" / "summary.json").exists(),
                              ["evaluate.py", "--name", name, "--split-file", split, "--no-pictures", *MODES["tile"]], run / "eval.log"))
    # B: N_MIX slides, even mix vs Karolinska only (same schedule)
    kar_name = f"karolinska_n{N_MIX}_{suffix}"
    train_step(f"B: {suffix} on {N_MIX} Karolinska slides", kar_name,
               [*arch, *KA, "--split", karolinska_split(N_MIX), *sched], steps)
    eval_steps(f"B Karolinska {N_MIX} ({suffix})", kar_name, "cv/all.json", steps)
    train_step(f"B: {suffix} on {N_MIX} mixed slides", f"mixed{N_MIX}_{suffix}", [*arch, "--sources", b_src, *sched], steps)
    eval_steps(f"B mixed {N_MIX} ({suffix})", f"mixed{N_MIX}_{suffix}", "cv/all.json", steps)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["unet", "uni"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="run only the steps whose label contains this text (e.g. 'baseline')")
    ap.add_argument("--until", help="stop before the first step whose label contains this text (e.g. 'C:')")
    ap.add_argument("--skip", help="leave out the steps whose label contains this text (e.g. 'ctx256': tile mode only)")
    a = ap.parse_args()
    steps = plan(a.stage)
    if a.until:
        cut = next((i for i, s in enumerate(steps) if a.until in s[0]), len(steps))
        steps = steps[:cut]
    if a.only:
        steps = [s for s in steps if a.only in s[0]]
    if a.skip:
        steps = [s for s in steps if a.skip not in s[0]]
    if a.dry_run:
        for label, done, cmd, _ in steps:
            print(f"{'done' if done else 'todo'}  {label}   [{' '.join(cmd)}]")
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
            if "fewshot/" in cmd[2]:
                (log.parent / "last.pt").unlink(missing_ok=True)
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"stage {a.stage} done; next: python all_public_report.py")


if __name__ == "__main__":
    main()
