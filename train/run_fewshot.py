"""
Few-shot adaptation to our site (Track C baselines): four starting models fine-tuned on 2, 4 or 8 of a fold's train
patients, then scored on the fold's test patients in tile mode; resumable:

    python run_fewshot.py [--dry-run]

Draws (data/fewshot/fold<k>_n<n>.json, the files used in the paper): per fold k of data/cv47, the fold's train patients in
a random order (numpy default_rng(k)); the first n of them (n = 2, 4, 8; nested) are the only labelled patients used for
training. The fold's val patients are only logged. No model selection on labels: every run has the same fixed budget
(10 epochs of 10,000 tile draws = 100,000 tiles, last weights kept) and the fold's test patients are scored, so each of
the 47 patients is scored by models that never saw it.

  start                    run name                            init
  U-Net from Karolinska    fewshot47/kar/n<n>/fold<k>          runs/karolinska_all_unet/best.pt (train/configs/public__karolinska_all_unet.json)
  U-Net from ImageNet      fewshot47/imagenet/n<n>/fold<k>     ImageNet ResNet-34, as in the site CV
  UNI, fresh head          fewshot47/uni_scratch/n<n>/fold<k>  frozen UNI, new head
  UNI from all-public      fewshot47/uni_public/n<n>/fold<k>   runs/all_public_uni/best.pt (train/configs/public__all_public_uni.json)

Every evaluation also saves the protocol-format predictions ($SITESHIFT_PRED/fewshot47_<start>_n<n>_fold<k>/ours47_tile) for
eval/protocol_cache47.py and eval/report_protocol47.py. Tables: python eval/report47.py fewshot (per-patient Dice),
python eval/report_protocol47.py (detection / delineation attribution of the gap).

Public benchmark (Track C): only fold 2's own draws (fewshot/fold2_n<n>.json; its adaptation patients are outside fold 2),
scored on fold 2. The draws of folds 0, 1, 3 and 4 contain fold-2 patients and are not part of the public benchmark.
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

import numpy as np

HERE = P.INDEX
SIZES = [2, 4, 8]
DATA = ["--data", str(P.DATASET47), "--index", "tiles47.csv"]
BUDGET = ["--epochs", "10", "--epoch-tiles", "10000", "--patience", "0", "--keep", "last", "--limit-val", "2000"]
INITS = {"kar": ["--init", str(P.RUNS / "karolinska_all_unet" / "best.pt")],
         "imagenet": [],
         "uni_scratch": ["--arch", "fm_uni"],
         "uni_public": ["--arch", "fm_uni", "--init", str(P.RUNS / "all_public_uni" / "best.pt")]}
TILE = ["--context", "0", "--no-tta", "--no-pictures"]


def split_file(k, n):
    """fewshot/fold<k>_n<n>.json: the fold with only the first n of its train patients (random order, seed k).
    The released draws are the ones of the paper; this only rebuilds a missing file the same way."""
    path = HERE / "fewshot" / f"fold{k}_n{n}.json"
    if not path.exists():
        fold = json.loads((HERE / "cv47" / f"fold{k}.json").read_text())
        order = [int(p) for p in np.random.default_rng(k).permutation(sorted(fold["patients"]["train"]))]
        chosen = order[:n]
        s = dict(fold, seed=k, fewshot_n=n, patient_order=order)
        s["splits"] = dict(fold["splits"], train=[x for x in fold["splits"]["train"] if int(x.split(".")[0]) in chosen])
        s["patients"] = dict(fold["patients"], train=chosen)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2))
    return path.relative_to(HERE).as_posix()


def train_step(label, run, args):
    r = P.RUNS / run
    trained = (r / "TRAINED").exists()
    return (label, trained, ["train.py", "--name", run, *DATA, *args, *(["--resume"] if (r / "last.pt").exists() and not trained else [])],
            r / "train.log")


def eval_step(label, run, split, tag):
    """Tile-mode evaluation on the split's test patients, protocol-format predictions saved."""
    return (label, (P.RUNS / run / f"eval_{tag}_test_tile" / "summary.json").exists(),
            ["evaluate.py", "--name", run, "--split-file", split, *DATA, *TILE,
             "--save-pred", str(P.PRED / run.replace("/", "_") / "ours47_tile")], P.RUNS / run / "eval.log")


def plan():
    steps = []
    for n in SIZES:
        for init, extra in INITS.items():
            for k in range(5):
                run, split = f"fewshot47/{init}/n{n}/fold{k}", split_file(k, n)
                steps.append(train_step(f"{init} n={n} fold {k}: train", run, [*extra, "--split", split, *BUDGET]))
                steps.append(eval_step(f"{init} n={n} fold {k}: test patients, tile", run, split, f"fold{k}_n{n}"))
    return steps


def run(steps, dry_run, next_step):
    """Run the steps that are not done yet, in order; each step's output goes to its log."""
    for label, done, _, _ in steps:
        print(f"{'done' if done else 'todo'}  {label}")
    if dry_run:
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
    print(f"done; next: {next_step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        for k in range(5):
            s = {n: json.loads((HERE / "fewshot" / f"fold{k}_n{n}.json").read_text())["patients"]["train"] for n in SIZES}
            print(f"fold {k}: " + "  ".join(f"n={n}: {s[n]}" for n in SIZES))
    run(plan(), a.dry_run, "python eval/protocol_cache47.py <prediction folders>, python eval/report47.py fewshot, "
        "python eval/report_protocol47.py")


if __name__ == "__main__":
    main()
