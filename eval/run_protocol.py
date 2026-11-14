"""
Predictions for the detection-delineation protocol (protocol_predict.py -> protocol.PRED), every model and
transfer direction we have; resumable, one GPU job at a time:

    python run_protocol.py [--dry-run]

  test set                   mode        models
  our 17 patients            tile        site (CV, each patient by the fold model that never saw it): U-Net from
                                         ImageNet and from SICAPv2, UNI, CONCH, DINOv2-L, UniMed-CLIP, Virchow2;
                                         public: SICAPv2 (+ seed 1, relabelled r = 40 / 80 um), Karolinska 20 /
                                         155 / 419 / 600 / 1,000 / 3,814 slides, 600 mixed slides, all public (A),
                                         the five foundation encoders trained on SICAPv2;
                                         few-shot: Karolinska / ImageNet start + 1 / 2 / 4 / 8 / all patients,
                                         all-public start + 2 / 4 / 8 (run_fewshot.py protocol)
  Karolinska test (1,092)    tile        Karolinska 3,814 (home), ours (each of the five fold models), SICAPv2
  SICAPv2 (238 regions)      tile        SICAPv2 CV (home, runs/reverse/A/imagenet), ours (five fold models), Karolinska
  our 17 patients            full        every U-Net above that has a full-mode score (context + blending + flips)
Analysis: protocol.py (CPU); report: protocol_report.py -> cv/protocol.md.
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

from protocol import PRED

HERE = P.INDEX
RUNS = P.RUNS
MODES = {"tile": ["--context", "0", "--no-tta"], "full": []}
DATA = {"ours": [],
        "karolinska": ["--data", str(P.DATA_ROOT / "karolinska_all"), "--index", "karolinska_all_tiles.csv",
                       "--split-file", "karolinska_all_split.json"],
        "sicap": ["--data", str(P.DATA_ROOT / "sicapv2"), "--index", "sicap_tiles.csv", "--zoom", "1.5", "2.0", "--batch", "8"]}
CV = lambda path: [(f"{path}/fold{k}", f"cv/fold{k}.json", "test") for k in range(5)]
ALL = lambda run: [(run, "cv/all.json", "test")]


def sets():
    """[(key, label, group, dataset, mode, [(run, split file, split), ...])]; models not trained are left out."""
    site = [("ours", "ours U-Net (CV)", "cv/imagenet"), ("ours_sicapstart", "ours U-Net, SICAPv2 start (CV)", "cv/sicap")] + \
           [(f"ours_{m}", f"ours {name} (CV)", f"cv/fm_{m}") for m, name in
            [("uni", "UNI"), ("conch", "CONCH"), ("dinov2", "DINOv2-L"), ("unimedclip", "UniMed-CLIP"), ("virchow2", "Virchow2")]] + \
           [(f"ours_{m}", f"ours {name} (CV)", f"cv/{m}") for m, name in [("nnunet", "nnU-Net-style 2D"), ("segformer", "SegFormer-B2")]]
    public = [("sicap", "SICAPv2", "sicap_pretrain"), ("sicap_seed1", "SICAPv2, seed 1", "sicap_pretrain_seed1"),
              ("sicap_r40", "SICAPv2 relabelled r40", "sicap_r40"), ("sicap_r80", "SICAPv2 relabelled r80", "sicap_r80")] + \
             [(f"kar{n}", f"Karolinska {n:,} slides", f"karolinska_n{n}_unet") for n in (20, 155, 419, 600, 1000)] + \
             [("kar3814", "Karolinska 3,814 slides", "karolinska_all_unet"), ("mixed600", "600 mixed slides", "mixed600_unet"),
              ("allpublic", "all public (A)", "all_public_unet"),
              ("allpublic_nnunet", "all public (A), nnU-Net-style 2D", "all_public_nnunet"),
              ("allpublic_segformer", "all public (A), SegFormer-B2", "all_public_segformer")]
    fm_sicap = [(f"sicap_{m}", f"SICAPv2 {name}", f"sicap_fm_{m}") for m, name in
                [("uni", "UNI"), ("conch", "CONCH"), ("dinov2", "DINOv2-L"), ("unimedclip", "UniMed-CLIP"), ("virchow2", "Virchow2")]]
    few = [(f"fs_{init}_n{n}", f"{label} + {n} patients", init, n) for init, label in [("kar", "Karolinska"), ("imagenet", "ImageNet")]
           for n in (1, 2, 4, 8, "all")] + [(f"fs_public_n{n}", f"all public + {n} patients", "public", n) for n in (2, 4, 8)]
    out = []
    for mode in ["tile", "full"]:
        for key, label, path in site:
            if mode == "tile" or path in ("cv/imagenet", "cv/sicap"):
                out.append((key, label, "site", "ours", mode, CV(path)))
        for key, label, run in public + (fm_sicap if mode == "tile" else []):
            if (RUNS / run / "best.pt").exists() and (RUNS / run / "TRAINED").exists() or run in OLD:
                out.append((key, label, "public", "ours", mode, ALL(run)))
        if mode == "tile":
            for key, label, init, n in few:
                runs = [(f"fewshot/{init}/n{n}/fold{k}", f"cv/fewshot/fold{k}_n{n}.json", "test") for k in range(5)]
                if all((RUNS / r / "last.pt").exists() or (RUNS / r / "best.pt").exists() for r, _, _ in runs):
                    out.append((key, label, "few-shot", "ours", mode, runs))
            out += [("kar3814", "Karolinska 3,814 slides (home)", "home", "karolinska", mode,
                     [("karolinska_all_unet", "karolinska_all_split.json", "test")]),
                    ("ours", "ours U-Net (5 fold models)", "away", "karolinska", mode,
                     [(f"cv/imagenet/fold{k}", "karolinska_all_split.json", "test") for k in range(5)]),
                    ("sicap", "SICAPv2", "away", "karolinska", mode, [("sicap_pretrain", "karolinska_all_split.json", "test")]),
                    ("sicap", "SICAPv2 CV (home)", "home", "sicap", mode,
                     [(f"reverse/A/imagenet/fold{k}", f"cv/sicap_fold{k}.json", "test") for k in range(5)]),
                    ("ours", "ours U-Net (5 fold models)", "away", "sicap", mode,
                     [(f"cv/imagenet/fold{k}", "sicap_split.json", "all") for k in range(5)]),
                    ("kar3814", "Karolinska 3,814 slides", "away", "sicap", mode, [("karolinska_all_unet", "sicap_split.json", "all")])]
    return out


# public runs trained before TRAINED markers existed
OLD = {"sicap_pretrain", "sicap_pretrain_seed1", "sicap_r40", "sicap_r80", "karolinska_n20_unet", "karolinska_n155_unet",
       "karolinska_n419_unet", "karolinska_n1000_unet", "karolinska_all_unet", "sicap_fm_uni", "sicap_fm_conch",
       "sicap_fm_dinov2", "sicap_fm_unimedclip", "sicap_fm_virchow2"}


def folder(run, dataset, mode, split_file, split):
    return PRED / run / f"{dataset}_{mode}_{Path(split_file).stem}_{split}"


def checkpoint(run):
    r = RUNS / run
    return r / ("best.pt" if (r / "best.pt").exists() else "last.pt")       # few-shot runs keep their last weights


def jobs():
    seen, out = set(), []
    for key, label, group, dataset, mode, members in sets():
        for run, split_file, split in members:
            f = folder(run, dataset, mode, split_file, split)
            if f in seen:
                continue
            seen.add(f)
            cmd = ["protocol_predict.py", "--ckpt", str(checkpoint(run)), "--out", str(f), *DATA[dataset],
                   *MODES[mode], "--split", split]
            if dataset != "karolinska":
                cmd += ["--split-file", split_file]
            out.append((f"{run} -> {dataset}, {mode}", (f / "done.json").exists(), cmd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = jobs()
    print(f"{sum(d for _, d, _ in steps)} of {len(steps)} prediction folders done")
    if a.dry_run:
        for label, done, cmd in steps:
            print(f"{'done' if done else 'todo'}  {label}")
        return
    log = RUNS / "protocol_predict.log"
    for label, done, cmd in steps:
        if done:
            continue
        print(f"\n[{time.strftime('%m-%d %H:%M')}] {label}", flush=True)
        t0 = time.time()
        with log.open("a") as f:
            code = subprocess.run([sys.executable, str(P.script(cmd[0])), *cmd[1:]], cwd=P.INDEX, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"{label} failed (exit {code}); see {log}")
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("PREDICTIONS DONE; next: python protocol.py, then python protocol_report.py")


if __name__ == "__main__":
    main()
