"""
Label-convention experiment (user 2026-11-10, item 4 of the final local jobs): the baseline U-Net recipe (cv47/imagenet:
train.py defaults + --epoch-tiles 60000 --limit-val 20000, seed 0, the fixed cv47 folds) trained on our tiles with
transformed labels, evaluated against the ORIGINAL labels (train/run_labels47.py).

    python labels47.py make        # transformed masks -> $SITESHIFT_DATA/labels47/<convention>/<slide>/masks/<tile>.png
    python labels47.py protocol    # protocol.py (parity-verified) on the finished folds and the matching baseline folds
    python labels47.py report      # -> cv47/labels_summary.md, cv47/labels_by_patient.csv

Conventions (fixed 2026-11-10, before any result):
  dilate50   tumour = every pixel within 50 um (Euclidean, 0.5 um/px) of an annotated tumour pixel
  erode50    tumour = annotated tumour pixels farther than 50 um from every non-tumour pixel
  Both are computed on the stitched slide (each tile with 101 px of its 8 neighbours; a missing tile counts as glass, not
  tumour, as in inference and protocol.py), not per tile, so tile borders do not shrink or grow the outlines. Purely
  geometric: dilation also covers glass. Train AND val masks of the fold use the transformed labels (checkpoint choice
  under the same convention); the tile set is the baseline's (min-tissue filter on the original tumour fraction).
  Test: the fold's test patients against the original labels, tile mode, compared with the matching baseline folds.
  Label Dice: Dice of the transformed masks against the originals, per slide over all its tiles; per patient the mean of
  its slides (as the model Dice).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

import protocol

protocol.INDEX["ours"] = "tiles47.csv"                      # as eval/protocol47.py: our 47 patients
from data import grid_position, load_mask, mask_path
from report47 import INFO, baseline, boot, per_patient, tissue
from threshold47 import mean_ci, paired

HERE = P.INDEX
CV = P.RESULTS / "cv47"                      # result tables (splits: P.INDEX / "cv47")
DATA = P.DATASET47
LAB = P.DATA_ROOT / "labels47"
PRED = P.PRED
CONVS = {"dilate50": "outlines dilated by 50 um", "erode50": "outlines eroded by 50 um"}
UM, PX = 50, 0.5
R = int(np.ceil(UM / PX)) + 1                                  # neighbour margin: the band radius + 1 px
BINS = ["missed", "fill", "boundary", "false"]


def run(conv, k):
    return f"cv47_labels/{conv}/fold{k}"


def pred_dir(conv, k):
    return PRED / (f"cv47_imagenet_fold{k}" if conv == "baseline" else run(conv, k).replace("/", "_")) / "ours47_tile"


def eval_dir(conv, k):
    r = f"cv47/imagenet/fold{k}" if conv == "baseline" else run(conv, k)
    return P.RUNS / r / f"eval_fold{k}_test_tile"


# ---------------------------------------------------------------- make
def make_slide(slide, images):
    pos = {grid_position(p): p for p in images}
    masks = {}
    for cr, p in pos.items():
        m = load_mask(DATA / mask_path(p))
        assert m.shape == (512, 512) and (m < 2).all(), (p, m.shape)
        masks[cr] = m.astype(bool)
    counts, n_edt, S = {c: np.zeros(3, np.int64) for c in CONVS}, 0, 512 + 2 * R
    for (c, r), p in pos.items():
        canvas = np.zeros((S, S), bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                m = masks.get((c + dc, r + dr))
                if m is None:
                    continue
                y0, x0 = R + dr * 512, R + dc * 512
                ys, ye, xs, xe = max(y0, 0), min(y0 + 512, S), max(x0, 0), min(x0 + 512, S)
                canvas[ys:ye, xs:xe] = m[ys - y0:ye - y0, xs - x0:xe - x0]
        o = masks[(c, r)]
        if not canvas.any() or canvas.all():
            out = {"dilate50": o, "erode50": o}
        else:
            n_edt += 1
            out = {"dilate50": ndi.distance_transform_edt(~canvas, sampling=PX)[R:R + 512, R:R + 512] <= UM,
                   "erode50": ndi.distance_transform_edt(canvas, sampling=PX)[R:R + 512, R:R + 512] > UM}
        for conv, x in out.items():
            dst = LAB / conv / mask_path(p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(x.astype(np.uint8) * 255).save(dst)
            counts[conv] += [(x & o).sum(), (x & ~o).sum(), (~x & o).sum()]
    return slide, {c: v.tolist() for c, v in counts.items()}, n_edt, len(pos)


def make():
    tiles = pd.read_csv(HERE / "tiles47.csv", usecols=["image", "slide"])
    groups = sorted(tiles.groupby("slide").image.apply(list).items(), key=lambda x: -len(x[1]))
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(make_slide, [g[0] for g in groups], [g[1] for g in groups]))
    rows = [dict(slide=s, convention=c, tp=v[0], fp=v[1], fn=v[2], tiles=n, edt_tiles=e) for s, cnt, e, n in res for c, v in cnt.items()]
    d = pd.DataFrame(rows)
    d["label_dice"] = 2 * d.tp / (2 * d.tp + d.fp + d.fn).clip(lower=1)
    d.to_csv(LAB / "label_dice_slides.csv", index=False)
    (LAB / "MADE.json").write_text(json.dumps(dict(
        conventions=CONVS, um=UM, um_per_px=PX, margin_px=R, slides=len(res), tiles=int(sum(r[3] for r in res)),
        edt_tiles=int(sum(r[2] for r in res)), minutes=round((time.time() - t0) / 60, 1), source=str(DATA),
        by="labels47.py make", when=time.strftime("%Y-%m-%d %H:%M")), indent=2))
    print(d.groupby("convention").label_dice.describe(), flush=True)


# ---------------------------------------------------------------- protocol + report
def done_folds(conv):
    return [k for k in range(5) if (eval_dir(conv, k) / "summary.json").exists() and (pred_dir(conv, k) / "done.json").exists()]


def run_protocol():
    for conv in CONVS:
        for k in done_folds(conv):
            for f in [pred_dir(conv, k), pred_dir("baseline", k)]:
                if not protocol.is_cached(f, False):
                    t0 = time.time()
                    s, _ = protocol.analyse(f)
                    print(f"{f}: {len(s)} slides, {time.time() - t0:.0f} s", flush=True)


def attribution(conv, folds):
    s = pd.concat([protocol.analyse(pred_dir(conv, k))[0] for k in folds])
    t = pd.concat([protocol.analyse(pred_dir(conv, k))[1] for k in folds])
    return protocol.measures(s, t, "ours", tissue())


def label_dice():
    d = pd.read_csv(LAB / "label_dice_slides.csv")
    d["p"] = d.slide.str.split(".").str[0].astype(int)
    return d.groupby(["convention", "p"]).label_dice.mean().unstack(0)


def report():
    LD = label_dice()
    lines = ["# Label-convention experiment, 47-patient folds, tile mode", "",
             f"_labels47.py report, {pd.Timestamp.now():%Y-%m-%d %H:%M}. Conventions and rules: labels47.py docstring "
             "(fixed before any result). Same recipe and folds as the baseline U-Net (cv47/imagenet); training (train + "
             "val) on transformed labels, test against the ORIGINAL labels, compared on the same patients with the "
             "matching baseline folds. Per-patient Dice = mean of the patient's slide Dice; 95% CIs: 10,000 patient "
             "resamples, seed 0; paired differences with patients better/worse and Wilcoxon p._", "",
             "## Transformed labels against the originals (all 47 patients, per-patient mean of slide Dice)", "",
             "| convention | label Dice vs original (mean [CI]) | median | min |", "|---|---|---|---|"]
    lines += [f"| {CONVS[c]} | {mean_ci(LD[c])} | {LD[c].median():.3f} | {LD[c].min():.3f} |" for c in CONVS]
    rows, attr, by_patient = [], [], []
    for conv in CONVS:
        folds = done_folds(conv)
        if not folds:
            continue
        t, _ = per_patient([eval_dir(conv, k) for k in folds])
        base = baseline("tile", t.index)
        a, ab = attribution(conv, folds), attribution("baseline", folds)
        for name, x, y in [(f"baseline U-Net (original labels)", base, ab), (f"U-Net, {CONVS[conv]}", t, a)]:
            r = {"convention": CONVS[conv], "model": name, "folds": ",".join(map(str, folds)), "patients": len(x),
                 "label Dice of the training convention": mean_ci(LD.loc[x.index, conv]) if x is t else "1",
                 "Dice per patient (mean [CI])": mean_ci(x.dice), "median": f"{x.dice.median():.3f}"}
            if x is t:
                r["vs baseline (diff [CI])"], r["better/worse"], r["Wilcoxon p"] = paired(t.dice - base.dice.reindex(t.index))
            else:
                r["vs baseline (diff [CI])"] = r["better/worse"] = r["Wilcoxon p"] = ""
            r.update({"clean-tile AUC": mean_ci(x.clean_auc), "sensitivity": mean_ci(x.sensitivity),
                      "precision": mean_ci(x.precision), "false-tumour rate": mean_ci(x.false_tumour)})
            rows.append(r)
            attr.append({"convention": CONVS[conv], "model": name, "patient-pooled Dice": f"{y.dice.mean():.4f}",
                         **{f"err {b}": f"{y[f'err_{b}'].mean():.4f}" for b in BINS},
                         **{f"share {b}": f"{100 * y[f'share_{b}'].mean():.0f}%" for b in BINS},
                         "recall (regions)": f"{y.recall.mean():.3f}", "dice_found": f"{y.dice_found.mean():.3f}"})
        d = {"convention": CONVS[conv], "model": "change (convention - baseline)"}
        d["clean-tile AUC"] = paired(t.clean_auc - base.clean_auc.reindex(t.index))[0]
        for b in BINS:
            d[f"err {b}"] = paired(a[f"err_{b}"] - ab[f"err_{b}"].reindex(a.index))[0]
        attr.append(d)
        bp = t[["dice", "clean_auc", "sensitivity", "precision", "false_tumour"]].add_suffix("_conv").join(
            base[["dice", "clean_auc"]].add_suffix("_base")).join(a[[f"err_{b}" for b in BINS]].add_suffix("_conv")).join(
            ab[[f"err_{b}" for b in BINS]].add_suffix("_base"))
        bp["label_dice"] = LD.loc[bp.index, conv]
        by_patient.append(bp.assign(convention=conv).join(INFO).rename_axis("p").reset_index())

    def table(rs):
        cols = list(dict.fromkeys(c for r in rs for c in r))
        return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rs]

    if rows:
        lines += ["", "## Results (test patients of the finished folds, original labels)", "", *table(rows), "",
                  "## Attribution (protocol.py, parity-verified Euclidean band; err = bin / (2 tp + fp + fn) per patient, "
                  "mean over patients; the change row is paired over patients, convention minus baseline)", "", *table(attr)]
        pd.concat(by_patient).to_csv(CV / "labels_by_patient.csv", index=False)
    (CV / "labels_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {CV / 'labels_summary.md'}")


if __name__ == "__main__":
    {"make": make, "protocol": run_protocol, "report": report}[sys.argv[1]]()
