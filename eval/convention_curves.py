"""
Step 3 of the protocol validation: convention curves on real model predictions (CPU only), from the saved protocol exports (export/<model>/predictions/).

    python convention_curves.py [key ...]        # default: every export that is complete (uni conch dinov2 nnunet nnunet_allpublic, + unet / segformer when exported)

Per slide, on the stitched 0.5 um/px masks reduced to 4 um/px (8 x 8 block majority; tile masks never need more than 4 um at tolerances >= 20 um):
  Dice after closing BOTH masks at r = 0 / 20 / 40 / 60 / 80 um     closing = dilation by r then erosion by r, Euclidean (as the reference attribution.py)
  normalized surface Dice at tolerance 20 / 40 / 80 um              (|boundary of G within tol of the boundary of P| + |boundary of P within tol of the boundary of G|) / (|bG| + |bP|),
                                                                    boundary = mask pixels with a non-mask 8-neighbour
  symmetric dilation tolerance at the same radii                    (|P and dilation(G, r)| + |G and dilation(P, r)|) / (|P| + |G|)
  median signed boundary distance (um)                              boundary pixels of the predicted components that touch a DETECTED annotated region (>= 2,500 um^2 and >= 10% marked,
                                                                    as the reference); distance to the annotation boundary, + outside the annotation, - inside; median over those pixels
Per patient: mean over the patient's slides (the headline convention); then mean over the 47 patients with a 95% bootstrap CI (10,000 resamples, seed 0).
Writes convention_curves.md, convention_curves.csv (per patient and model), fig_convention.pdf / .png.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from data import grid_position  # noqa: E402

OUT = P.OUT
EXPORT = P.PRED
F, UM = 8, 4.0
S8 = np.ones((3, 3), bool)
RADII, TOLS = (0, 20, 40, 60, 80), (20, 40, 80)
MIN_PX, DET = 2500 / UM ** 2, 0.10
LABEL = {"uni": "UNI + head", "conch": "CONCH + head", "dinov2": "DINOv2-L + head", "nnunet": "nnU-Net-style, site CV", "nnunet_allpublic": "nnU-Net-style, all-public",
         "unet": "U-Net baseline, site CV", "segformer": "SegFormer, site CV", "segformer_allpublic": "SegFormer, all-public", "virchow2": "Virchow2 + head",
         "unimedclip": "UniMed-CLIP + head", "transunet": "TransUNet, site CV", "transunet_allpublic": "TransUNet, all-public"}
EXPECT = lambda k: 1 if k.endswith("_allpublic") else 5


def stitch(path):
    z = np.load(path)
    S, images = int(z["size"]), z["image"]
    b = S // F
    pos = np.array([grid_position(p) for p in images])
    c0, r0 = pos.min(0)
    G = np.zeros(((pos[:, 1].max() - r0 + 1) * b, (pos[:, 0].max() - c0 + 1) * b), bool)
    P = np.zeros_like(G)
    for i, (c, r) in enumerate(pos):
        for arr, key in ((G, "truth"), (P, "pred")):
            m = np.unpackbits(z[key][i], count=S * S).reshape(b, F, b, F).mean((1, 3)) >= 0.5
            arr[(r - r0) * b:(r - r0 + 1) * b, (c - c0) * b:(c - c0 + 1) * b] = m
    return str(z["slide"]), G, P


def dil(X, r):
    if r <= 0 or not X.any():
        return X.copy()
    return ndi.distance_transform_edt(~X, sampling=UM) <= r


def ero(X, r):
    return X.copy() if r <= 0 else ndi.distance_transform_edt(X, sampling=UM) > r


def dice(G, P):
    d = int(G.sum()) + int(P.sum())
    return 2 * int((G & P).sum()) / d if d else np.nan


def outline(X):
    return X & ~ndi.binary_erosion(X, S8, border_value=0)


def dist_to(B):
    return ndi.distance_transform_edt(~B, sampling=UM) if B.any() else np.full(B.shape, np.inf)


def slide_metrics(path):
    os.nice(10)
    name, G, P = stitch(path)
    r = {"slide": name}
    for rad in RADII:
        r[f"close{rad}"] = dice(ero(dil(G, rad), rad), ero(dil(P, rad), rad)) if rad else dice(G, P)
    bG, bP = outline(G), outline(P)
    dG, dP = dist_to(bG), dist_to(bP)
    for t in TOLS:
        den = int(bG.sum()) + int(bP.sum())
        r[f"nsd{t}"] = ((dP[bG] <= t).sum() + (dG[bP] <= t).sum()) / den if den else np.nan
        den2 = int(P.sum()) + int(G.sum())
        r[f"tol{t}"] = ((P & dil(G, t)).sum() + (G & dil(P, t)).sum()) / den2 if den2 else np.nan
    lab, n = ndi.label(G, S8)
    if n:
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        marked = np.bincount(lab[P], minlength=n + 1)[1:] if P.any() else np.zeros(n)
        det_ids = np.nonzero((area >= MIN_PX) & (marked >= DET * area))[0] + 1
        D = np.isin(lab, det_ids) if len(det_ids) else np.zeros_like(G)
        lp, npc = ndi.label(P, S8)
        touch = np.unique(lp[D & P]) if D.any() else np.array([], int)
        touch = touch[touch > 0]
        if len(touch):
            pb = outline(np.isin(lp, touch))
            sd = np.where(G, -ndi.distance_transform_edt(G, sampling=UM), ndi.distance_transform_edt(~G, sampling=UM))
            r["signed_um"] = float(np.median(sd[pb]))
    return r


def _dist_stats(a, b, d_to_a, d_to_b):
    """ASSD and HD95 (um) between boundary pixel sets a and b; d_to_a / d_to_b = distance maps to a / to b. nan when one set is empty."""
    if not (a.any() and b.any()):
        return np.nan, np.nan
    x, y = d_to_a[b], d_to_b[a]                       # b -> a and a -> b distances
    return (x.sum() + y.sum()) / (len(x) + len(y)), max(np.percentile(x, 95), np.percentile(y, 95))


def extra_metrics(path):
    """ASSD, HD95 (global and within detected regions) and the signed boundary bias of one slide, same masks (4 um/px) and detected-region rule as slide_metrics."""
    os.nice(10)
    name, G, P = stitch(path)
    r = {"slide": name, "gt_px": int(G.sum()), "pred_px": int(P.sum())}
    bG, bP = outline(G), outline(P)
    r["assd_global"], r["hd95_global"] = _dist_stats(bG, bP, dist_to(bG), dist_to(bP))
    lab, n = ndi.label(G, S8)
    if n:
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        marked = np.bincount(lab[P], minlength=n + 1)[1:] if P.any() else np.zeros(n)
        det_ids = np.nonzero((area >= MIN_PX) & (marked >= DET * area))[0] + 1
        D = np.isin(lab, det_ids) if len(det_ids) else np.zeros_like(G)
        r["n_detected"] = len(det_ids)
        lp, npc = ndi.label(P, S8)
        touch = np.unique(lp[D & P]) if D.any() else np.array([], int)
        touch = touch[touch > 0]
        if len(touch):
            pb = outline(np.isin(lp, touch))
            bD = outline(D)
            r["assd_det"], r["hd95_det"] = _dist_stats(bD, pb, dist_to(bD), dist_to(pb))
            sd = np.where(G, -ndi.distance_transform_edt(G, sampling=UM), ndi.distance_transform_edt(~G, sampling=UM))
            v = sd[pb]
            r["signed_mean_um"], r["signed_median_um"], r["frac_outside"] = float(v.mean()), float(np.median(v)), float((v > 0).mean())
    return r


def run_extra(key, workers=8):
    out = OUT / f"convention_extra_{key}.csv"
    if out.exists():
        return pd.read_csv(out)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(extra_metrics, files(key), chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{key} (extra: ASSD, HD95, signed bias): {len(df)} slides, {time.time() - t0:.0f} s", flush=True)
    return df


def files(key):
    d = EXPORT / key / "predictions"
    fs = sorted(d.glob("*/done.json")) if d.exists() else []
    return [p for f in fs for p in sorted(f.parent.glob("*.npz"))] if len(fs) >= EXPECT(key) else []


def boot(v, n=10000, seed=0):
    v = np.asarray(v[~np.isnan(v)], float)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(v):
    lo, hi = boot(v.to_numpy())
    return f"{np.nanmean(v):.3f} [{lo:.3f}, {hi:.3f}]"


def run(key, workers=8):
    out = OUT / f"convention_slides_{key}.csv"
    if out.exists():
        return pd.read_csv(out)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(slide_metrics, files(key), chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{key}: {len(df)} slides, {time.time() - t0:.0f} s", flush=True)
    return df


def main():
    keys = [k for k in sys.argv[1:] if not k.startswith("--")] or [k for k in LABEL if files(k)]
    per, native, slides_nan = {}, {}, {}
    import export_model as em
    W = int(os.environ.get("CONV_WORKERS", 8))
    for k in keys:
        s = run(k, W)
        if "--no-extra" not in sys.argv:
            x = run_extra(k, W)
            s = s.merge(x.drop(columns=["gt_px", "pred_px"], errors="ignore"), on="slide", how="left", suffixes=("", "_x"))
        slides_nan[k] = (int(s.assd_global.isna().sum()) if "assd_global" in s else 0, len(s))
        s["pt"] = s.slide.str.split(".").str[0].astype(int)
        per[k] = s.groupby("pt").mean(numeric_only=True)
        if k in em.MODELS or k in em.ALLPUBLIC:
            try:
                nat = pd.concat([pd.read_csv(d / "per_slide.csv") for d in em.eval_dirs(k)]).set_index("slide").dice
                native[k] = (s.set_index("slide").close0 - nat.reindex(s.slide).values).abs()
            except Exception:
                pass
    pd.concat({k: v for k, v in per.items()}, names=["model", "patient"]).to_csv(OUT / "convention_curves.csv")
    meta = pd.read_csv(str(P.RESULTS / "cv47" / "baseline_imagenet_by_patient.csv")).set_index("pt")
    L = ["# Protocol validation, step 3: convention curves on real predictions", "",
         "Per patient: mean over the patient's slides; table cells: mean over the patients [95% bootstrap CI, 10,000 resamples, seed 0]. Predictions: the protocol exports "
         "(tile mode, threshold 0.5). Masks reduced from 0.5 to 4 um/px (8 x 8 block majority) for speed. Closing and dilation are Euclidean, in um; "
         "signed boundary distance: + outside the annotation, - inside, median over the boundary pixels of predicted components touching a detected annotated region "
         "(>= 2,500 um^2 and >= 10% marked). Sets not on this server (SICAPv2, Karolinska 3,814, 600 mixed) are not included.", "",
         "Resolution check: |Dice at 4 um/px - native Dice| per slide, " + "; ".join(f"{LABEL[k]}: mean {v.mean():.4f}, max {v.max():.4f}" for k, v in native.items()) + ".", ""]
    cols = [("close0", "Dice r=0"), ("close20", "r=20"), ("close40", "r=40"), ("close60", "r=60"), ("close80", "r=80")]
    L += ["## Dice after closing both masks (r in um)", "", "| model | " + " | ".join(n for _, n in cols) + " | gain r=0 -> 80 |", "|---|" + "---|" * (len(cols) + 1)]
    for k, p in per.items():
        L.append(f"| {LABEL[k]} | " + " | ".join(cell(p[c]) for c, _ in cols) + f" | {(p.close80 - p.close0).mean():+.3f} |")
    L += ["", "## Normalized surface Dice (tolerance in um)", "", "| model | 20 | 40 | 80 |", "|---|---|---|---|"]
    for k, p in per.items():
        L.append(f"| {LABEL[k]} | " + " | ".join(cell(p[f"nsd{t}"]) for t in TOLS) + " |")
    L += ["", "## Symmetric dilation tolerance (radius in um)", "", "| model | 20 | 40 | 80 |", "|---|---|---|---|"]
    for k, p in per.items():
        L.append(f"| {LABEL[k]} | " + " | ".join(cell(p[f"tol{t}"]) for t in TOLS) + " |")
    L += ["", "## Median signed boundary distance (um; + outside the annotation; detected regions only)", "", "| model | all patients | biopsies | large sections |", "|---|---|---|---|"]
    for k, p in per.items():
        typ = meta.type.reindex(p.index)
        L.append(f"| {LABEL[k]} | {cell(p.signed_um)} | {cell(p.signed_um[typ == 'biopsy'])} | {cell(p.signed_um[typ == 'large'])} |")
    if all("assd_global" in p for p in per.values()):
        ext = [("assd_global", "ASSD global"), ("assd_det", "ASSD within detected regions"), ("hd95_global", "HD95 global"), ("hd95_det", "HD95 within detected regions")]
        L += ["", "## Average symmetric surface distance (ASSD) and 95th-percentile Hausdorff distance (HD95), um (lower is better)", "",
              "Boundaries = outline pixels of the masks at 4 um/px. ASSD = mean of all boundary-to-boundary distances in both directions; HD95 = the larger of the two directed 95th percentiles. "
              "Global: every annotated and predicted boundary of the slide (missed regions and false tumour add large distances); within detected regions: annotated boundary of the "
              "detected regions (>= 2,500 um^2 and >= 10% marked) against the boundary of the predicted components that touch them. A slide where one boundary set is empty has no value and is "
              "left out of that patient's mean.", "", "| model | " + " | ".join(n for _, n in ext) + " |", "|---|" + "---|" * len(ext)]
        for k, p in per.items():
            L.append(f"| {LABEL[k]} | " + " | ".join(cell(p[c]) for c, _ in ext) + " |")
        L += ["", "By specimen type (within detected regions):", "", "| model | ASSD biopsies | ASSD large sections | HD95 biopsies | HD95 large sections |", "|---|---|---|---|---|"]
        for k, p in per.items():
            typ = meta.type.reindex(p.index)
            L.append(f"| {LABEL[k]} | " + " | ".join(cell(p[c][typ == t]) for c in ("assd_det", "hd95_det") for t in ("biopsy", "large")) + " |")
        L += ["", "## Signed boundary bias (predicted boundary against the annotation; + = outside the annotation, i.e. over-segmentation; detected regions only)", "",
              "| model | median signed distance (um) | mean signed distance (um) | share of predicted boundary outside the annotation |", "|---|---|---|---|"]
        for k, p in per.items():
            L.append(f"| {LABEL[k]} | {cell(p.signed_median_um)} | {cell(p.signed_mean_um)} | {cell(p.frac_outside)} |")
        L += ["", "Slides without a value (global ASSD / HD95): " + "; ".join(f"{LABEL[k]}: {int(slides_nan[k][0])} of {slides_nan[k][1]}" for k in per) + "."]
    (OUT / "convention_curves.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_x = all("assd_global" in p for p in per.values())
    fig, axs = plt.subplots(2 if has_x else 1, 4, figsize=(16, 8 if has_x else 4), squeeze=False)
    ax = axs[0]
    cm = plt.get_cmap("tab10")
    for j, (k, p) in enumerate(per.items()):
        c = cm(j)
        for a, (names, xs, tt) in zip(ax[:3], [([f"close{r}" for r in RADII], RADII, "a  Dice after closing both masks"),
                                               ([f"nsd{t}" for t in TOLS], TOLS, "b  Normalized surface Dice"),
                                               ([f"tol{t}" for t in TOLS], TOLS, "c  Symmetric dilation tolerance")]):
            m = [p[n].mean() for n in names]
            ci = np.array([boot(p[n].to_numpy()) for n in names])
            a.plot(xs, m, "-o", color=c, label=LABEL[k], ms=3)
            a.fill_between(xs, ci[:, 0], ci[:, 1], color=c, alpha=0.15)
            a.set_title(tt, loc="left", fontsize=10)
            a.set_xlabel("radius / tolerance (um)")
        m = p.signed_um.mean()
        lo, hi = boot(p.signed_um.to_numpy())
        ax[3].errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c)
    ax[3].axhline(0, color="#888", lw=0.8)
    ax[3].set_xticks(range(len(per)))
    ax[3].set_xticklabels([LABEL[k] for k in per], rotation=40, ha="right", fontsize=7)
    ax[3].set_title("d  Median signed boundary distance (um)", loc="left", fontsize=10)
    ax[0].set_ylabel("per-patient mean")
    ax[0].legend(fontsize=7, frameon=False)
    if has_x:
        for a, (col, tt) in zip(axs[1], [("assd_global", "e  ASSD, global (um)"), ("assd_det", "f  ASSD within detected regions (um)"),
                                         ("hd95_global", "g  HD95, global (um)"), ("hd95_det", "h  HD95 within detected regions (um)")]):
            for j, (k, p) in enumerate(per.items()):
                m = p[col].mean()
                lo, hi = boot(p[col].to_numpy())
                a.errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="o", color=cm(j))
            a.set_xticks(range(len(per)))
            a.set_xticklabels([LABEL[k] for k in per], rotation=40, ha="right", fontsize=7)
            a.set_title(tt, loc="left", fontsize=10)
    for a in axs.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convention.pdf")
    fig.savefig(OUT / "fig_convention.png", dpi=170)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
