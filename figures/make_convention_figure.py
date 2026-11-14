"""
Regenerate the convention-curve figure (Dice after closing, normalized surface Dice, dilation tolerance, signed boundary distance, ASSD, HD95)
from results/analysis/convention_curves.csv, the per-patient table written by eval/convention_curves.py. Needs numpy, pandas, matplotlib only.

    python figures/make_convention_figure.py [--csv results/analysis/convention_curves.csv] [--out results/analysis]

Cells are means over patients with a 95% bootstrap interval (10,000 resamples, seed 0), as in the paper.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"uni": "UNI + head", "conch": "CONCH + head", "dinov2": "DINOv2-L + head", "nnunet": "nnU-Net-style, site CV", "nnunet_allpublic": "nnU-Net-style, all-public",
         "unet": "U-Net baseline, site CV", "segformer": "SegFormer, site CV", "segformer_allpublic": "SegFormer, all-public", "virchow2": "Virchow2 + head"}
ORDER = ["uni", "conch", "dinov2", "nnunet", "segformer", "unet", "nnunet_allpublic", "segformer_allpublic"]
RADII, TOLS = (0, 20, 40, 60, 80), (20, 40, 80)


def boot(v, n=10000, seed=0):
    v = np.asarray(v[~np.isnan(v)], float)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "analysis" / "convention_curves.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "analysis")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    per = {k: df[df.model == k].set_index("patient") for k in ORDER if k in set(df.model)}
    has_x = all("assd_global" in p for p in per.values())
    fig, axs = plt.subplots(2 if has_x else 1, 4, figsize=(16, 8 if has_x else 4), squeeze=False)
    ax = axs[0]
    cm = plt.get_cmap("tab10")
    for j, (k, p) in enumerate(per.items()):
        c = cm(j)
        for a_, (names, xs, tt) in zip(ax[:3], [([f"close{r}" for r in RADII], RADII, "a  Dice after closing both masks"),
                                                ([f"nsd{t}" for t in TOLS], TOLS, "b  Normalized surface Dice"),
                                                ([f"tol{t}" for t in TOLS], TOLS, "c  Symmetric dilation tolerance")]):
            m = [p[n].mean() for n in names]
            ci = np.array([boot(p[n].to_numpy()) for n in names])
            a_.plot(xs, m, "-o", color=c, label=LABEL[k], ms=3)
            a_.fill_between(xs, ci[:, 0], ci[:, 1], color=c, alpha=0.15)
            a_.set_title(tt, loc="left", fontsize=10)
            a_.set_xlabel("radius / tolerance (um)")
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
        for a_, (col, tt) in zip(axs[1], [("assd_global", "e  ASSD, global (um)"), ("assd_det", "f  ASSD within detected regions (um)"),
                                          ("hd95_global", "g  HD95, global (um)"), ("hd95_det", "h  HD95 within detected regions (um)")]):
            for j, (k, p) in enumerate(per.items()):
                m = p[col].mean()
                lo, hi = boot(p[col].to_numpy())
                a_.errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="o", color=cm(j))
            a_.set_xticks(range(len(per)))
            a_.set_xticklabels([LABEL[k] for k in per], rotation=40, ha="right", fontsize=7)
            a_.set_title(tt, loc="left", fontsize=10)
    for a_ in axs.ravel():
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "fig_convention.pdf")
    fig.savefig(a.out / "fig_convention.png", dpi=170)
    print("wrote", a.out / "fig_convention.pdf")


if __name__ == "__main__":
    main()
