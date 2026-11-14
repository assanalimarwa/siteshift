"""
Per-dataset report from ds_props.py (and ds_distance.py when it has run) (CPU, a minute):

    python ds_report.py   ->  paper_figs/datasets/: <ds>_report.md, comparison_table.tex, fig_stain.pdf,
                              fig_label_geometry.pdf, data/summary.csv

Unit = slide (SICAPv2: overlap region rebuilt from the patches; RINGS: slide prefix P*_D*). Every number is a median over
units (mean for rates) with a 95% bootstrap CI over units (2,000 resamples, seed 0). Field measures are first reduced
per unit (median over its fields; tumour geometry: mean over its tumour fields). Region statistics use regions not cut
by the unit / patch border. Metadata: data/metadata.json (from the dataset papers and pages, with sources).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import COL_W, FULL_W, OI, plt, style
from ds_lib import AB_BINS, ADAPTERS, DATA, L_BINS, OUT, PANDA, load_fields

ORDER = ["ours", "sicapv2", "karolinska", "radboud"]          # RINGS excluded (user 2026-11-04): never trained on
LABEL = {"ours": "Ours", "sicapv2": "SICAPv2", "karolinska": "PANDA-Karolinska", "radboud": "PANDA-Radboud", "rings": "RINGS"}
COLOR = {"ours": OI["blue"], "sicapv2": OI["orange"], "karolinska": OI["green"], "radboud": OI["purple"], "rings": OI["vermillion"]}
rng = np.random.default_rng(0)


def ci(v, stat=np.median, n=2000):
    v = np.asarray(pd.Series(v).dropna(), float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    b = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return stat(v), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(t, d=3, pct=False):
    if np.isnan(t[0]):
        return "n/a"
    k = 100 if pct else 1
    return (f"{t[0] * k:.{d}f} [{t[1] * k:.{d}f}, {t[2] * k:.{d}f}]") + ("%" if pct else "")


def unit_tables(d):
    f = load_fields(d)
    ti = f[f.kind == "tissue"].copy()
    ti["pen_f"] = ti.pen >= 0.03          # >= 3% ink pixels (checked by eye, data/check_pen.png); fold / bubble
    ti["fold_f"], ti["bubble_f"] = ti.fold > 0.05, ti.bubbles > 0   # detectors failed the check: not reported
    per = ti.groupby("unit").agg(L=("L", "median"), a=("a", "median"), b=("b", "median"), blur=("blur", "median"),
                                 blurry=("blur", lambda x: (x < 30).mean()), tissue=("tissue", "median"),
                                 pen=("pen_f", "mean"), fold=("fold_f", "mean"), bubble=("bubble_f", "mean"))
    tu = f[f.kind == "tumour"].groupby("unit")[["gap_fill_40", "glass_in_tumour", "lumen_in_tumour", "regions",
                                                "outline_per_1000um2"]].mean()
    u = pd.read_csv(DATA / f"{d}_units.csv").drop_duplicates("unit").set_index("unit")
    allu = pd.read_csv(DATA / f"{d}_units_all.csv")
    allu["share"] = (allu.tumour_mm2 / allu.tissue_mm2).clip(upper=1)
    reg = pd.read_csv(DATA / f"{d}_regions.csv") if (DATA / f"{d}_regions.csv").exists() else pd.DataFrame(columns=["unit", "area_um2", "roughness", "cut"])
    reg["cut"] = reg.cut.astype(str) == "True"
    return f, per, tu, u, allu, reg


def gleason(d, units):
    """Gleason information the labels allow."""
    if d in ("karolinska", "radboud"):
        lab = pd.read_csv(PANDA / ("Karolinska" if d == "karolinska" else "Panda") / "labels.csv")
        vc = lab.gleason_score.replace({"negative": "0+0"}).value_counts()
        out = {"slides by Gleason score (all slides)": ", ".join(f"{k} {v:,}" for k, v in vc.items())}
        if d == "radboud":
            import tifffile
            ad = ADAPTERS[d]()
            px = np.zeros(6)
            for s in units:
                with tifffile.TiffFile(ad.msk / f"{s}_mask.tiff") as tf:
                    m = tf.series[0].levels[2].asarray()
                m = m[..., 0] if m.ndim == 3 else m
                px += np.bincount(m.ravel(), minlength=6)[:6]
            c = px[3:6] / px[3:6].sum()
            out["cancer pixels by Gleason pattern (sampled slides)"] = f"pattern 3 {c[0]:.0%}, 4 {c[1]:.0%}, 5 {c[2]:.0%}"
        return out
    if d == "sicapv2":
        from pathlib import Path
        t = pd.read_csv(P.INDEX / "sicap_tiles.csv")
        lv = t[[c for c in t.columns if c.startswith("level_")]].sum()
        lv = lv / lv.sum()
        return {"cancer pixels by mask grey level (Kaggle copy; mapping to Gleason patterns not documented)":
                ", ".join(f"{k[6:]} {v:.0%}" for k, v in lv.items() if v >= 0.005)}
    return {"Gleason": "not annotated (tumour vs non-tumour only)"}


def summarise():
    rows, tables = {}, {}
    for d in ORDER:
        if not (DATA / f"{d}_units.csv").exists():
            continue
        f, per, tu, u, allu, reg = unit_tables(d)
        ru = reg[~reg.cut]
        withT = allu[allu.tumour_mm2 > 0]
        r = {"units (all)": len(allu), "units sampled": len(u), "fields (tissue / tumour)": f"{(f.kind == 'tissue').sum():,} / {(f.kind == 'tumour').sum():,}",
             "units with tumour": f"{len(withT):,}/{len(allu):,} ({len(withT) / len(allu):.0%})",
             "tumour share of tissue per unit": fmt(ci(withT.share), 1, True)}
        if allu.patient.notna().any():
            pp = allu.groupby("patient")[["tumour_mm2", "tissue_mm2"]].sum()
            r["tumour share of tissue per patient"] = fmt(ci((pp.tumour_mm2 / pp.tissue_mm2).clip(upper=1)), 1, True) + f" (n = {len(pp)})"
        r.update({
            "H-E angle, deg (Macenko)": fmt(ci(u.HE_angle), 1), "mean H optical density": fmt(ci(u.H_od)),
            "mean E optical density": fmt(ci(u.E_od)), "L* (tissue)": fmt(ci(per.L), 1), "a* (tissue)": fmt(ci(per.a), 1),
            "b* (tissue)": fmt(ci(per.b), 1), "blur: Laplacian variance": fmt(ci(per.blur), 0),
            "blurry fields (variance < 30)": fmt(ci(per.blurry, np.mean), 1, True),
            "fields with pen ink (>= 3% of the field)": fmt(ci(per.pen, np.mean), 1, True),
            "tissue fraction of tissue fields": fmt(ci(per.tissue), 2),
            "tumour regions >= 2,500 um^2 (uncut)": f"{len(ru):,} ({reg.cut.mean():.0%} of {len(reg):,} cut by a border)",
            "region area, mm^2 (median [CI])": fmt(ci(ru.area_um2 / 1e6), 4),
            "region roughness P^2/(4 pi A)": fmt(ci(ru.roughness), 2),
            "gap fill at 40 um": fmt(ci(tu.gap_fill_40), 1, True), "glass inside tumour": fmt(ci(tu.glass_in_tumour), 1, True),
            "lumen inside tumour": fmt(ci(tu.lumen_in_tumour), 1, True), "tumour regions per 256-um window": fmt(ci(tu.regions), 2),
            "outline, um per 1,000 um^2": fmt(ci(tu.outline_per_1000um2), 1)})
        r.update(gleason(d, list(u.index)))
        rows[d] = r
        tables[d] = dict(per=per, tu=tu, u=u, allu=allu, reg=reg)
    return rows, tables


def fig_stain(tables):
    style()
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.0), gridspec_kw=dict(wspace=0.38, left=0.07, right=0.99, bottom=0.2, top=0.88))
    for d, t in tables.items():
        u, per = t["u"], t["per"]
        axs[0].scatter(u.H_od, u.E_od, s=4, color=COLOR[d], alpha=0.7, lw=0, label=LABEL[d])
        axs[1].scatter(per.a, per.b, s=4, color=COLOR[d], alpha=0.7, lw=0)
        h = np.load(DATA / f"{d}_hist.npz")["L"]
        c = (L_BINS[1:] + L_BINS[:-1]) / 2
        axs[2].plot(c, h / h.sum() / np.diff(L_BINS), color=COLOR[d], lw=1.0)
    axs[0].set(xlabel="Mean haematoxylin OD", ylabel="Mean eosin OD")
    axs[1].set(xlabel="a* (tissue, per unit)", ylabel="b*")
    axs[2].set(xlabel="L* of tissue pixels", ylabel="Density")
    for ax, t in zip(axs, ["a  Stain amount (Macenko)", "b  Colour", "c  Lightness"]):
        ax.set_title(t, loc="left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR[d], label=LABEL[d]) for d in tables],
                  frameon=False, fontsize=6, loc="best", handletextpad=0.1)
    fig.savefig(OUT / "fig_stain.pdf")
    fig.savefig(DATA / "preview_fig_stain.png", dpi=200)


def fig_geometry(tables):
    style()
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.0), gridspec_kw=dict(wspace=0.38, left=0.07, right=0.99, bottom=0.2, top=0.88))
    bins = np.logspace(np.log10(2500e-6), 2, 30)
    names = [d for d in tables]
    for i, d in enumerate(names):
        reg = tables[d]["reg"]
        ru = reg[~reg.cut]
        if len(ru):
            a = np.sort(ru.area_um2 / 1e6)
            axs[0].plot(a, np.arange(1, len(a) + 1) / len(a), color=COLOR[d], lw=1.0)
            x = np.sort(ru.roughness)
            axs[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[d], lw=1.0, label=LABEL[d])
        g = (tables[d]["tu"].gap_fill_40.dropna() * 100).clip(lower=0.1)      # log axis: floor 0.1%
        axs[2].boxplot(g, positions=[i], widths=0.55, showfliers=False, medianprops=dict(color="black", lw=0.8),
                       boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        axs[2].scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=3, color=COLOR[d], alpha=0.6, lw=0, zorder=3)
    axs[0].set(xscale="log", xlabel="Tumour region area (mm$^2$)", ylabel="Cumulative share of regions")
    axs[1].set(xscale="log", xlabel="Roughness P$^2$/(4$\\pi$A)", ylabel="Cumulative share of regions")
    axs[2].set_xticks(range(len(names)), [LABEL[d].replace("PANDA-", "") for d in names], rotation=30, ha="right")
    axs[2].set(ylabel="Gap fill at 40 µm (%)", yscale="log")
    axs[1].legend(frameon=False, fontsize=6, loc="lower right")
    for ax, t in zip(axs, ["a  Region sizes", "b  Outline roughness", "c  Gaps a 40-µm closing fills"]):
        ax.set_title(t, loc="left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig_label_geometry.pdf")
    fig.savefig(DATA / "preview_fig_label_geometry.png", dpi=200)


def main():
    rows, tables = summarise()
    s = pd.DataFrame(rows)
    s.columns = [LABEL[c] for c in s.columns]
    s.to_csv(DATA / "summary.csv")
    print(s.to_string())
    fig_stain(tables)
    fig_geometry(tables)
    return rows, tables


if __name__ == "__main__":
    main()
