"""
cv/karolinska_all.md: the U-Net trained on all Karolinska slides, compared with our earlier results.

    python karolinska_all_report.py

On our 17 patients (unit = patient), U-Nets with the same recipe trained on:
  ours (CV)            each patient scored by the cross-validation fold model that never saw it (reference)
  Karolinska, all      3,814 train slides (run_karolinska_all.py)
  Karolinska, 600      419 train slides (run_karolinska.py)
  SICAPv2              runs/sicap_pretrain
On the 1,092 Karolinska test slides (unit = slide): the all-slide and 600-slide Karolinska models
(scaling), our five fold models (reverse direction; per slide the mean of the five models) and the
SICAPv2 model. Home vs away: each model on its own domain's test units and on the other domain's.
Per unit: Dice and sensitivity (units with cancer), false-tumour rate (share of tumour-free tissue
predicted tumour), tile AUC (tissue tiles; does a tile's predicted tumour share rank tiles with cancer
above tiles without, independent of outline style), clean tile AUC (the same with only tiles at least
half under the tumour outline against tiles with none, so a coarse outline that merely grazes a tile
cannot make it a "cancer" tile). Mean over units [95% bootstrap CI]; paired differences on the same units.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cv_summary import bootstrap_ci

HERE = P.INDEX
RUNS = P.RUNS
COUNTS = ["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]
CLEAN = 0.5                            # clean tile AUC: tumour tiles at least this much under the outline


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_unit(paths, unit, tissue):
    t = pd.concat([pd.read_csv(p) for p in paths])
    key = t.slide if unit == "slide" else t.slide.str.split(".").str[0].astype(int)
    g = t.groupby(key)[COUNTS].sum()
    cancer = g.tumour_px > 0
    out = pd.DataFrame({"dice": (2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)).where(cancer),
                        "sensitivity": (g.tp / g.tumour_px.clip(lower=1)).where(cancer),
                        "false_tumour": g.fp_tissue / g.benign_tissue_px.clip(lower=1)})
    tt = t.merge(tissue, on="image")
    tt = tt[tt.tissue_frac >= 0.25].assign(score=lambda d: (d.tp + d.fp) / 512 ** 2, y=lambda d: d.tumour_px > 0)
    tkey = tt.slide if unit == "slide" else tt.slide.str.split(".").str[0].astype(int)
    out["tile_auc"] = tt.groupby(tkey).apply(lambda d: auc(d.score, d.y), include_groups=False)
    # clean tiles only: at least CLEAN of the tile under the tumour outline vs no tumour at all; tiles only
    # partly covered (where outline style decides the label) are left out
    cl = tt[(tt.tumour_px == 0) | (tt.tumour_px >= CLEAN * 512 ** 2)]
    ckey = cl.slide if unit == "slide" else cl.slide.str.split(".").str[0].astype(int)
    out["clean_auc"] = cl.groupby(ckey).apply(lambda d: auc(d.score, d.y), include_groups=False)
    return out


def mean_of_models(paths, unit, tissue):
    """Several models scoring the same units: per unit the mean of the models' metrics, plus each model's mean Dice."""
    each = [per_unit([p], unit, tissue) for p in paths]
    return pd.concat(each).groupby(level=0).mean(), [x.dice.mean() for x in each]


def cell(v):
    v = v.dropna()
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def table(found, biopsy=None):
    rows = []
    for name, x in found.items():
        r = {"trained on": name, "units": len(x), "Dice": cell(x.dice), "sensitivity": cell(x.sensitivity),
             "false-tumour rate": cell(x.false_tumour), "tile AUC": cell(x.tile_auc), "clean tile AUC": cell(x.clean_auc)}
        if biopsy is not None:
            r["Dice biopsies / large sections"] = (f"{x.dice[[p for p in x.index if p in biopsy]].mean():.3f} / "
                                                   f"{x.dice[[p for p in x.index if p not in biopsy]].mean():.3f}")
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


def paired(found, pairs, metrics=("dice", "sensitivity", "false_tumour", "tile_auc", "clean_auc")):
    rows = []
    for a, b in pairs:
        if a in found and b in found:
            for metric in metrics:
                d = (found[a][metric] - found[b][metric]).dropna()
                lo, hi = bootstrap_ci(d)
                rows.append({"difference": f"{a} minus {b}", "metric": metric, "units": len(d),
                             "mean [95% CI]": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", "higher / lower": f"{(d > 0).sum()} / {(d < 0).sum()}"})
    return pd.DataFrame(rows).to_markdown(index=False) if rows else ""


def home_away(on_ours, on_kar):
    """Dice and tile AUC of our model and the all-slide Karolinska model, each at home and away."""
    rows = []
    for model, home, away in [("ours (CV)", on_ours, on_kar), ("Karolinska, all 3,814 slides", on_kar, on_ours)]:
        if model in home and model in away:
            for metric in ["dice", "tile_auc", "clean_auc"]:
                h, w = home[model][metric].mean(), away[model][metric].mean()
                rows.append({"trained on": model, "metric": metric, "at home": f"{h:.3f}", "away": f"{w:.3f}", "drop": f"{h - w:+.3f}"})
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True) if rows else ""


KAR_RUNS = [("Karolinska, all 3,814 slides", "karolinska_all_unet", 3814), ("Karolinska, 1,000 slides", "karolinska_n1000_unet", 1000),
            ("Karolinska, 419 slides", "karolinska_n419_unet", 419), ("Karolinska, 155 slides", "karolinska_n155_unet", 155),
            ("Karolinska, 20 slides", "karolinska_n20_unet", 20),
            ("Karolinska, 419 slides, older setup (q95 tiles, 60 val slides)", "karolinska_unet", None)]


def scaling(on_ours, on_kar):
    """Karolinska models by training slides: on our patients (and the gap to our own models) and at home."""
    rows = []
    for name, _, n in sorted((r for r in KAR_RUNS if r[2]), key=lambda r: r[2]):
        if name not in on_ours:
            continue
        o = on_ours[name]
        r = {"train slides": n, "Dice on ours": f"{o.dice.mean():.3f}", "clean AUC on ours": f"{o.clean_auc.mean():.3f}"}
        if "ours (CV)" in on_ours:
            d = (on_ours["ours (CV)"].clean_auc - o.clean_auc).dropna()
            lo, hi = bootstrap_ci(d)
            r["clean AUC: ours minus it"] = f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if name in on_kar:
            r["Dice at home"], r["clean AUC at home"] = f"{on_kar[name].dice.mean():.3f}", f"{on_kar[name].clean_auc.mean():.3f}"
        rows.append(r)
    if "ours (CV)" in on_kar and rows:
        k = on_kar["ours (CV)"]
        rows.append({"train slides": "ours (CV) on Karolinska", "Dice on ours": "", "clean AUC on ours": "",
                     "clean AUC: ours minus it": "", "Dice at home": f"{k.dice.mean():.3f} (away)", "clean AUC at home": f"{k.clean_auc.mean():.3f} (away)"})
    return pd.DataFrame(rows).fillna("").to_markdown(index=False, disable_numparse=True) if rows else ""


def main():
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    ours_tissue = pd.read_csv(HERE / "tiles.csv")[["image", "tissue_frac"]]
    k_tissue = pd.read_csv(HERE / "karolinska_all_tiles.csv", usecols=["image", "tissue_frac"])
    lines = ["# U-Net trained on all Karolinska slides vs our earlier results", "",
             "Same U-Net recipe (ResNet-34, resnet34_v2). Mean over units [95% CI]. Karolinska has no patient ids (unit = slide).",
             "Ours (CV) on our patients: each patient scored by the fold model that never saw it. Ours (CV) on Karolinska: every "
             "slide scored by all five fold models, per slide the mean of the five.", ""]
    for mode, title in [("ctx256_blend_tta", "full mode (context + blending + flips)"), ("tile", "tile mode (each tile alone)")]:
        src = {"ours (CV)": [RUNS / "cv" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_{mode}" / "per_tile.csv" for k in range(5)],
               **{name: [RUNS / run / f"eval_all_test_{mode}" / "per_tile.csv"] for name, run, _ in KAR_RUNS},
               "SICAPv2": [RUNS / "sicap_pretrain" / f"eval_all_test_{mode}" / "per_tile.csv"]}
        on_ours = {k: per_unit(p, "patient", ours_tissue) for k, p in src.items() if all(x.exists() for x in p)}

        k = f"eval_karolinska_all_split_test_{mode}"
        on_kar, spread = {}, ""
        for name, run, _ in KAR_RUNS:
            if (RUNS / run / k / "per_tile.csv").exists():
                on_kar[name] = per_unit([RUNS / run / k / "per_tile.csv"], "slide", k_tissue)
        folds = [RUNS / "cv" / "imagenet" / f"fold{j}" / k / "per_tile.csv" for j in range(5)]
        if all(p.exists() for p in folds):
            on_kar["ours (CV)"], dices = mean_of_models(folds, "slide", k_tissue)
            spread = f"- Our five fold models on Karolinska test, mean Dice per model: {min(dices):.3f} to {max(dices):.3f}."
        if (RUNS / "sicap_pretrain" / k / "per_tile.csv").exists():
            on_kar["SICAPv2"] = per_unit([RUNS / "sicap_pretrain" / k / "per_tile.csv"], "slide", k_tissue)

        lines += [f"## Home vs away, {title}", "", "Mean Dice / tile AUC over each domain's test units; drop = at home minus away.", "",
                  home_away(on_ours, on_kar), ""]
        lines += [f"## Scaling: Karolinska models by number of training slides, {title}", "",
                  "On our 17 patients, and at home on the 1,092 Karolinska test slides (where scored in this mode). "
                  "Our CV models train on 11-13 patients (16-20 slides, ~35,000 tiles); a Karolinska slide has ~150 tiles.", "",
                  scaling(on_ours, on_kar), ""]
        lines += [f"## Our 17 patients, {title}", "", table(on_ours, biopsy), "", "Paired differences on the same patients:", "",
                  paired(on_ours, [("ours (CV)", "Karolinska, all 3,814 slides"), ("Karolinska, all 3,814 slides", "Karolinska, 419 slides"),
                                   ("Karolinska, all 3,814 slides", "SICAPv2")]), ""]
        if on_kar:
            lines += [f"## Karolinska test, all 1,092 slides, {title}", "", table(on_kar), "", spread, "",
                      "Paired differences on the same slides:", "",
                      paired(on_kar, [("Karolinska, all 3,814 slides", "ours (CV)"), ("Karolinska, all 3,814 slides", "Karolinska, 419 slides"),
                                      ("ours (CV)", "SICAPv2")]), ""]
    (HERE / "cv" / "karolinska_all.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
