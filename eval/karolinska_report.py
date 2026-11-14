"""
cv/karolinska.md: U-Net trained on one source, tested on another (run_karolinska.py first).

    python karolinska_report.py

Sources: Karolinska (PANDA, needle biopsies), our data (17 patients), SICAPv2 (needle biopsies).
Test sets: the 121 Karolinska test slides (unit = slide) and our 17 patients (unit = patient, each
scored by the cross-validation fold model that never saw it, or by a model never trained on ours).
The five fold models trained on ours each score every Karolinska test slide; per slide their counts
are added up (and the spread of the five model-level means is reported).
Per unit: Dice and sensitivity (units with cancer), false-tumour rate = share of tumour-free tissue
predicted tumour (all units). Mean over units, 95% bootstrap interval over units; paired differences
on the same units. Both prediction modes: tile (each tile alone) and full (context, blending, flips).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

from pathlib import Path

import numpy as np
import pandas as pd

from cv_summary import bootstrap_ci

HERE = P.INDEX
RUNS = P.RUNS
COUNTS = ["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]
MODES = {"tile": "each tile alone", "ctx256_blend_tta": "full (context + blending + flips)"}


def units(paths, unit):
    """Per-unit metrics from one or more per_tile.csv files covering the same units (counts added)."""
    t = pd.concat([pd.read_csv(p) for p in paths])
    key = t.slide if unit == "slide" else t.slide.str.split(".").str[0].astype(int)
    g = t.groupby(key)[COUNTS].sum()
    cancer = g.tumour_px > 0
    return pd.DataFrame({"dice": (2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)).where(cancer),
                         "sensitivity": (g.tp / g.tumour_px.clip(lower=1)).where(cancer),
                         "false_tumour": g.fp_tissue / g.benign_tissue_px.clip(lower=1)})


def sources(mode):
    k = f"eval_karolinska_split_test_{mode}"
    folds_k = [RUNS / "cv" / "imagenet" / f"fold{j}" / k / "per_tile.csv" for j in range(5)]
    folds_o = [RUNS / "cv" / "imagenet" / f"fold{j}" / f"eval_fold{j}_test_{mode}" / "per_tile.csv" for j in range(5)]
    return {
        ("Karolinska", "Karolinska test"): ([RUNS / "karolinska_unet" / k / "per_tile.csv"], "slide"),
        ("ours (CV)", "Karolinska test"): (folds_k, "slide"),
        ("SICAPv2", "Karolinska test"): ([RUNS / "sicap_pretrain" / k / "per_tile.csv"], "slide"),
        ("Karolinska", "our patients"): ([RUNS / "karolinska_unet" / f"eval_all_test_{mode}" / "per_tile.csv"], "patient"),
        ("ours (CV)", "our patients"): (folds_o, "patient"),
        ("SICAPv2", "our patients"): ([RUNS / "sicap_pretrain" / f"eval_all_test_{mode}" / "per_tile.csv"], "patient"),
    }


def fmt(v):
    v = v.dropna()
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    import json
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    lines = ["# Karolinska (PANDA) and our data: U-Net trained on one, tested on the other", "",
             "U-Net (ResNet-34, resnet34_v2 recipe). Karolinska test: 121 slides (unit = slide; PANDA has no patient ids). "
             "Ours: 17 patients. Mean over units [95% CI].", ""]
    for mode, mode_name in MODES.items():
        found = {key: units(p, u) for key, (p, u) in sources(mode).items() if all(x.exists() for x in p)}
        if not found:
            continue
        rows = []
        for (train, test), x in found.items():
            r = {"trained on": train, "tested on": test, "units": len(x), "Dice": fmt(x.dice), "sensitivity": fmt(x.sensitivity),
                 "false-tumour rate": fmt(x.false_tumour)}
            if test == "our patients":
                r["Dice biopsies / large sections"] = (f"{x.dice[[p for p in x.index if p in biopsy]].mean():.3f} / "
                                                       f"{x.dice[[p for p in x.index if p not in biopsy]].mean():.3f}")
            rows.append(r)
        lines += [f"## {mode_name}", "", pd.DataFrame(rows).fillna("").to_markdown(index=False), ""]
        if ("ours (CV)", "Karolinska test") in found:
            spread = [units([p], "slide").dice.mean() for p in sources(mode)[("ours (CV)", "Karolinska test")][0]]
            lines.append(f"- Our five fold models on Karolinska test, Dice per model: {min(spread):.3f} to {max(spread):.3f}.")
        comps = []
        for a, b, test in [("Karolinska", "ours (CV)", "Karolinska test"), ("ours (CV)", "Karolinska", "our patients"),
                           ("Karolinska", "SICAPv2", "Karolinska test"), ("Karolinska", "SICAPv2", "our patients")]:
            if (a, test) in found and (b, test) in found:
                for metric in ["dice", "false_tumour"]:
                    d = (found[(a, test)][metric] - found[(b, test)][metric]).dropna()
                    lo, hi = bootstrap_ci(d)
                    comps.append({"on": test, "difference": f"{a} minus {b}", "metric": metric, "units": len(d),
                                  "mean [95% CI]": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", "better / worse": f"{(d > 0).sum()} / {(d < 0).sum()}"})
        if comps:
            lines += ["", "Paired differences on the same units:", "", pd.DataFrame(comps).to_markdown(index=False), ""]
    (HERE / "cv" / "karolinska.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
