"""
cv/error_breakdown.md: where each model's wrong pixels are, for every train -> test direction
(run_breakdown.py first; classes in breakdown.py).

    python error_breakdown_report.py

Per test set and model, per unit (our patient / Karolinska slide / SICAPv2 region; a model scoring a
unit several times - our five fold models on Karolinska or SICAPv2 - has its counts added):
  false tumour     share of the false-tumour pixels on glass / near the outline (<= 50 um) / far from it
  missed tumour    share of the missed pixels on glass / near the outline / far inside the region
  rates            far false-tumour rate = far false tumour / non-tumour tissue far from any outline
                   (false regions); near rate = near false tumour / non-tumour tissue near an outline
                   (outlines drawn wider); deep miss rate = far missed / tumour tissue deep inside a
                   region (missed cores); edge miss rate = near missed / tumour tissue near an outline
Mean over units [95% bootstrap CI] for the rates; shares are the mean over units of each unit's split.
"""

import pandas as pd

from breakdown import CLASSES, COLUMNS
from cv_summary import bootstrap_ci
from karolinska_all_report import HERE, RUNS

FOLDS = [RUNS / "cv" / "imagenet" / f"fold{k}" for k in range(5)]
TESTS = {
    "our 17 patients (unit = patient)": ("patient", {
        "ours (CV)": [f / f"eval_fold{k}_test_tile_breakdown" for k, f in enumerate(FOLDS)],
        "SICAPv2": [RUNS / "sicap_pretrain" / "eval_all_test_tile_breakdown"],
        "SICAPv2 closed r = 40 um": [RUNS / "sicap_r40" / "eval_all_test_tile_breakdown"],
        "SICAPv2 closed r = 80 um": [RUNS / "sicap_r80" / "eval_all_test_tile_breakdown"],
        "Karolinska (3,814 slides)": [RUNS / "karolinska_all_unet" / "eval_all_test_tile_breakdown"]}),
    "Karolinska test, 1,092 slides (unit = slide)": ("slide", {
        "Karolinska (3,814 slides)": [RUNS / "karolinska_all_unet" / "eval_karolinska_all_split_test_tile_breakdown"],
        "ours (5 fold models)": [f / "eval_karolinska_all_split_test_tile_breakdown" for f in FOLDS],
        "SICAPv2": [RUNS / "sicap_pretrain" / "eval_karolinska_all_split_test_tile_breakdown"]}),
    "SICAPv2 held-out, val + guard (unit = region)": ("slide", {
        "SICAPv2": [RUNS / "sicap_pretrain" / f"eval_sicap_{s}_tile_breakdown" for s in ["val", "guard"]],
        "ours (5 fold models)": [f / f"eval_sicap_{s}_tile_breakdown" for f in FOLDS for s in ["val", "guard"]],
        "Karolinska (3,814 slides)": [RUNS / "karolinska_all_unet" / f"eval_sicap_{s}_tile_breakdown" for s in ["val", "guard"]]}),
}


def units(dirs, unit):
    t = pd.concat([pd.read_csv(d / "per_tile.csv") for d in dirs])
    key = t.slide if unit == "slide" else t.slide.str.split(".").str[0].astype(int)
    g = t.groupby(key)[["tp", "fp", "fn", "tumour_px", *COLUMNS]].sum()
    out = pd.DataFrame(index=g.index)
    cancer = g.tumour_px > 0
    out["dice"] = (2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)).where(cancer)
    for k in ["fp", "fn"]:
        tot = g[[f"{k}_{c}" for c in CLASSES]].sum(1)
        for c in CLASSES:
            out[f"{k}_share_{c}"] = (g[f"{k}_{c}"] / tot).where(tot > 0)
    out["far_false_rate"] = g.fp_far / g.benign_far.clip(lower=1)
    out["near_false_rate"] = (g.fp_near / g.benign_near).where(g.benign_near > 0)
    out["deep_miss_rate"] = (g.fn_far / g.tumour_far).where(g.tumour_far > 0)
    out["edge_miss_rate"] = (g.fn_near / g.tumour_near).where(g.tumour_near > 0)
    return out


def ci(v, digits=3):
    v = v.dropna()
    if not len(v):
        return "-"
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    lines = ["# Where the wrong pixels are", "",
             "Tile mode. glass = bright pixel (glass, lumen, fat); near = tissue within 50 um of the annotated outline; "
             "far = tissue farther from it. Shares: mean over units of each unit's split of its wrong pixels.", ""]
    for test, (unit, models) in TESTS.items():
        rows = []
        for name, dirs in models.items():
            dirs = [d for d in dirs if (d / "per_tile.csv").exists()]
            if not dirs:
                continue
            u = units(dirs, unit)
            share = lambda k: " / ".join(f"{u[f'{k}_share_{c}'].mean():.2f}" for c in CLASSES)
            rows.append({"model": name, "units": len(u), "Dice": ci(u.dice),
                         "false tumour: glass / near / far": share("fp"), "far false-tumour rate": ci(u.far_false_rate),
                         "near false-tumour rate": ci(u.near_false_rate),
                         "missed: glass / near / far": share("fn"), "deep miss rate": ci(u.deep_miss_rate),
                         "edge miss rate": ci(u.edge_miss_rate)})
        if rows:
            lines += [f"## {test}", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    (HERE / "cv" / "error_breakdown.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
