"""
cv/label_harmonization.md: SICAPv2 U-Nets trained on the original vs region-style (closed) masks, on our
17 patients (run_label_harmonization.py first).

    python label_harmonization_report.py

Rows: original masks (runs/sicap_pretrain, seed 0), original masks seed 1 (rerun noise), closed with
r = 40 and 80 um; our CV U-Nets as the ceiling. Per patient: Dice, sensitivity, false-tumour rate,
tile AUC, clean tile AUC (karolinska_all_report.per_unit); mean [95% CI]; paired differences on the
same patients, also per specimen type. Plus each model's Dice on SICAPv2's held-out tiles with the
original masks (val + guard regions with cancer).
"""

import json

import pandas as pd

from cv_summary import bootstrap_ci
from karolinska_all_report import HERE, RUNS, paired, per_unit, table

MODELS = {"SICAPv2 original (seed 0)": "sicap_pretrain", "SICAPv2 original, seed 1": "sicap_pretrain_seed1",
          "SICAPv2 closed r = 40 um": "sicap_r40", "SICAPv2 closed r = 80 um": "sicap_r80"}
ORIG = "SICAPv2 original (seed 0)"


def sicap_home(run):
    parts = [RUNS / run / f"eval_sicap_{s}_tile" / "per_tile.csv" for s in ["val", "guard"]]
    if not all(p.exists() for p in parts):
        return None
    g = pd.concat([pd.read_csv(p) for p in parts]).groupby("slide")[["tp", "fp", "fn", "tumour_px"]].sum()
    g = g[g.tumour_px > 0]
    d = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    lo, hi = bootstrap_ci(d)
    return f"{d.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    tissue = pd.read_csv(HERE / "tiles.csv")[["image", "tissue_frac"]]
    lines = ["# Label harmonization: SICAPv2 trained on gland-level vs region-style masks, tested on our patients", "",
             "Same recipe and tiles as runs/sicap_pretrain; only the masks (or, for the noise reference, the seed) differ. "
             "Mean over our 17 patients [95% CI].", ""]
    for mode, title in [("ctx256_blend_tta", "full mode"), ("tile", "tile mode")]:
        found = {}
        for name, run in MODELS.items():
            p = RUNS / run / f"eval_all_test_{mode}" / "per_tile.csv"
            if p.exists():
                found[name] = per_unit([p], "patient", tissue)
        found["ours (CV), ceiling"] = per_unit([RUNS / "cv" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_{mode}" / "per_tile.csv"
                                                for k in range(5)], "patient", tissue)
        pairs = [(n, ORIG) for n in MODELS if n != ORIG]
        lines += [f"## {title}", "", table(found, biopsy), "", "Paired differences against the original-mask model:", "",
                  paired(found, pairs, metrics=("dice", "sensitivity", "false_tumour", "clean_auc")), ""]
        for kind, keep in [("needle biopsies", lambda p: p in biopsy), ("large sections", lambda p: p not in biopsy)]:
            sub = {n: x[[keep(p) for p in x.index]] for n, x in found.items()}
            lines += [f"Paired, {kind} only:", "", paired(sub, pairs, metrics=("dice", "sensitivity", "false_tumour")), ""]
    home = {n: sicap_home(r) for n, r in MODELS.items()}
    lines += ["## On SICAPv2's own held-out tiles (original masks), tile mode", "",
              pd.DataFrame([{"model": n, "Dice per region with cancer": v or "not scored"} for n, v in home.items()]).to_markdown(index=False), ""]
    (HERE / "cv" / "label_harmonization.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
