"""
cv/reverse.md: does our dataset help on SICAPv2? (run_reverse.py first; finished parts are reported)

    python reverse_report.py

Design (make_sicap_folds.py, run_reverse.py): 5-fold cross-validation over SICAPv2 regions (groups of
overlapping tiles; slide ids are not in the Kaggle copy), every one of the 238 regions tested once.
A (SICAPv2 only), E (U-Net from random weights) and C (our fold-k model, then SICAPv2 fold k) are
pooled over their five test folds; B (our five fold models, no SICAPv2 training) scores every
region with each model and averages them per region.
Per region: Dice and sensitivity (regions with cancer), style-tolerant Dice (pixels within 50 um of an
annotated tumour boundary ignored), false tumour on tumour-free tissue split into near (within 50 um
of annotated tumour, likely coarse-outline stroma) and far (real errors). Tile AUC: score =
predicted tumour share of the tile, pooled over tiles. 95% intervals: 2,000 bootstrap resamples of
regions (seed 0), paired between conditions. Fold spread = range of the five fold-level means.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

from pathlib import Path

import numpy as np
import pandas as pd

HERE = P.INDEX
RUNS = P.RUNS
METHODS = {"imagenet": "U-Net (ResNet-34)", "fm_uni": "UNI", "fm_conch": "CONCH", "fm_unimedclip": "UniMed-CLIP",
           "fm_dinov2": "DINOv2-L", "fm_virchow2": "Virchow2"}
CONDS = {"A": "A: SICAPv2 only", "E": "E: SICAPv2 only, random init", "B": "B: ours only", "C": "C: ours, then SICAPv2"}
B_, SEED = 2000, 0
COUNTS = ["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px", "fp_near", "benign_near_px", "fp_far",
          "benign_far_px", "tp_st", "fp_st", "fn_st", "tumour_st"]
METRICS = {"dice": "Dice", "dice_st": "style-tolerant Dice", "sensitivity": "sensitivity", "false_tumour": "false tumour",
           "ft_near": "false tumour near", "ft_far": "false tumour far"}


def paths(m, c):
    if c == "B":
        return [RUNS / "cv" / m / f"fold{j}" / "eval_reverse_all" / "per_tile.csv" for j in range(5)]
    base = RUNS / "reverse" / ("E" if c == "E" else c) / ("" if c == "E" else m)
    if c == "E" and m != "imagenet":
        return []
    return [base / f"fold{k}" / "eval_reverse_test" / "per_tile.csv" for k in range(5)]


def region_metrics(t):
    g = t.groupby("slide")[COUNTS].sum()
    cancer = g.tumour_px > 0
    r = pd.DataFrame(index=g.index)
    r["dice"] = (2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)).where(cancer)
    r["dice_st"] = (2 * g.tp_st / (2 * g.tp_st + g.fp_st + g.fn_st).clip(lower=1)).where(g.tumour_st > 0)
    r["sensitivity"] = (g.tp / g.tumour_px.clip(lower=1)).where(cancer)
    r["false_tumour"] = (g.fp_tissue / g.benign_tissue_px.clip(lower=1)).where(g.benign_tissue_px > 0)
    r["ft_near"] = (g.fp_near / g.benign_near_px.clip(lower=1)).where(g.benign_near_px > 0)
    r["ft_far"] = (g.fp_far / g.benign_far_px.clip(lower=1)).where(g.benign_far_px > 0)
    return r


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return np.nan
    r = pd.Series(score).rank().values
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Condition:
    """Per-region metrics and per-tile scores of one method under one condition."""

    def __init__(self, c, tables):
        if c == "B":                                     # five models, each on every tile: average them
            self.parts = [region_metrics(t) for t in tables]
            self.regions = pd.concat(self.parts).groupby(level=0).mean()
            base = tables[0][["slide", "image"]].assign(label=tables[0].tumour_px > 0)
            self.tiles = base.assign(score=np.mean([t.set_index("image").loc[base.image, "score"].values for t in tables], 0))
        else:                                            # five folds, each region once: pool them
            self.parts = [region_metrics(t) for t in tables]
            self.regions = pd.concat(self.parts)
            pooled = pd.concat(tables)
            self.tiles = pooled[["slide", "image", "score"]].assign(label=pooled.tumour_px > 0)
        self.tile_groups = {r: g.index.values for r, g in self.tiles.reset_index(drop=True).groupby("slide")}

    def spread(self, metric):
        v = [p[metric].mean() for p in self.parts]
        return f"{min(v):.3f}-{max(v):.3f}"

    def auc_regions(self, regions):
        t = self.tiles.reset_index(drop=True)
        idx = np.concatenate([self.tile_groups[r] for r in regions if r in self.tile_groups])
        return auc(t.score.values[idx], t.label.values[idx])


def ci(est, samples):
    lo, hi = np.nanpercentile(samples, 2.5), np.nanpercentile(samples, 97.5)
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]", (lo > 0 or hi < 0)


def compare(x, y, rng):
    cells, sig = {}, {}
    for metric, label in METRICS.items():
        d = (x.regions[metric] - y.regions[metric]).dropna()
        if d.empty:
            continue
        v = d.values
        samp = v[rng.integers(0, len(v), (B_, len(v)))].mean(1)
        cells[label], sig[metric] = ci(v.mean(), samp)
        cells[label] += f" (n={len(v)})" if metric in ("dice", "false_tumour") else ""
    regions = sorted(set(x.regions.index) & set(y.regions.index))
    est = x.auc_regions(regions) - y.auc_regions(regions)
    samp = []
    for _ in range(B_ // 4):                             # AUC resampling is slower; 500 resamples
        pick = list(rng.choice(regions, len(regions)))
        samp.append(x.auc_regions(pick) - y.auc_regions(pick))
    cells["tile AUC"], sig["tile_auc"] = ci(est, samp)
    return cells, sig


def main():
    rng = np.random.default_rng(SEED)
    conds = {}
    for m in METHODS:
        for c in CONDS:
            ps = paths(m, c)
            if ps and all(p.exists() for p in ps):
                conds[(m, c)] = Condition(c, [pd.read_csv(p) for p in ps])
    if not conds:
        raise SystemExit("no condition is complete yet (all five folds); run run_reverse.py")
    n_regions = max(len(x.regions) for x in conds.values())
    lines = ["# Does our dataset help on SICAPv2? (reverse direction, cross-validated over SICAPv2 regions)", "",
             f"Every SICAPv2 region is a test region once ({n_regions} regions; Dice over regions with cancer). Tile mode, "
             "threshold 0.5, tiles enlarged to our scale. Intervals: paired bootstrap over regions. "
             "An earlier run on the 6-cancer-region guard split only is superseded by this one.", ""]

    rows = []
    for (m, c), x in conds.items():
        r = {"method": METHODS[m], "condition": CONDS[c]}
        r.update({label: round(x.regions[metric].mean(), 3) for metric, label in METRICS.items()})
        r["tile AUC"] = round(x.auc_regions(list(x.regions.index)), 3)
        r["Dice spread (5 folds / models)"] = x.spread("dice")
        rows.append(r)
    lines += ["## All conditions", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    summary = []
    for key, (c1, c2), title, meaning in [
            ("C-A", ("C", "A"), "C vs A: does pretraining on our data improve SICAPv2 results?", "starting from our data"),
            ("B-A", ("B", "A"), "B vs A: how far does our data alone transfer to SICAPv2?", "our data alone"),
            ("A-E", ("A", "E"), "A vs E: ImageNet vs random initialisation (U-Net)", "ImageNet initialisation"),
            ("C-E", ("C", "E"), "C vs E: our data vs random initialisation as the start (U-Net)", "starting from our data")]:
        tab, verdicts = [], []
        for m in METHODS:
            if (m, c1) in conds and (m, c2) in conds:
                cells, sig = compare(conds[(m, c1)], conds[(m, c2)], rng)
                tab.append({"method": METHODS[m], **cells})
                for metric in ("dice", "dice_st"):
                    if metric in sig:
                        d = (conds[(m, c1)].regions[metric] - conds[(m, c2)].regions[metric]).mean()
                        verdicts.append((METHODS[m], metric, d, sig[metric]))
        if not tab:
            continue
        lines += [f"## {title}", "", f"{CONDS[c1]} minus {CONDS[c2]}, paired over regions [95% CI].", "",
                  pd.DataFrame(tab).to_markdown(index=False), ""]
        parts = []
        for metric, label in [("dice", "Dice"), ("dice_st", "style-tolerant Dice")]:
            v = [x for x in verdicts if x[1] == metric]
            up = [n for n, _, d, s in v if s and d > 0]
            down = [n for n, _, d, s in v if s and d < 0]
            parts.append(f"{label}: clearly higher for {', '.join(up) or 'none'}; clearly lower for {', '.join(down) or 'none'}; "
                         f"no clear difference for the other {len(v) - len(up) - len(down)}")
        summary.append(f"- **{key}** ({meaning}): " + "; ".join(parts) + ".")
    lines += ["## Summary", "", *summary, ""]
    (HERE / "cv" / "reverse.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
