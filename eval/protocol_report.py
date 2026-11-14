"""
cv/protocol.md: the detection-delineation protocol (protocol.py) for every model and transfer direction of
run_protocol.py, with stacked error bars cv/protocol/error_bins_*.png. cv/protocol/summary.md (written by
hand once the results are in) goes first.

    python protocol_report.py

Mean over units (our patients, Karolinska slides, SICAPv2 regions) [95% bootstrap CI]. Sets scored by
several models on the same units (our five fold models on Karolinska / SICAPv2) use per unit the mean of
the models. Rankings: Kendall tau-b between the models' mean scores, 95% CI from resampling the units
(the same resample for both scores).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

from functools import lru_cache
from pathlib import Path

import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import protocol
from cv_summary import bootstrap_ci
from karolinska_all_report import paired
from protocol import BINS, INDEX, analyse, measures, pooled_clean_auc
from run_protocol import folder, sets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = P.INDEX
OUT = HERE / "cv" / "protocol"
TITLE = {("ours", "tile"): "Our 17 patients, tile mode (each tile alone)",
         ("ours", "full"): "Our 17 patients, full mode (context + blending + flips)",
         ("karolinska", "tile"): "Karolinska test (1,092 slides), tile mode",
         ("sicap", "tile"): "SICAPv2 (238 regions), tile mode"}
UNITS = {"ours": "patients", "karolinska": "slides", "sicap": "regions"}
BIN_LABEL = {"missed": "missed regions (detection)", "false": "false tumour > 50 um from any tumour (detection)",
             "fill": "incomplete fill of found regions (delineation)", "boundary": "boundary band +-50 um (delineation)"}
COLOR = {"missed": "#2a78d6", "false": "#eb6834", "fill": "#1baf7a", "boundary": "#eda100"}   # dataviz slots 1-4
STACK = ["missed", "false", "fill", "boundary"]                    # detection bins first, from the baseline
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


@lru_cache
def tissue(dataset):
    return pd.read_csv(HERE / INDEX[dataset], usecols=["image", "tissue_frac"])


def load(per_tile=False):
    """{(dataset, mode): {label: (group, per-unit measures)}} for every set whose predictions are all saved."""
    found = {}
    for key, label, group, dataset, mode, members in sets():
        dirs = [folder(run, dataset, mode, sf, sp) for run, sf, sp in members]
        if not all((d / "done.json").exists() for d in dirs):
            continue
        res = [analyse(d, per_tile) for d in dirs]
        parts = [measures(sl, ti, dataset, tissue(dataset)) for sl, ti in res]
        units = pd.concat(parts).groupby(level=0).mean()
        units["err_detection"] = units.err_missed + units.err_false
        units["err_delineation"] = units.err_fill + units.err_boundary
        if dataset == "sicap":
            disjoint = sum(len(x) for x in parts) == len(units)          # CV folds: each region once
            tiles = [pd.concat([ti for _, ti in res])] if disjoint else [ti for _, ti in res]
            units.attrs["pooled_clean_auc"] = float(np.mean([pooled_clean_auc(t, tissue(dataset)) for t in tiles]))
        found.setdefault((dataset, mode), {})[label] = (group, units)
    return found


def clean_auc(u, ci=True):
    if "pooled_clean_auc" in u.attrs:
        return f"{u.attrs['pooled_clean_auc']:.3f} (pooled)"
    return cell(u.clean_auc) if ci else f"{u.clean_auc.mean():.3f}"


def cell(v, pct=False):
    v = v.dropna()
    if not len(v):
        return ""
    lo, hi = bootstrap_ci(v)
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v.mean())} [{f(lo)}, {f(hi)}]"


def measures_table(models):
    rows = []
    for label, (group, u) in models.items():
        rows.append({"model": label, "group": group, "units": int(u.dice.notna().sum()), "Dice": cell(u.dice),
                     "clean-tile AUC": clean_auc(u), "region recall": cell(u.recall),
                     "recall (area)": f"{u.recall_area.mean():.3f}",
                     "false regions / mm2": f"{u.false_regions_mm2.mean():.3f}", "false-region area %": f"{100 * u.false_area.mean():.2f}",
                     "Dice in found regions": cell(u.dice_found), "boundary-tolerant Dice": f"{u.dice_tolerant.mean():.3f}",
                     "fill": f"{u.fill.mean():.3f}"})
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def bins_table(models, biopsy=None):
    """Mean err_<bin> over units; in brackets its share of the mean error (the bars of the figure), so the
    shares, the 'largest' column and the generated text agree."""
    rows = []
    for label, (group, u) in models.items():
        parts = [("all", u)] if biopsy is None else [("biopsies", u[u.index.isin(biopsy)]), ("large sections", u[~u.index.isin(biopsy)])]
        for spec, x in parts:
            c = x[[f"err_{b}" for b in BINS]].mean()
            r = {"model": label}
            if biopsy is not None:
                r["specimen"] = spec
            r |= {"1 - Dice": f"{c.sum():.3f}", **{b: f"{c[f'err_{b}']:.3f} ({100 * c[f'err_{b}'] / c.sum():.0f}%)" for b in STACK},
                  "detection share": f"{100 * (c.err_missed + c.err_false) / c.sum():.0f}%",
                  "largest": max(STACK, key=lambda b: c[f"err_{b}"])}
            rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


FM_NAMES = ["UNI", "CONCH", "DINOv2-L", "UniMed-CLIP", "Virchow2"]


def reference(label):
    """The site model a transfer model is compared with: same architecture, trained on our patients (CV)."""
    arch = next((f for f in FM_NAMES if label.startswith("SICAPv2 ") and label.endswith(f" {f}")), None)
    return f"ours {arch} (CV)" if arch else "ours U-Net (CV)"


def gap_table(models):
    """Transfer and few-shot models: per-unit difference to the site model in each error bin (positive = more
    error), mean [95% CI]; the four add up to the Dice gap."""
    rows = []
    for label, (group, u) in models.items():
        ref = reference(label)
        if group == "site" or ref not in models:
            continue
        r = models[ref][1]
        diff = lambda k: (u[k] - r[k]).dropna()
        rows.append({"model": label, "compared with": ref, "Dice gap": cell(diff("err_detection") + diff("err_delineation")),
                     **{b: cell(diff(f"err_{b}")) for b in STACK},
                     "detection (missed + false)": cell(diff("err_detection")),
                     "delineation (fill + boundary)": cell(diff("err_delineation"))})
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def false_split_table(models):
    """Post hoc: the false bin (false tumour > 50 um from tumour) split into isolated false regions (components
    that never come within 50 um of tumour) and spill of components that do reach tumour (over-painting beyond
    the band). With the spill counted as delineation instead of detection, the gap split changes."""
    rows = []
    for label, (group, u) in models.items():
        r = {"model": label, "false bin": f"{u.err_false.mean():.3f}", "isolated regions": f"{u.err_false_isolated.mean():.3f}",
             "attached spill": f"{u.err_false_attached.mean():.3f}",
             "spill share": f"{100 * u.err_false_attached.mean() / max(u.err_false.mean(), 1e-9):.0f}%"}
        ref = reference(label)
        if group != "site" and ref in models:
            s = models[ref][1]
            diff = lambda k: (u[k] - s[k]).dropna()
            r |= {"gap: isolated": cell(diff("err_false_isolated")), "gap: spill": cell(diff("err_false_attached")),
                  "gap: detection without spill": cell(diff("err_missed") + diff("err_false_isolated")),
                  "gap: delineation with spill": cell(diff("err_fill") + diff("err_boundary") + diff("err_false_attached"))}
        rows.append(r)
    return pd.DataFrame(rows).fillna("").to_markdown(index=False, disable_numparse=True)


def tau_ci(a, b, n=2000, seed=0):
    """Kendall tau-b between the model means of two units x models score tables, CI over unit resamples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t0 = kendalltau(np.nanmean(a, 0), np.nanmean(b, 0)).statistic
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        ts.append(kendalltau(np.nanmean(a[i], 0), np.nanmean(b[i], 0)).statistic)
    lo, hi = np.nanpercentile(ts, [2.5, 97.5])
    return f"{t0:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def rankings(models):
    """Tau for all models, and for the transfer models alone (trained on public data only: the cross-site case)."""
    subsets = {"all": list(models), "public only": [n for n, (g, _) in models.items() if g == "public"]}
    pairs = [("Dice", "detection (clean-tile AUC)"), ("Dice", "delineation (Dice in found regions)"),
             ("detection (clean-tile AUC)", "delineation (Dice in found regions)"),
             ("Dice", "region recall"), ("Dice", "boundary-tolerant Dice")]
    rows = []
    for subset, names in subsets.items():
        if len(names) < 4:
            continue
        m = lambda k: pd.concat({n: models[n][1][k] for n in names}, axis=1)
        score = {"Dice": m("dice"), "detection (clean-tile AUC)": m("clean_auc"), "delineation (Dice in found regions)": m("dice_found"),
                 "region recall": m("recall"), "boundary-tolerant Dice": m("dice_tolerant")}
        rows += [{"models": f"{subset} ({len(names)})", "ranking A": x, "ranking B": y,
                  "Kendall tau-b [95% CI]": tau_ci(score[x], score[y])} for x, y in pairs]
        if subset == "all":
            top = pd.DataFrame({k: [f"{n} ({v:.3f})" for n, v in s.mean().sort_values(ascending=False).head(5).items()]
                                for k, s in list(score.items())[:3]})
    return pd.DataFrame(rows).to_markdown(index=False), top.to_markdown(index=False)


def describe(label, u):
    c = {b: u[f"err_{b}"].mean() for b in BINS}
    big = max(STACK, key=c.get)
    total = sum(c.values())
    return (f"- **{label}**: 1 - Dice {total:.3f}, mostly {BIN_LABEL[big]} ({100 * c[big] / total:.0f}% of the error); "
            f"detection: clean-tile AUC {clean_auc(u, ci=False)}, region recall {u.recall.mean():.3f}, "
            f"{u.false_regions_mm2.mean():.3f} false regions / mm2; delineation: Dice in found regions "
            f"{u.dice_found.mean():.3f}, fill {u.fill.mean():.3f}.")


def figure(models, title, path):
    labels = list(models)[::-1]
    groups = [models[n][0] for n in labels]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.32 * len(labels)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    left = np.zeros(len(labels))
    for b in STACK:
        v = np.array([models[n][1][f"err_{b}"].mean() for n in labels])
        ax.barh(range(len(labels)), v, left=left, height=0.62, color=COLOR[b], edgecolor=SURFACE, linewidth=1.5,
                label=BIN_LABEL[b])
        left += v
    for i, total in enumerate(left):
        ax.text(total + 0.006, i, f"{total:.2f}", va="center", fontsize=8, color=INK2)
    for i in range(1, len(labels)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color=GRID, linewidth=1)
    ax.set_yticks(range(len(labels)), [f"{n}  [{g}]" for n, g in zip(labels, groups)], fontsize=8, color=INK)
    ax.set_xlim(0, max(left.max() * 1.12, 0.1))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("1 - Dice, split into the four error bins (mean over units)", fontsize=9, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8, frameon=False, labelcolor=INK,
                    title=title, title_fontsize=10, alignment="left")
    leg.get_title().set_color(INK)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    found, per_tile = load(), load(per_tile=True)
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    lines = []
    summary = OUT / "summary.md"
    if summary.exists():
        lines += [summary.read_text(encoding="utf-8"), ""]
    rules = protocol.__doc__[protocol.__doc__.index("Fixed before"):].strip()
    lines += ["# Detection-delineation protocol: definitions", "", "```text", rules, "```", ""]
    for (dataset, mode), title in TITLE.items():
        models = found.get((dataset, mode))
        if not models:
            continue
        png = OUT / f"error_bins_{dataset}_{mode}.png"
        figure(models, title, png)
        lines += [f"# {title}", "", f"Units: {UNITS[dataset]}. Figure: {png.relative_to(HERE / 'cv').as_posix()}", "",
                  "## Detection and delineation", "", measures_table(models), "",
                  "## Error bins: contribution to 1 - Dice (in brackets: share of the mean error)", "", bins_table(models), ""]
        if dataset == "ours":
            lines += ["## Error bins by specimen type", "", bins_table(models, biopsy), "",
                      "## Where the transfer gap comes from: difference to the site model per error bin", "",
                      "Per patient, transfer (or few-shot) model minus the site model of the same architecture; positive = "
                      "more error. The four bins add up to the Dice gap.", "", gap_table(models), "",
                      "## Post hoc: false bin = isolated false regions + spill attached to found tumour", "",
                      "Not part of the rules fixed in advance (2026-10-29). Isolated = predicted components with no pixel "
                      "within 50 um of annotated tumour; spill = the rest of the false bin, i.e. parts of components that "
                      "reach tumour but extend more than 50 um beyond it (over-painting). Gaps against the site model as above; "
                      "'detection without spill' = missed + isolated, 'delineation with spill' = fill + boundary + spill.", "",
                      false_split_table(models), ""]
            rel = [(a, "SICAPv2") for a in ["SICAPv2 relabelled r40", "SICAPv2 relabelled r80", "SICAPv2, seed 1"]]
            fm = {k: v[1] for k, v in models.items()}
            if all(a in fm for a, _ in rel):
                lines += ["## Check: relabelled SICAPv2 (only the training labels changed) vs SICAPv2",
                          "", "Seed 1 = the same training run with another seed: the noise reference.", "",
                          paired(fm, rel, metrics=["dice", "err_detection", "err_delineation", "clean_auc", "recall",
                                                   "recall_area", "false_regions_mm2", "false_area", "fill", "dice_found",
                                                   "dice_tolerant"]), ""]
            if len(models) >= 4:
                taus, top = rankings(models)
                lines += ["## Rankings: Dice vs detection vs delineation", "", taus, "", "Top five by each score:", "", top, ""]
            if mode == "tile" and (dataset, mode) in per_tile:
                pt = per_tile[(dataset, mode)]
                miss = lambda x: 100 * x.err_missed.mean() / x[[f"err_{b}" for b in BINS]].mean().sum()
                rows = [{"model": n, "missed share, slide regions": f"{miss(u):.0f}%",
                         "missed share, per tile": f"{miss(pt[n][1]):.0f}%",
                         "recall, slide regions": f"{u.recall.mean():.3f}", "recall, per tile": f"{pt[n][1].recall.mean():.3f}"}
                        for n, (_, u) in models.items() if n in pt]
                lines += ["## Sensitivity: regions per tile instead of per slide", "", pd.DataFrame(rows).to_markdown(index=False), ""]
        lines += ["## In words (generated)", "", *[describe(n, u) for n, (_, u) in models.items()], ""]
    (HERE / "cv" / "protocol.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {HERE / 'cv' / 'protocol.md'}")


if __name__ == "__main__":
    main()
