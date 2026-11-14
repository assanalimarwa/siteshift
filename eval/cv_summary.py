"""
Summarise the patient cross-validation (run_cv.py): per-patient results, means with 95%
bootstrap confidence intervals over patients, and paired comparisons between configurations.

    python cv_summary.py                       # inference mode ctx256_blend_tta (default)
    python cv_summary.py --mode tile           # each tile alone

Every patient is scored once, by the fold model that did not train on it (or by a zero-shot
model that never trained on our tiles). Folds still missing are reported and left out; paired
comparisons use only patients scored by both configurations.

Metrics, fixed before the cross-validation was run (2026-10-19):
    dice                pixel Dice over all of the patient's tiles (main number)
    sensitivity         share of annotated tumour pixels predicted tumour
    false_tumour_rate   share of tumour-free tissue predicted tumour (glass excluded; tissue as in
                        prepare.tissue_mask)
    precision           share of predicted tumour pixels that are annotated tumour
The "mean" row averages patients (each patient counts once); "pooled" adds up all pixels, so
big patients (36 holds a quarter of the tiles) dominate it. CI = 2.5 / 97.5 percentiles of
10,000 bootstrap resamples of the patients (seed 0).

Writes cv/summary_<mode>.csv (one row per patient and configuration) and cv/summary_<mode>.md.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = P.INDEX
METRICS = ["dice", "sensitivity", "false_tumour_rate", "precision"]
REFERENCE = "imagenet"


def sources(mode):
    """configuration -> list of per_tile.csv paths that together cover the scored patients."""
    from run_cv import CONFIGS, ENSEMBLES, STAIN, STAINED, ZERO_SHOT
    n = len(json.loads((HERE / "cv" / "folds.json").read_text())["folds"])
    out = {}
    for name, run in ZERO_SHOT.items():
        p = P.RUNS / run / f"eval_all_test_{mode}" / "per_tile.csv"
        out[name] = [p] if p.exists() else []
    for name, run in STAINED.items():
        p = P.RUNS / run / f"eval_all_test_{mode}_stain-{Path(STAIN).stem}" / "per_tile.csv"
        out[name] = [p] if p.exists() else []
    for config in CONFIGS:
        out[config] = [p for k in range(n)
                       if (p := P.RUNS / "cv" / config / f"fold{k}" / f"eval_fold{k}_test_{mode}" / "per_tile.csv").exists()]
    for name, e in ENSEMBLES.items():
        out[name] = [p for k in range(n)
                     if (p := P.RUNS / "cv" / e["base"] / f"fold{k}" / f"eval_fold{k}_test_{mode}_ens-{e['tag']}" / "per_tile.csv").exists()]
    return {k: v for k, v in out.items() if v}


def per_patient(tiles):
    t = tiles.assign(patient=tiles.slide.str.split(".").str[0].astype(int))
    if "fp_tissue" not in t:
        raise SystemExit("per_tile.csv lacks fp_tissue / benign_tissue_px; re-run evaluate.py")
    p = t.groupby("patient")[["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]].sum()
    p["tiles"] = t.groupby("patient").size()
    return p


def rates(c):
    """Metrics from summed counts (works on a per-patient frame or on one row of totals)."""
    return pd.DataFrame(dict(
        dice=2 * c.tp / (2 * c.tp + c.fp + c.fn),
        sensitivity=c.tp / c.tumour_px,
        false_tumour_rate=c.fp_tissue / c.benign_tissue_px,
        precision=c.tp / (c.tp + c.fp),
    ))


def bootstrap_ci(values, n=10_000, seed=0):
    v = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def tau_b_ties(a, b, n=2000, seed=0):
    """Kendall's tau-b between the model orders of two benchmarks, where a benchmark ties a pair of
    models it cannot tell apart (95% normal interval of the mean paired per-unit difference includes 0).
    a, b: units x models arrays (units paired within a benchmark, independent between the two).
    tau-b from the pairwise orders: sum(sA * sB) / sqrt(sum|sA| * sum|sB|), s = -1 / 0 / +1 per pair.
    Returns tau, 95% interval from resampling each benchmark's units, share of resamples in which tau is
    undefined (a benchmark ties every pair), and the number of pairs each benchmark separates."""
    def signs(x):
        i, j = np.triu_indices(x.shape[1], 1)
        d = x[:, i] - x[:, j]
        m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
        return np.where(np.abs(m) > 1.96 * se, np.sign(m), 0.0)

    def tau(sa, sb):
        den = np.sqrt(np.abs(sa).sum() * np.abs(sb).sum())
        return (sa * sb).sum() / den if den else np.nan

    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    t = np.array([tau(signs(a[rng.integers(0, len(a), len(a))]), signs(b[rng.integers(0, len(b), len(b))])) for _ in range(n)])
    lo, hi = (np.nanpercentile(t, 2.5), np.nanpercentile(t, 97.5)) if not np.isnan(t).all() else (np.nan, np.nan)
    return tau(signs(a), signs(b)), (lo, hi), float(np.isnan(t).mean()), (int(np.abs(signs(a)).sum()), int(np.abs(signs(b)).sum()))


def fmt_tau_ties(res, n_pairs):
    t, (lo, hi), undefined, (sa, sb) = res
    if np.isnan(t):
        return f"undefined (separated pairs {sa} / {sb} of {n_pairs}: one benchmark ties every pair)"
    return (f"{t:+.2f} [{lo:+.2f}, {hi:+.2f}] (separated pairs {sa} / {sb} of {n_pairs}"
            + (f"; undefined in {undefined:.1%} of resamples" if undefined >= 0.001 else "") + ")")


def fmt(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def overview(per):
    rows = []
    for name, p in per.items():
        if not len(p):
            continue
        r = {"configuration": name, "patients": len(p)}
        for m in METRICS:
            r[m] = fmt(p[m].mean(), *bootstrap_ci(p[m]))
        pooled = rates(p[["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]].sum().to_frame().T).iloc[0]
        r["pooled dice"] = f"{pooled.dice:.3f}"
        r["pooled false tumour"] = f"{pooled.false_tumour_rate:.3f}"
        rows.append(r)
    return pd.DataFrame(rows)


def paired(per):
    """Each configuration minus REFERENCE on the patients both scored: mean [CI], patients better / worse."""
    comp = []
    if REFERENCE not in per:
        return pd.DataFrame()
    for name, p in per.items():
        both = p.index.intersection(per[REFERENCE].index)
        if name == REFERENCE or len(both) == 0:
            continue
        r = {"configuration": f"{name} - {REFERENCE}", "patients": len(both)}
        for m, higher_better in [("dice", True), ("sensitivity", True), ("false_tumour_rate", False)]:
            d = p.loc[both, m] - per[REFERENCE].loc[both, m]
            better, worse = ((d > 0).sum(), (d < 0).sum()) if higher_better else ((d < 0).sum(), (d > 0).sum())
            r[f"{m} difference"] = fmt(d.mean(), *bootstrap_ci(d))
            r[f"{m}: better / worse"] = f"{int(better)} / {int(worse)}"
        comp.append(r)
    return pd.DataFrame(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctx256_blend_tta", choices=["ctx256_blend_tta", "tile"])
    a = ap.parse_args()

    all_patients = sorted({p for f in json.loads((HERE / "cv" / "folds.json").read_text())["folds"] for p in f["test"]})
    found = sources(a.mode)
    if not found:
        raise SystemExit(f"no results for mode {a.mode} yet; run run_cv.py")

    per, lines = {}, [f"# Patient cross-validation, mode {a.mode}", ""]
    for name, paths in found.items():
        counts = per_patient(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
        per[name] = pd.concat([counts, rates(counts)], axis=1)
    missing = {n: sorted(set(all_patients) - set(p.index)) for n, p in per.items()}

    folds = json.loads((HERE / "cv" / "folds.json").read_text())
    biopsy = set(folds["biopsy_patients"])
    lines += ["## Test patients per fold, by specimen type", "",
              pd.DataFrame([dict(fold=f["fold"], biopsy=", ".join(str(x) for x in f["test"] if x in biopsy),
                                 large_section=", ".join(str(x) for x in f["test"] if x not in biopsy),
                                 val=", ".join(str(x) for x in f["val"])) for f in folds["folds"]]).to_markdown(index=False), ""]

    groups = {"all patients": None, "needle biopsies": True, "large sections": False}
    for title, want_biopsy in groups.items():
        keep = lambda idx: [x for x in idx if want_biopsy is None or (x in biopsy) == want_biopsy]
        sub = {n: p.loc[keep(p.index)] for n, p in per.items()}
        lines += [f"## {title[0].upper() + title[1:]}: mean over patients [95% CI]", "",
                  overview(sub).to_markdown(index=False), ""]
        if title == "all patients":
            lines += [f"- {n}: patients not scored yet: {miss}" for n, miss in missing.items() if miss] + [""]
        comp = paired(sub)
        if len(comp):
            lines += [f"### {title[0].upper() + title[1:]}: paired difference to {REFERENCE}, same patients [95% CI]", "",
                      comp.to_markdown(index=False), ""]

    table = pd.concat({n: p[["tiles"] + METRICS] for n, p in per.items()}, names=["configuration", "patient"])
    wide = table[["dice", "false_tumour_rate"]].unstack(0)
    wide.columns = [f"{m} {c}" for m, c in wide.columns]
    lines += ["## Per patient", "", wide.round(3).to_markdown(), ""]

    (HERE / "cv").mkdir(exist_ok=True)
    table.round(5).to_csv(HERE / "cv" / f"summary_{a.mode}.csv")
    (HERE / "cv" / f"summary_{a.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> cv/summary_{a.mode}.csv, cv/summary_{a.mode}.md")


if __name__ == "__main__":
    main()
