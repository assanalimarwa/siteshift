"""
Public benchmark vs our patients, with every unit tested once on both sides:

    python ranking_cv.py              # cv/ranking_cv.md

Same six methods (U-Net and the frozen encoders with the shared head) under three benchmarks:
  public      trained and tested on SICAPv2: condition A of run_reverse.py, 5-fold cross-validation
              over SICAPv2 regions, Dice per region with cancer (150 regions)
  site        trained and tested on our data: run_cv.py patient cross-validation, Dice per patient (17)
  transfer    trained on SICAPv2 only (run_cv.ZERO_SHOT runs), tested on our patients (17)
All in tile mode, threshold 0.5. For every pair of methods: paired difference on each benchmark with
a 95% bootstrap interval (2,000 resamples of regions or patients); a benchmark "separates" a pair
when the interval excludes 0. Kendall's tau between the rankings, with intervals from resampling
regions and patients independently.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import reverse_report as rr
from cv_summary import fmt_tau_ties, per_patient, rates, sources, tau_b_ties

HERE = P.INDEX
METHODS = {"imagenet": ("U-Net", "sicap_only"), "fm_uni": ("UNI", "sicap_fm_uni"), "fm_virchow2": ("Virchow2", "sicap_fm_virchow2"),
           "fm_conch": ("CONCH", "sicap_fm_conch"), "fm_dinov2": ("DINOv2-L", "sicap_fm_dinov2"),
           "fm_unimedclip": ("UniMed-CLIP", "sicap_fm_unimedclip")}
B, SEED = 2000, 0


def main():
    src = sources("tile")
    public, site, transfer = {}, {}, {}
    for m, (name, zs) in METHODS.items():
        ps = rr.paths(m, "A")
        if not all(p.exists() for p in ps):
            raise SystemExit(f"condition A of {name} is not complete")
        public[name] = rr.Condition("A", [pd.read_csv(p) for p in ps]).regions["dice"]
        site[name] = rates(per_patient(pd.concat([pd.read_csv(p) for p in src[m]]))).dice
        transfer[name] = rates(per_patient(pd.concat([pd.read_csv(p) for p in src[zs]]))).dice
    bench = {"public (SICAPv2, 150 regions)": pd.DataFrame(public).dropna(),
             "site (ours, trained on ours)": pd.DataFrame(site).dropna(),
             "transfer (ours, trained on SICAPv2)": pd.DataFrame(transfer).dropna()}
    names = list(bench["public (SICAPv2, 150 regions)"].columns)
    rng = np.random.default_rng(SEED)
    boot = {k: v.values[rng.integers(0, len(v), (B, len(v)))].mean(1) for k, v in bench.items()}   # B x methods

    lines = ["# Public benchmark vs our patients (cross-validated on both sides)", "",
             "Mean Dice per unit (SICAPv2 region with cancer / our patient), tile mode, threshold 0.5. [95% CI] by bootstrap.", ""]
    scores = pd.DataFrame({k: [f"{v[n].mean():.3f} [{np.percentile(boot[k][:, i], 2.5):.3f}, {np.percentile(boot[k][:, i], 97.5):.3f}]"
                               for i, n in enumerate(names)] for k, v in bench.items()}, index=names)
    ranks = pd.DataFrame({f"rank: {k.split(' (')[0]}": v.mean().rank(ascending=False).astype(int) for k, v in bench.items()})
    lines += ["## Scores and ranks", "", scores.join(ranks).reset_index().rename(columns={"index": "method"}).to_markdown(index=False), "",
              "Range of the six means: " + "; ".join(f"{k.split(' (')[0]} {v.mean().min():.3f}-{v.mean().max():.3f}" for k, v in bench.items()), ""]

    rows, counts = [], {k: 0 for k in bench}
    for i, j in itertools.combinations(range(len(names)), 2):
        r = {"pair": f"{names[i]} - {names[j]}"}
        for k in bench:
            d = boot[k][:, i] - boot[k][:, j]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            sep = lo > 0 or hi < 0
            counts[k] += sep
            r[k.split(" (")[0]] = f"{bench[k][names[i]].mean() - bench[k][names[j]].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if sep else ''}"
        rows.append(r)
    n_pairs = len(rows)
    lines += ["## Separation: paired differences between methods", "",
              "* = interval excludes 0. " + "; ".join(f"{k} separates {c} of {n_pairs} pairs" for k, c in counts.items()) + ".", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    lines += ["## Ranking agreement (Kendall's tau-b)", "",
              "Plain: the orders of the mean scores. With ties: a pair a benchmark cannot tell apart (95% normal interval of "
              "the paired difference includes 0) counts as tied on that benchmark. Intervals: 2,000 resamples of each "
              "benchmark's units.", ""]
    keys = list(bench)
    for a, b in itertools.combinations(keys, 2):
        tau = kendalltau(bench[a].mean().values, bench[b].mean().values).statistic
        taus = [kendalltau(boot[a][s], boot[b][s]).statistic for s in range(B)]
        lines.append(f"- {a.split(' (')[0]} vs {b.split(' (')[0]}: plain {tau:+.2f} [{np.nanpercentile(taus, 2.5):+.2f}, "
                     f"{np.nanpercentile(taus, 97.5):+.2f}]; with ties "
                     f"{fmt_tau_ties(tau_b_ties(bench[a][names].values, bench[b][names].values), n_pairs)}")
    best = {k: v.mean().idxmax() for k, v in bench.items()}
    regret = boot[keys[1]][:, names.index(best[keys[1]])] - boot[keys[1]][:, names.index(best[keys[0]])]
    lines += ["", f"- Best on the public benchmark: {best[keys[0]]}; best on our patients when trained on ours: {best[keys[1]]}. "
                  f"Choosing by the public benchmark costs {regret.mean():.3f} Dice on our patients "
                  f"[{np.percentile(regret, 2.5):.3f}, {np.percentile(regret, 97.5):.3f}].", ""]
    (HERE / "cv" / "ranking_cv.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
