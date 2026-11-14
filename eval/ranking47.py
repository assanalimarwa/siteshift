"""
Ranking test on 47 patients (user 2026-11-12, item 4): does SICAPv2's held-out set rank the methods as our site does?

    python ranking47.py              # -> cv47/ranking47.md, cv47/ranking47_scores.csv

Methods with both sides: U-Net (ResNet-34), UNI, Virchow2, CONCH, DINOv2-L (frozen encoder + the shared head). UniMed-CLIP
has a SICAPv2 model but no 47-patient site model, so it is left out (its SICAPv2 score is listed for reference).
  SICAPv2   trained on SICAPv2 only (run_cv.ZERO_SHOT runs), Dice per held-out SICAPv2 region with cancer (val + guard,
            ranking.public_scores), tile mode, threshold 0.5
  site      trained on our data, 47-patient 5-fold patient CV, per-patient Dice (mean of the patient's slide Dice), tile
            mode: U-Net = cv47/imagenet (4090), encoders = the H100 site CV exports (results/cv47/<model>_by_patient.csv, export_model.py)
Separation: paired difference per pair of methods on each side with a 95% bootstrap CI (10,000 resamples of regions /
patients, seed 0); a side separates a pair when the CI excludes 0. Agreement: Kendall's tau-b between the two orders of
the mean scores, CI from resampling both sides independently; also tau-b with ties for pairs a side cannot tell apart
(cv_summary.tau_b_ties).
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

from cv_summary import fmt_tau_ties, tau_b_ties
from ranking import public_scores

HERE = P.INDEX
CV = P.RESULTS / "cv47"                      # result tables (splits: P.INDEX / "cv47")
METHODS = {"U-Net (ResNet-34)": ("sicap_pretrain", "baseline"), "UNI": ("sicap_fm_uni", "uni"),
           "Virchow2": ("sicap_fm_virchow2", "virchow2"), "CONCH": ("sicap_fm_conch", "conch"), "DINOv2-L": ("sicap_fm_dinov2", "dinov2")}
B, SEED = 10000, 0


def site_scores(key):
    d = pd.read_csv(CV / f"{'uni' if key == 'baseline' else key}_by_patient.csv").set_index("patient")
    return d["baseline_dice" if key == "baseline" else "dice"], d.type


def main():
    P = pd.DataFrame({m: public_scores(run) for m, (run, _) in METHODS.items()}).dropna()
    O = pd.DataFrame({m: site_scores(k)[0] for m, (_, k) in METHODS.items()})
    typ = site_scores("uni")[1]
    assert len(O) == 47 and not O.isna().any().any()
    models = list(METHODS)
    rng = np.random.default_rng(SEED)
    Pb = P.values[rng.integers(0, len(P), (B, len(P)))].mean(1)
    Ob = O.values[rng.integers(0, len(O), (B, len(O)))].mean(1)
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    rank = lambda s: s.rank(ascending=False).astype(int)
    score = pd.DataFrame({
        "method": models,
        "SICAPv2 held-out (trained on SICAPv2)": [f"{P[m].mean():.3f} [{ci(Pb[:, i])[0]:.3f}, {ci(Pb[:, i])[1]:.3f}]" for i, m in enumerate(models)],
        "rank SICAPv2": rank(P.mean()).values,
        "our site, 47 patients (site CV)": [f"{O[m].mean():.3f} [{ci(Ob[:, i])[0]:.3f}, {ci(Ob[:, i])[1]:.3f}]" for i, m in enumerate(models)],
        "rank site": rank(O.mean()).values,
        "site: biopsies (25)": [f"{O.loc[typ == 'biopsy', m].mean():.3f}" for m in models],
        "site: large sections (22)": [f"{O.loc[typ == 'large', m].mean():.3f}" for m in models]})
    um = public_scores("sicap_fm_unimedclip")
    pairs = []
    for i, j in itertools.combinations(range(len(models)), 2):
        dp, do = Pb[:, i] - Pb[:, j], Ob[:, i] - Ob[:, j]
        sep = lambda d: "yes" if ci(d)[0] > 0 or ci(d)[1] < 0 else "no"
        flip = np.sign(P.iloc[:, i].mean() - P.iloc[:, j].mean()) != np.sign(O.iloc[:, i].mean() - O.iloc[:, j].mean())
        pairs.append({"pair": f"{models[i]} - {models[j]}",
                      "SICAPv2 difference [CI]": f"{dp.mean():+.3f} [{ci(dp)[0]:+.3f}, {ci(dp)[1]:+.3f}]", "SICAPv2 separates": sep(dp),
                      "site difference [CI]": f"{do.mean():+.3f} [{ci(do)[0]:+.3f}, {ci(do)[1]:+.3f}]", "site separates": sep(do),
                      "order flips": "yes" if flip else "no"})
    pairs = pd.DataFrame(pairs)
    taus = np.array([kendalltau(Pb[b], Ob[b]).statistic for b in range(B)])
    tau = kendalltau(P.mean().values, O.mean().values).statistic
    n_sp, n_ss = (pairs["SICAPv2 separates"] == "yes").sum(), (pairs["site separates"] == "yes").sum()
    lines = ["# Ranking test, 47 patients: SICAPv2 held-out vs our site", "",
             f"_ranking47.py, {pd.Timestamp.now():%Y-%m-%d %H:%M}. Rules in its docstring. SICAPv2 held-out: {len(P)} regions with "
             f"cancer; site: {len(O)} patients ({(typ == 'biopsy').sum()} biopsies, {(typ == 'large').sum()} large sections). Tile mode, "
             f"threshold 0.5; 95% bootstrap CIs, {B:,} resamples, seed 0. UniMed-CLIP has no 47-patient site model and is left out "
             f"(SICAPv2 held-out Dice {um.mean():.3f})._", "",
             "## Scores", "", score.to_markdown(index=False), "",
             "## Separation", "",
             f"**SICAPv2's held-out set separates {n_sp} of {len(pairs)} pairs; our site separates {n_ss} of {len(pairs)}; the order "
             f"flips in {(pairs['order flips'] == 'yes').sum()} pairs.**", "", pairs.to_markdown(index=False), "",
             "## Agreement", "",
             f"- Kendall's tau-b between the two orders: {tau:+.2f} [{np.nanpercentile(taus, 2.5):+.2f}, {np.nanpercentile(taus, 97.5):+.2f}]"
             + (f" (undefined in {np.isnan(taus).mean():.1%} of resamples)" if np.isnan(taus).any() else ""),
             f"- tau-b with ties for pairs a side cannot tell apart: {fmt_tau_ties(tau_b_ties(P.values, O.values, n=B), len(pairs))}",
             f"- Best on SICAPv2: {P.mean().idxmax()}; best on our site: {O.mean().idxmax()}", ""]
    (CV / "ranking47.md").write_text("\n".join(lines), encoding="utf-8")
    pd.concat({"sicap_heldout": P.mean(), "site47": O.mean()}, axis=1).to_csv(CV / "ranking47_scores.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
