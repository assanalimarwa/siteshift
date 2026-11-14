"""
Is a public test set enough to benchmark models for a new site? Models trained on the same public
data (SICAPv2 only) are scored on SICAPv2's held-out tiles and on our patients, and the two
benchmarks are compared.

    python ranking.py                      # cv/ranking.md, cv/ranking_scores.csv

Models: every run trained on SICAPv2 only that has both scores (run_cv.ZERO_SHOT: the U-Net and
the frozen-encoder heads). Each tile is predicted alone, threshold 0.5 (tile mode).
Public score: mean Dice over SICAPv2 held-out regions that contain cancer (val + guard, see
prepare_sicap.py; regions = groups of overlapping tiles). Our score: mean Dice over our 17 patients.

Three questions, all with 95% bootstrap intervals (2,000 resamples of regions and of patients):
  separation   for every pair of models, the paired Dice difference on each benchmark: how many
               pairs does each benchmark tell apart (interval excluding 0)?
  agreement    Kendall's tau-b between the model rankings on the two benchmarks; also with ties for
               the pairs a benchmark cannot tell apart (cv_summary.tau_b_ties)
  regret       Dice on our patients lost by choosing the model that is best on SICAPv2 instead of
               the one that is best on our patients
Also per specimen type (needle biopsies, large sections) for our side.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from cv_summary import fmt_tau_ties, per_patient, rates, sources, tau_b_ties

HERE = P.INDEX
NAMES = {"sicap_only": "U-Net (ResNet-34)", "sicap_fm_uni": "UNI", "sicap_fm_virchow2": "Virchow2", "sicap_fm_conch": "CONCH",
         "sicap_fm_dinov2": "DINOv2-L", "sicap_fm_dinov3": "DINOv3-L", "sicap_fm_unimedclip": "UniMed-CLIP"}
B, SEED = 2000, 0


def dice(c):
    return 2 * c.tp / (2 * c.tp + c.fp + c.fn).clip(lower=1)


def public_scores(run):
    """Dice per held-out SICAPv2 region with cancer (val + guard), tile mode."""
    parts = [pd.read_csv(P.RUNS / run / f"eval_sicap_{s}_tile" / "per_tile.csv") for s in ["val", "guard"]]
    if not all(len(p) for p in parts):
        return None
    t = pd.concat(parts)
    g = t.groupby("slide")[["tp", "fp", "fn", "tumour_px"]].sum()
    return dice(g[g.tumour_px > 0])


def main():
    from run_cv import ZERO_SHOT
    src = sources("tile")
    bio = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    pub, ours = {}, {}
    for key, run in ZERO_SHOT.items():
        if key not in NAMES or key not in src or not (P.RUNS / run / "eval_sicap_val_tile" / "per_tile.csv").exists():
            continue
        pub[key] = public_scores(run)
        c = per_patient(pd.concat([pd.read_csv(p) for p in src[key]]))
        ours[key] = rates(c).dice
    models = sorted(pub, key=lambda k: list(NAMES).index(k))
    if len(models) < 3:
        raise SystemExit(f"need at least 3 models with both scores, have {models}")
    P = pd.DataFrame({m: pub[m] for m in models}).dropna()            # regions x models
    O = pd.DataFrame({m: ours[m] for m in models}).dropna()           # patients x models
    rng = np.random.default_rng(SEED)
    ri = rng.integers(0, len(P), (B, len(P)))
    pi = rng.integers(0, len(O), (B, len(O)))
    Pb = P.values[ri].mean(1)                                          # B x models
    Ob = O.values[pi].mean(1)
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))

    lines = ["# Public benchmark vs our patients", "",
             f"{len(models)} models trained on SICAPv2 only; SICAPv2 held-out: {len(P)} regions with cancer; ours: {len(O)} patients "
             f"({sum(p in bio for p in O.index)} needle biopsies, {sum(p not in bio for p in O.index)} large sections). "
             "Mean Dice [95% CI], tile mode, threshold 0.5.", ""]
    score = pd.DataFrame({
        "model": [NAMES[m] for m in models],
        "SICAPv2 held-out": [f"{P[m].mean():.3f} [{ci(Pb[:, i])[0]:.3f}, {ci(Pb[:, i])[1]:.3f}]" for i, m in enumerate(models)],
        "rank on SICAPv2": pd.Series(P.mean().values).rank(ascending=False).astype(int).values,
        "our patients": [f"{O[m].mean():.3f} [{ci(Ob[:, i])[0]:.3f}, {ci(Ob[:, i])[1]:.3f}]" for i, m in enumerate(models)],
        "rank on ours": pd.Series(O.mean().values).rank(ascending=False).astype(int).values,
        "our biopsies": [round(O.loc[[p for p in O.index if p in bio], m].mean(), 3) for m in models],
        "our large sections": [round(O.loc[[p for p in O.index if p not in bio], m].mean(), 3) for m in models],
        "drop SICAPv2 -> ours": [f"{P[m].mean() - O[m].mean():+.3f} [{ci(Pb[:, i] - Ob[:, i])[0]:+.3f}, {ci(Pb[:, i] - Ob[:, i])[1]:+.3f}]"
                                  for i, m in enumerate(models)]})
    lines += ["## Scores", "", score.to_markdown(index=False), ""]

    pairs = []
    for i, j in itertools.combinations(range(len(models)), 2):
        dp, do = Pb[:, i] - Pb[:, j], Ob[:, i] - Ob[:, j]
        sep = lambda d: "yes" if ci(d)[0] > 0 or ci(d)[1] < 0 else "no"
        flip = np.sign(P[models[i]].mean() - P[models[j]].mean()) != np.sign(O[models[i]].mean() - O[models[j]].mean())
        pairs.append({"pair": f"{NAMES[models[i]]} - {NAMES[models[j]]}",
                      "SICAPv2 difference": f"{dp.mean():+.3f} [{ci(dp)[0]:+.3f}, {ci(dp)[1]:+.3f}]", "SICAPv2 separates": sep(dp),
                      "ours difference": f"{do.mean():+.3f} [{ci(do)[0]:+.3f}, {ci(do)[1]:+.3f}]", "ours separates": sep(do),
                      "order flips": "yes" if flip else "no"})
    pairs = pd.DataFrame(pairs)
    lines += ["## Separation: paired differences between models", "",
              f"SICAPv2 tells apart {(pairs['SICAPv2 separates'] == 'yes').sum()} of {len(pairs)} pairs; our patients tell apart "
              f"{(pairs['ours separates'] == 'yes').sum()}; the order flips in {(pairs['order flips'] == 'yes').sum()} pairs.", "",
              pairs.to_markdown(index=False), ""]

    taus = np.array([kendalltau(Pb[b], Ob[b]).statistic for b in range(B)])
    tau = kendalltau(P.mean().values, O.mean().values).statistic
    best_pub, best_ours = int(np.argmax(P.mean().values)), int(np.argmax(O.mean().values))
    regret = Ob[:, best_ours] - Ob[:, best_pub]
    lines += ["## Agreement and regret", "",
              f"- Kendall's tau-b between the two rankings: {tau:+.2f} [{np.nanpercentile(taus, 2.5):+.2f}, {np.nanpercentile(taus, 97.5):+.2f}]",
              f"- Kendall's tau-b with ties for pairs a benchmark cannot tell apart: {fmt_tau_ties(tau_b_ties(P.values, O.values), len(pairs))}",
              f"- Best on SICAPv2: {NAMES[models[best_pub]]}; best on our patients: {NAMES[models[best_ours]]}",
              f"- Regret (Dice lost on our patients by trusting SICAPv2): {regret.mean():.3f} [{ci(regret)[0]:.3f}, {ci(regret)[1]:.3f}]", ""]

    (HERE / "cv" / "ranking.md").write_text("\n".join(lines), encoding="utf-8")
    pd.concat({"sicap": P.mean(), "ours": O.mean()}, axis=1).rename(index=NAMES).to_csv(HERE / "cv" / "ranking_scores.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
