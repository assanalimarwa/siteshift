"""
Threshold plan on saved per-pixel probabilities (protocol_probs.py): how much Dice is left on the table by the fixed 0.5 threshold.

    python threshold_sweep.py --name site_unet --folders probs/site_unet/fold0 probs/site_unet/fold1 ... [--out paper_figs/protocol]

For every slide the probability histogram of tumour and of non-tumour pixels (256 bins of the uint8 probability) is read once (cached in
<out>/hist_<name>.csv); every threshold t then means prob >= ceil(255 t) (quantisation step 1/255). Dice per patient = mean of the patient's slide Dice
(the paper's headline; slides with no tumour are left out of the mean only if the patient has none).
  oracle tau*       the threshold on a 0.01 grid (0.05 .. 0.95) that maximises the mean per-patient Dice over all patients of the set
                    (an upper bound: tau* is chosen on the test patients themselves)
  site-calibrated   tau chosen on CAL = 2 randomly drawn patients (max of their mean Dice), scored on the other patients; 500 draws, seed 0:
                    mean and 2.5 / 97.5 percentiles of the held-out Dice, of tau, and of the paired gain over tau = 0.5
Writes <out>/threshold_<name>.md and <out>/threshold_<name>_curve.csv (mean per-patient Dice at every t).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
CAL, DRAWS = 2, 500


def slide_hist(path):
    z = np.load(path)
    prob, size = z["prob"], int(z["size"])
    truth = np.unpackbits(z["truth"], axis=1, count=size * size).reshape(prob.shape).view(bool)
    return str(z["slide"]), np.bincount(prob[truth], minlength=256), np.bincount(prob[~truth], minlength=256)


def histograms(folders, cache):
    if cache.exists():
        h = pd.read_csv(cache)
        have = set(h.slide)
    else:
        h, have = pd.DataFrame(), set()
    rows = []
    for f in folders:
        for p in sorted(Path(f).glob("*.npz")):
            s, ht, hf = slide_hist(p)
            if s not in have:
                rows.append({"slide": s, **{f"t{i}": v for i, v in enumerate(ht)}, **{f"f{i}": v for i, v in enumerate(hf)}})
                have.add(s)
    if rows:
        h = pd.concat([h, pd.DataFrame(rows)])
        h.to_csv(cache, index=False)
    return h


def curve(h):
    """per-patient Dice (mean of slide Dice) at every grid threshold: DataFrame patients x thresholds."""
    ht = h[[f"t{i}" for i in range(256)]].to_numpy(float)
    hf = h[[f"f{i}" for i in range(256)]].to_numpy(float)
    ct, cf = ht[:, ::-1].cumsum(1)[:, ::-1], hf[:, ::-1].cumsum(1)[:, ::-1]        # counts of pixels with prob >= i
    tot = ht.sum(1)
    out = {}
    for t in GRID:
        k = int(np.ceil(255 * t - 1e-9))
        tp, fp = ct[:, k], cf[:, k]
        out[t] = 2 * tp / np.maximum(2 * tp + fp + (tot - tp), 1)
    d = pd.DataFrame(out)
    d["pt"] = h.slide.str.split(".").str[0].astype(int).values
    d["has_tumour"] = tot > 0
    d = d[d.has_tumour].drop(columns="has_tumour")
    return d.groupby("pt").mean()


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("paper_figs/protocol"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    h = histograms(a.folders, a.out / f"hist_{a.name}.csv")
    c = curve(h)
    c.mean().rename("mean per-patient Dice").to_csv(a.out / f"threshold_{a.name}_curve.csv", header=True)
    mean = c.mean()
    t_star = mean.idxmax()
    lines = [f"# Threshold sweep: {a.name} ({len(c)} patients, tile mode, probabilities quantised to uint8: step 1/255)", ""]
    d05, dstar = c[0.5], c[t_star]
    lines += ["| threshold | mean per-patient Dice [95% CI] |", "|---|---|",
              f"| 0.50 (fixed) | {d05.mean():.3f} [{boot(d05)[0]:.3f}, {boot(d05)[1]:.3f}] |",
              f"| oracle tau* = {t_star:.2f} | {dstar.mean():.3f} [{boot(dstar)[0]:.3f}, {boot(dstar)[1]:.3f}] |", "",
              f"Oracle gain over 0.50: {dstar.mean() - d05.mean():+.3f} (tau* is chosen on the test patients themselves, an upper bound).", ""]
    if len(c) > CAL + 2:
        rng = np.random.default_rng(0)
        pts = c.index.to_numpy()
        res = []
        for _ in range(DRAWS):
            cal = rng.choice(pts, CAL, replace=False)
            tau = c.loc[cal].mean().idxmax()
            rest = np.setdiff1d(pts, cal)
            res.append((tau, c.loc[rest, tau].mean(), c.loc[rest, 0.5].mean()))
        r = pd.DataFrame(res, columns=["tau", "dice_cal", "dice_05"])
        gain = r.dice_cal - r.dice_05
        lines += [f"Site-calibrated (tau chosen on {CAL} random patients, scored on the other {len(c) - CAL}; {DRAWS} draws, seed 0):", "",
                  "| | mean | 2.5% | 97.5% |", "|---|---|---|---|",
                  f"| tau | {r.tau.mean():.2f} | {r.tau.quantile(.025):.2f} | {r.tau.quantile(.975):.2f} |",
                  f"| held-out Dice at the calibrated tau | {r.dice_cal.mean():.3f} | {r.dice_cal.quantile(.025):.3f} | {r.dice_cal.quantile(.975):.3f} |",
                  f"| held-out Dice at 0.50 | {r.dice_05.mean():.3f} | {r.dice_05.quantile(.025):.3f} | {r.dice_05.quantile(.975):.3f} |",
                  f"| gain (calibrated - 0.50), paired per draw | {gain.mean():+.3f} | {gain.quantile(.025):+.3f} | {gain.quantile(.975):+.3f} |", ""]
    (a.out / f"threshold_{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
