"""
Domain distance between each dataset and ours in UNI feature space (CPU, a few minutes):

    python ds_distance.py      ->  paper_figs/datasets/data/distance.csv, distance_transfer.csv, fig_domain_distance.pdf

Features: UNI class tokens of up to 2,000 random tissue tiles per dataset (~256 um fields; paper_figs/data/uni_embeddings.npz
from embed_uni.py, plus datasets/data/uni_extra.npz for datasets embedded later, e.g. RINGS). Ours = 2,000 biopsy +
2,000 large-section tiles (balanced by specimen type; random tiles would be ~92% large sections); also reported against
ours-biopsies only, since every public set is biopsies. Noise floor: ours vs ours (two random halves of each type).
  Frechet distance (FD): ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^1/2) on the raw features (as FID).
  MMD^2: unbiased, Gaussian kernel on L2-normalised features, bandwidth = median pairwise distance of ours (fixed for
  all comparisons, so the numbers are comparable).
Uncertainty: 200 random subsamples WITHOUT replacement of M = 1,000 tiles from each set (seed 0); reported as the mean
over subsamples [2.5th, 97.5th percentile]. Resampling with replacement was tried first and dropped: duplicated tiles
inflate both FD and MMD (ours vs ours: FD 92.5 on the full halves, bootstrap interval [164, 174]). FD depends on the
sample size, so all comparisons use the same M; the full-set values are listed as FD_full / MMD2_full.
Noise floor = ours vs ours (two random halves, balanced by specimen type). FD_ratio = FD / floor FD, interval from the
ratios of paired independent draws. MMD^2 is unbiased, so its floor is ~0 and a ratio is meaningless; instead
MMD2_above_floor says whether the comparison's interval lies entirely above the floor's.
Transfer Dice (distance_transfer.csv, the plot): mean per-patient Dice on our 47 patients of the U-Net trained only on
that dataset, tile mode, from cv47/public_by_patient.csv (report47.py public); datasets without a single-source model
are listed without Dice. The plot is descriptive (one point per dataset, no fitted line or correlation: user 2026-11-04).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import numpy as np
import pandas as pd
from scipy import linalg

from common import COL_W, FIGS, OI, U, plt, style
from ds_lib import DATA, OUT

B, SEED, M = 200, 0, 1000
NAMES = {"SICAPv2": "SICAPv2", "PANDA-Karolinska": "PANDA-Karolinska", "PANDA-Radboud": "PANDA-Radboud"}
SINGLE_SOURCE = {"SICAPv2": "SICAPv2 U-Net", "PANDA-Karolinska": "Karolinska 3,814 U-Net",   # report47 PUBLIC_LABELS
                 "PANDA-Radboud": "Radboud U-Net"}


def load():
    z = np.load(DATA / "uni_embeddings.npz", allow_pickle=True)
    feats, groups = [z["feat"].astype(np.float64)], [z["group"]]
    extra = DATA / "uni_extra.npz"
    if extra.exists():
        e = np.load(extra, allow_pickle=True)
        feats.append(e["feat"].astype(np.float64)), groups.append(e["group"])
    return np.concatenate(feats), np.concatenate(groups)


def fd(a, b):
    mu = a.mean(0) - b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2, disp=False)[0].real
    return float(mu @ mu + np.trace(s1 + s2 - 2 * covmean))


def mmd2(a, b, sigma):
    def k(x, y):
        d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T
        return np.exp(-np.clip(d, 0, None) / (2 * sigma ** 2))
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    n, m = len(a), len(b)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())


def main():
    import sys
    if "--plot-only" in sys.argv:                                  # reuse distance.csv (45 min to compute)
        return plot(pd.read_csv(DATA / "distance.csv"))
    f, g = load()
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    ours = np.isin(g, ["Ours, biopsies", "Ours, large sections"])
    bio = g == "Ours, biopsies"
    rng = np.random.default_rng(SEED)
    sub = fn[ours][rng.choice(ours.sum(), 2000, replace=False)]
    d = np.sqrt(np.clip((sub * sub).sum(1)[:, None] + (sub * sub).sum(1)[None] - 2 * sub @ sub.T, 0, None))
    sigma = float(np.median(d[np.triu_indices(len(sub), 1)]))
    # ours vs ours: random halves within each specimen type
    halves = np.zeros(len(g), bool)
    for grp in ["Ours, biopsies", "Ours, large sections"]:
        idx = np.nonzero(g == grp)[0]
        halves[rng.choice(idx, len(idx) // 2, replace=False)] = True
    pairs = [("ours (half) vs ours (half)", ours & halves, ours & ~halves), ("ours biopsies vs ours large sections", bio, g == "Ours, large sections")]
    others = [x for x in dict.fromkeys(g) if not x.startswith("Ours")]
    for x in others:
        pairs += [(f"{x} vs ours", g == x, ours), (f"{x} vs ours biopsies", g == x, bio)]
    rows, draws = [], {}
    for name, ia, ib in pairs:
        a, b, an, bn = f[ia], f[ib], fn[ia], fn[ib]
        r = dict(comparison=name, n_a=len(a), n_b=len(b), FD_full=fd(a, b), MMD2_full=mmd2(an, bn, sigma))
        bf, bm = [], []
        for _ in range(B):
            i, j = rng.choice(len(a), M, replace=False), rng.choice(len(b), M, replace=False)
            bf.append(fd(a[i], b[j])), bm.append(mmd2(an[i], bn[j], sigma))
        r.update(FD=np.mean(bf), FD_lo=np.percentile(bf, 2.5), FD_hi=np.percentile(bf, 97.5), MMD2=np.mean(bm),
                 MMD2_lo=np.percentile(bm, 2.5), MMD2_hi=np.percentile(bm, 97.5))
        draws[name] = np.array(bf)
        if name != pairs[0][0]:                                   # FD as a multiple of our own noise floor: draw k of this
            ratio = draws[name] / draws[pairs[0][0]]               # comparison over draw k of ours-vs-ours (independent)
            r.update(FD_ratio=np.mean(draws[name]) / np.mean(draws[pairs[0][0]]), FD_ratio_lo=np.percentile(ratio, 2.5),
                     FD_ratio_hi=np.percentile(ratio, 97.5), MMD2_above_floor=r["MMD2_lo"] > rows[0]["MMD2_hi"])
        rows.append(r)
        print(f"{name:45s} FD {r['FD']:8.1f} [{r['FD_lo']:.1f}, {r['FD_hi']:.1f}]  MMD2 {r['MMD2']:.4f} "
              f"[{r['MMD2_lo']:.4f}, {r['MMD2_hi']:.4f}]", flush=True)
    dist = pd.DataFrame(rows).assign(sigma=sigma)
    dist.to_csv(DATA / "distance.csv", index=False)
    plot(dist)


def plot(dist):
    others = [c.replace(" vs ours", "") for c in dist.comparison if c.endswith(" vs ours")]
    # transfer Dice of single-source models on our 47 patients
    pub = P.RESULTS / "cv47" / "public_by_patient.csv"
    tr = []
    for x in others:
        r = dist[dist.comparison == f"{x} vs ours"].iloc[0].to_dict()
        r["dataset"] = x
        if pub.exists() and x in SINGLE_SOURCE:
            p = pd.read_csv(pub)
            p = p[(p.model == SINGLE_SOURCE[x]) & (p["mode"] == "tile")]
            if len(p):
                r.update(transfer_model=SINGLE_SOURCE[x], transfer_dice=p.dice.mean(), transfer_clean_auc=p.clean_auc.mean(),
                         patients=len(p))
        tr.append(r)
    tr = pd.DataFrame(tr)
    tr.to_csv(DATA / "distance_transfer.csv", index=False)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    col = {"SICAPv2": OI["orange"], "PANDA-Karolinska": OI["green"], "PANDA-Radboud": OI["purple"]}
    d = dist.set_index("comparison")
    fl, bl = d.loc["ours (half) vs ours (half)"], d.loc["ours biopsies vs ours large sections"]
    # descriptive (user 2026-11-04): one point per dataset, no fitted line; FD as a multiple of our own noise floor
    for ax, m, lab in [(axes[0], "ratio", "Frechet distance / noise floor"), (axes[1], "MMD2", "MMD$^2$ to ours")]:
        ref = [(1.0 if m == "ratio" else fl.MMD2, "ours vs ours"), (bl.FD / fl.FD if m == "ratio" else bl.MMD2, "our biopsies vs\nour large sections")]
        for x, t in ref:
            ax.axvline(x, color=OI["grey"], lw=0.6, ls=":")
            ax.text(x, 0.88, t, rotation=90, fontsize=5.5, color=OI["grey"], ha="right", va="top")
        for _, r in tr.iterrows():
            x, lo, hi = ((r.FD_ratio, r.FD_ratio_lo, r.FD_ratio_hi) if m == "ratio" else (r.MMD2, r.MMD2_lo, r.MMD2_hi))
            if pd.notna(r.get("transfer_dice", np.nan)):
                ax.errorbar(x, r.transfer_dice, xerr=[[x - lo], [hi - x]], fmt="o", ms=3.5, color=col[r.dataset], lw=0.7, capsize=1.5)
                up = r.dataset == "SICAPv2"                     # labels above / below their points (close in y)
                ax.annotate(r.dataset.replace("PANDA-", ""), (x, r.transfer_dice), fontsize=6, xytext=(0, 6 if up else -9),
                            textcoords="offset points", ha="center", va="bottom" if up else "top")
            else:                                                   # no single-source model scored yet: tick at the bottom
                ax.plot([x], [0.1], marker="^", ms=3.5, color=col[r.dataset], clip_on=False)
                ax.annotate(r.dataset.replace("PANDA-", "") + "\n(model pending)", (x, 0.1), fontsize=5.5, xytext=(0, 5),
                            textcoords="offset points", ha="center", va="bottom", color=col[r.dataset])
        ax.set_xlabel(lab)
        ax.set_xlim(0, 6.2 if m == "ratio" else 0.125)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.1, 0.9)
    axes[0].set_ylabel("Transfer Dice on ours (tile)")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_domain_distance.pdf")
    fig.savefig(DATA / "preview_fig_domain_distance.png", dpi=200)
    print(tr.to_string())


if __name__ == "__main__":
    main()
