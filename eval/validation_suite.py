"""
Step 2 of the protocol validation: the perturbation suite on 10 REAL annotation masks (5 biopsies, 5 large sections; fixed seed 0), CPU only.
The metric engine is the reference attribution.py (identical to protocol.py on real slides, see parity.md).

Each mask is a window of up to 4096 x 4096 px (2.05 mm) of the stitched annotation of one slide, positioned (coarse search) on the most regions.
Perturbed predictions P of the annotation G: delete 10 / 30 / 50% of the regions (>= 2,500 um^2); add 5 false blobs (60 um radius) farther than 50 um from
every tumour pixel; erode / dilate by 20 / 50 / 100 um; gland-style = opening at 20 / 40 um; region-style = closing at 40 / 80 um; equal-Dice pairs: deletion
(10 / 30 / 50%) against erosion tuned by bisection to the same Dice (within 0.005).

    python validation_suite.py            # writes protocol_validation.md, validation_rows.csv, validation_pairs.csv, fig_validation.pdf / .png
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import attribution as ref  # noqa: E402
from data import grid_position  # noqa: E402

OUT = P.OUT
EXP = Path(str(P.PRED / "nnunet" / "predictions"))
CODE = P.RESULTS
UM, WIN, MIN_PX = ref.UM, 4096, int(ref.MIN_REGION_UM2 / ref.UM ** 2)
BINS = ["miss", "false", "fill", "bnd"]
PERT = ["delete 10%", "delete 30%", "delete 50%", "false blobs", "erode 20 um", "erode 50 um", "erode 100 um", "dilate 20 um", "dilate 50 um",
        "dilate 100 um", "opening 20 um", "opening 40 um", "closing 40 um", "closing 80 um"]
# bins a perturbation is MEANT to move; anything else is leakage (expected small: erased tiny regions -> miss, gaps wider than 100 um -> false, ...)
INTENDED = {"delete 10%": ["miss"], "delete 30%": ["miss"], "delete 50%": ["miss"], "false blobs": ["false"], "erode 20 um": ["bnd"], "erode 50 um": ["bnd"],
            "erode 100 um": ["bnd", "fill"], "dilate 20 um": ["bnd"], "dilate 50 um": ["bnd"], "dilate 100 um": ["bnd", "false"], "opening 20 um": ["bnd"],
            "opening 40 um": ["bnd"], "closing 40 um": ["bnd"], "closing 80 um": ["bnd", "false"]}


def stitch_truth(path):
    z = np.load(path)
    S, images = int(z["size"]), z["image"]
    pos = np.array([grid_position(p) for p in images])
    c0, r0 = pos.min(0)
    G = np.zeros(((pos[:, 1].max() - r0 + 1) * S, (pos[:, 0].max() - c0 + 1) * S), bool)
    for i, (c, r) in enumerate(pos):
        G[(r - r0) * S:(r - r0 + 1) * S, (c - c0) * S:(c - c0 + 1) * S] = np.unpackbits(z["truth"][i], count=S * S).reshape(S, S).view(bool)
    return G


def pick_slides():
    meta = pd.read_csv(CODE / "cv47" / "baseline_imagenet_by_patient.csv").set_index("pt")
    t = pd.read_csv(CODE / "tiles47.csv", usecols=["slide", "tumour_frac"])
    s = t.groupby("slide").agg(tiles=("tumour_frac", "size"), tum=("tumour_frac", "mean"))
    s["pt"] = s.index.str.split(".").str[0].astype(int)
    s["type"] = s.pt.map(meta.type)
    s = s[(s.tum > 0.02)]
    reg = pd.concat([pd.read_csv(f) for f in sorted(EXP.glob("fold*/deprecated_square_band_protocol_slides.csv"))]).set_index("slide").regions   # region counts do not depend on the band
    s = s[s.index.map(reg) >= 8]                                     # enough regions for 10 / 30 / 50% deletions to differ
    rng = np.random.default_rng(0)
    out = []
    for typ, lo in [("biopsy", 60), ("large", 600)]:
        c = s[(s.type == typ) & (s.tiles >= lo)]
        c = c.groupby("pt").head(1)                                  # one slide per patient
        out += [(sl, typ) for sl in c.index[rng.permutation(len(c))[:5]]]
    return out


def window(G):
    """Coarse search (1/8 scale) for the 4096-px window holding the most regions >= 2,500 um^2."""
    if min(G.shape) <= WIN:
        return G
    g8 = G[::8, ::8]
    lab, n = ndi.label(g8, ref.S8)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * 64
    big = np.nonzero(area >= MIN_PX)[0] + 1
    cm = np.array(ndi.center_of_mass(g8, lab, big)) * 8 if len(big) else np.zeros((0, 2))
    best, arg = (-1, -1), (0, 0)
    for y0 in range(0, G.shape[0] - WIN + 1, 1024):
        for x0 in range(0, G.shape[1] - WIN + 1, 1024):
            m = (cm[:, 0] >= y0) & (cm[:, 0] < y0 + WIN) & (cm[:, 1] >= x0) & (cm[:, 1] < x0 + WIN)
            k = (int(m.sum()), int(G[y0:y0 + WIN:8, x0:x0 + WIN:8].sum()))
            if k > best:
                best, arg = k, (y0, x0)
    return G[arg[0]:arg[0] + WIN, arg[1]:arg[1] + WIN].copy()


def regions(G):
    lab, n = ndi.label(G, ref.S8)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, [i for i in range(1, n + 1) if area[i] >= MIN_PX]


def delete(G, lab, big, order, frac):
    k = max(1, int(round(frac * len(big))))
    P = G.copy()
    for i in order[:k]:
        P[lab == big[i]] = False
    return P, k


def far_blobs(G, rng, n=5, r_um=60):
    r = int(r_um / UM)
    d = ndi.distance_transform_edt(~G) * UM
    ok = d >= ref.BAND_UM + r_um + 10
    ys, xs = np.nonzero(ok[r:-r:16, r:-r:16])
    P, put = G.copy(), 0
    disk = ref.disk(r)
    for j in rng.permutation(len(ys)):
        y, x = ys[j] * 16 + r, xs[j] * 16 + r
        if P[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1].any() and (P & ~G)[y - r:y + r + 1, x - r:x + r + 1].any():
            continue                                                  # do not overlap an earlier blob
        P[y - r:y + r + 1, x - r:x + r + 1] |= disk
        put += 1
        if put == n:
            break
    return P, put


def opening(X, r):
    return ref.dilate_um(ref.erode_um(X, r), r)


def score(G, P):
    a = ref.attribution(G, P)
    return dict(dice=a["dice"], **a["s"], exact=a["exact"], n_det=a["n_det"], n_reg=a["n_reg"], denom=a["denom"], bias_um=ref.signed_bias_um(G, P))


def one_mask(args):
    idx, slide, typ = args
    t0 = time.time()
    path = next(EXP.glob(f"fold*/{re.sub(r'[^\w.-]+', '_', slide)}.npz"))
    G = window(stitch_truth(path))
    lab, big = regions(G)
    rng = np.random.default_rng(1000 + idx)
    order = list(rng.permutation(len(big)))
    rows, pairs = [], []
    base = score(G, G)
    rows.append(dict(mask=idx, slide=slide, type=typ, pert="identical", n_big=len(big), **base))
    perts = {}
    for f in (0.1, 0.3, 0.5):
        perts[f"delete {int(f * 100)}%"] = delete(G, lab, big, order, f)[0]
    perts["false blobs"] = far_blobs(G, rng)[0]
    for r in (20, 50, 100):
        perts[f"erode {r} um"] = ref.erode_um(G, r)
        perts[f"dilate {r} um"] = ref.dilate_um(G, r)
    perts["opening 20 um"], perts["opening 40 um"] = opening(G, 20), opening(G, 40)
    perts["closing 40 um"], perts["closing 80 um"] = ref.close_um(G, 40), ref.close_um(G, 80)
    for name in PERT:
        rows.append(dict(mask=idx, slide=slide, type=typ, pert=name, n_big=len(big), **score(G, perts[name])))
    for f in (0.1, 0.3, 0.5):                                           # equal-Dice pairs: deletion A against erosion B tuned to the same Dice
        A = perts[f"delete {int(f * 100)}%"]
        sa = score(G, A)
        lo, hi = 0.0, 300.0
        if score(G, ref.erode_um(G, hi))["dice"] > sa["dice"]:
            pairs.append(dict(mask=idx, slide=slide, type=typ, f=f, reachable=False))
            continue
        for _ in range(14):
            mid = (lo + hi) / 2
            if score(G, ref.erode_um(G, mid))["dice"] > sa["dice"]:
                lo = mid
            else:
                hi = mid
            sb = score(G, ref.erode_um(G, mid))
            if abs(sb["dice"] - sa["dice"]) <= 0.002:
                break
        sb = score(G, ref.erode_um(G, mid))
        pairs.append(dict(mask=idx, slide=slide, type=typ, f=f, reachable=True, erode_um=mid, dice_A=sa["dice"], dice_B=sb["dice"],
                          **{f"A_{b}": sa[b] for b in BINS}, **{f"B_{b}": sb[b] for b in BINS}, exact=sa["exact"] and sb["exact"]))
    print(f"mask {idx} ({typ}, {slide}): {len(big)} regions, {G.shape}, {time.time() - t0:.0f} s", flush=True)
    return rows, pairs


def report(df, pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"miss": "#2a78d6", "false": "#eb6834", "fill": "#1baf7a", "bnd": "#eda100"}
    NAME = {"miss": "missed regions", "false": "false tumour > 50 um", "fill": "unmarked interior", "bnd": "boundary band"}
    perts = df[df.pert != "identical"]
    m = perts.groupby("pert")[["dice"] + BINS].mean().loc[PERT]
    lines = ["# Protocol validation, step 2: perturbation suite on real annotation masks", "",
             f"{df["mask"].nunique()} real annotation masks ({(df.groupby('mask').type.first() == 'biopsy').sum()} biopsies, {(df.groupby('mask').type.first() == 'large').sum()} large sections), "
             "fixed seed 0, windows of up to 4096 x 4096 px (2.05 mm) of the stitched annotation at 0.5 um/px; metric engine: the reference attribution.py "
             "(equal to protocol.py, parity.md). Bins are shares of 2 TP + FP + FN, so they add up to 1 - Dice.", ""]
    mk = df[df.pert == "identical"][["mask", "slide", "type", "n_big"]]
    lines += ["## Mask selection (rule and seed)", "",
              "Slides of the 47-patient set with at least 8 countable regions (>= 2,500 um^2) on the whole slide (region counts of the nnU-Net-style site-CV protocol analysis; "
              "they do not depend on the band), a mean tumour tile fraction > 0.02, at least 60 tiles (biopsies) or 600 tiles (large sections), one slide per patient; "
              "5 biopsies and 5 large sections drawn with numpy default_rng(0) (pick_slides in validation_suite.py). Each mask is the 4096 x 4096 px window (2.05 mm) of the stitched "
              "annotation holding the most regions (coarse search at 1/8 scale). **Limit:** the annotated regions are large and sparse, so a window holds only "
              + f"{mk.n_big.min()} to {mk.n_big.max()} of the slide's regions (even 8192-px windows hold 3 to 6 in the candidate slides). "
              "'Delete 10 / 30 / 50% of the regions' therefore removes max(1, round(f n)) regions of the n in the window; the achieved fractions are in the table below and the "
              "deletion rows are best read by their achieved area fraction.", "",
              "| mask | slide | type | regions in the window |", "|---|---|---|---|"]
    lines += [f"| {r.mask} | {r.slide.split('.bif')[0]} | {r.type} | {r.n_big} |" for r in mk.itertuples()]
    dl = df[df.pert.str.startswith("delete")].copy()
    dl["f"] = dl.pert.str.extract(r"(\d+)")[0].astype(int) / 100
    dl["k"] = [max(1, int(round(f * n))) for f, n in zip(dl.f, dl.n_big)]
    dl["area_deleted"] = 2 * dl.miss / (1 + dl.miss)                      # FN / (TP + FN) when only regions are deleted
    lines += ["", "Achieved deletions (mean over the masks):", "", "| requested | regions deleted | share of the regions | share of the tumour area |", "|---|---|---|---|"]
    for f, g in dl.groupby("f"):
        lines.append(f"| {int(f * 100)}% | {g.k.mean():.1f} of {g.n_big.mean():.1f} | {(g.k / g.n_big).mean():.0%} | {g.area_deleted.mean():.0%} |")
    lines += [""]
    ident = df[df.pert == "identical"]
    lines += [f"**Identity.** 1 - Dice = miss + false + fill + boundary holds exactly (integer pixel counts) in all {len(df)} scored predictions "
              f"({'YES' if df.exact.all() else 'NO: ' + str((~df.exact).sum()) + ' violations'}); the identical prediction gives Dice {ident.dice.min():.3f} and all bins "
              f"{ident[BINS].abs().max().max():.3f}.", "",
              "## Which bins each perturbation moves (mean over the masks; shares of the error)", "",
              "| perturbation | Dice | missed | false tumour | interior | boundary | share in the intended bin(s) | leakage |", "|---|---|---|---|---|---|---|---|"]
    leak = {}
    for p in PERT:
        r = m.loc[p]
        tot = r[BINS].sum()
        inten = r[INTENDED[p]].sum()
        leak[p] = 1 - inten / tot if tot > 0 else np.nan
        lines.append(f"| {p} | {r.dice:.3f} | {r.miss:.3f} | {r.false:.3f} | {r.fill:.3f} | {r.bnd:.3f} | {', '.join(INTENDED[p])}: {inten / tot:.0%} | {leak[p]:.1%} |")
    lines += ["", "Expected leakage: tiny regions erased by erosion or opening (-> missed), gaps wider than 100 um filled by closing 80 um or dilation 100 um (-> false tumour), "
              "deleted regions are replaced one for one (no leakage into other bins except through sub-2,500 um^2 pieces cut by the window border).", "",
              "### by specimen type (mean Dice and share in the intended bins)", "", "| perturbation | biopsy Dice | biopsy intended | large Dice | large intended |", "|---|---|---|---|---|"]
    for p in PERT:
        cells = []
        for typ in ("biopsy", "large"):
            q = perts[(perts.pert == p) & (perts.type == typ)][["dice"] + BINS].mean()
            cells += [f"{q.dice:.3f}", f"{q[INTENDED[p]].sum() / q[BINS].sum():.0%}"]
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    ok = pr[pr.reachable == True]  # noqa: E712
    lines += ["", "## Equal-Dice pairs: deletion of regions (A) against erosion tuned to the same Dice (B)", "",
              f"Erosion radius found by bisection so that |Dice(B) - Dice(A)| <= 0.002 (the pairs below all satisfy the requested 0.005: "
              f"{'YES' if (ok.dice_A - ok.dice_B).abs().max() <= 0.005 else 'NO'}); {len(ok)} of {len(pr)} pairs reachable.", "",
              "| deleted | pairs | Dice A | Dice B | A: detection (miss + false) | A: delineation | B: detection | B: delineation | erosion radius (um) | separated |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f, g in ok.groupby("f"):
        ad, adl = (g.A_miss + g.A_false).mean(), (g.A_fill + g.A_bnd).mean()
        bd, bdl = (g.B_miss + g.B_false).mean(), (g.B_fill + g.B_bnd).mean()
        sep = ((g.A_miss + g.A_false) / (1 - g.dice_A) > 0.5) & ((g.B_fill + g.B_bnd) / (1 - g.dice_B) > 0.5)
        lines.append(f"| {int(f * 100)}% | {len(g)} | {g.dice_A.mean():.3f} | {g.dice_B.mean():.3f} | {ad:.3f} | {adl:.3f} | {bd:.3f} | {bdl:.3f} | {g.erode_um.mean():.0f} | {sep.sum()} of {len(g)} |")
    lines += ["", "separated = the pair has the same Dice but the deletion error is mostly detection (> 50%) and the erosion error mostly delineation (> 50%).", ""]
    (OUT / "protocol_validation.md").write_text("\n".join(lines) + "\n")
    # figure: (a) which bins each perturbation moves, (b) equal-Dice pairs
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.0, 1.3]})
    bottom = np.zeros(len(PERT))
    for b in BINS:
        ax[0].bar(range(len(PERT)), m[b].values, bottom=bottom, color=COL[b], label=NAME[b], width=0.72)
        bottom += m[b].values
    area_by = dl.groupby("pert").area_deleted.mean()                        # achieved share of the tumour area (nominal 10 / 30 / 50% -> 36 / 36 / 74%)
    shown = [f"delete {area_by[p] * 100:.0f}% of area" if p.startswith("delete") else p for p in PERT]
    ax[0].set_xticks(range(len(PERT)))
    ax[0].set_xticklabels(shown, rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("1 - Dice, split into the four bins")
    ax[0].set_title("a  What each perturbation moves (mean of 10 real masks)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    labels, k = [], 0
    for f, g in ok.groupby("f"):
        for who, pre in (("deletion", "A"), ("erosion", "B")):
            bt = 0
            for b in BINS:
                v = g[f"{pre}_{b}"].mean()
                ax[1].bar(k, v, bottom=bt, color=COL[b], width=0.7)
                bt += v
            labels.append(f"deletion\n{area_by[f'delete {int(f * 100)}%'] * 100:.0f}% area" if who == "deletion" else "erosion\nsame Dice")
            k += 1
        k += 0.6
    ax[1].set_xticks([i + 0.3 * (i // 2) * 0 for i in range(0)])
    xs = [j + 0.6 * (j // 2) for j in range(len(labels))]
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_title("b  Same Dice, different error (equal-Dice pairs)", loc="left", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    cap = ("10 real annotation masks (5 biopsies, 5 large sections) of the 47-patient set; slides with >= 8 countable regions (>= 2,500 um^2), one per patient, drawn with "
           "numpy default_rng(0); 4096 x 4096 px window (2.05 mm) with the most regions, 2 to 4 regions per window. Deletions are labelled by the achieved share of the "
           "tumour area; the two 36% bars coincide (one region deleted for nominal 10% and 30%).")
    import textwrap
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(cap, 190)), fontsize=7, color="#52514e", va="bottom")
    (OUT / "fig_validation_caption.txt").write_text(cap + "\n")
    fig.savefig(OUT / "fig_validation.pdf")
    fig.savefig(OUT / "fig_validation.png", dpi=170)


def main():
    slides = pick_slides()
    print("masks:", slides, flush=True)
    with ProcessPoolExecutor(5) as ex:
        res = list(ex.map(one_mask, [(i, s, t) for i, (s, t) in enumerate(slides)]))
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    pr = pd.DataFrame([p for _, ps in res for p in ps])
    df.to_csv(OUT / "validation_rows.csv", index=False)
    pr.to_csv(OUT / "validation_pairs.csv", index=False)
    report(df, pr)
    print("done")


if __name__ == "__main__":
    main()
