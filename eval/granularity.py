"""
How finely each source outlines tumour (SICAPv2, Karolinska, ours), from the label geometry and the
image brightness only, and the closing radius for the SICAPv2 relabelling test (prepare_sicap_closed.py):

    python granularity.py           # cv/granularity.md, cv/granularity_windows.csv, cv/sicap_closing.json

A stain-based epithelium / stroma split was tried first (2026-10-25) and dropped: per-slide Macenko
fits, fixed Ruifrok vectors with a global or a per-slide threshold, a nuclei top-hat density and a
per-slide two-cluster split all failed to separate epithelium from stroma consistently on our
purple-tinted slides (cv/tissue_*check.png, local only).

Windows of 256 x 256 um at 1 um per pixel, cut from the centre of tiles with tumour (masks resampled
by nearest neighbour, images by box filter):
  ours        0.5 um/px, the whole tile; up to 150 tiles per patient (17 patients); unit = patient
  Karolinska  0.503 um/px, centre crop; 2,000 random tiles of the 1,092 test slides; unit = slide
  SICAPv2     0.875 um/px (the x1.75 of evaluate_sicap.py), centre 293 px; 2,000 random training
              tiles (the set that picks the closing radius); unit = overlap region
Per unit, then mean over units [95% bootstrap CI]:
  empty inside tumour   share of tumour-labelled pixels that are glass or lumen (optical density
                        below 0.15 in every channel, stain.BETA)
  outline density       outline length per tumour area, um per 1,000 um^2 (window borders not counted;
                        pixel edges, so diagonal outlines count up to ~1.3x, the same for every source)
  regions               8-connected tumour regions per window; median area of regions not cut by the
                        window border; share of regions cut by it; median solidity (area / convex hull)
  gap fill at p um      tumour area a closing with a disk of radius p (10, 20, 40 um) would add, as a
                        share of the labelled area: gaps between labelled glands that a region-style
                        outline would include (unlike outline density, hardly affected by region size)
SICAPv2 relabelling (fixed 2026-10-25, before any relabelled training; relabel()): closing of the
native SICAPv2 mask with a disk of radius r um (reflect padding at the tile border), except on glass
(empty areas >= 2,500 um^2 and debris specks inside them) - plain closing also filled the glass between
biopsy fragments (cv/sicap_closing_examples.png), and leaving out every empty pixel instead punched
pinholes into the filled regions and kept debris specks as islands.
r = 0, 5, ..., 80 swept on the SICAPv2 training sample; the rule "pooled gap fill at 10, 20 and 40 um
closest to ours (sum of squared differences)" is reported, but the sweep flattens once the gaps are
filled, so two radii are trained as a dose-response: 40 and 80 um. (Matching the pooled outline
density was the first rule; it cannot work: SICAPv2 at r = 80 um still has 2.3x our outline density,
because its tumour regions are smaller - biopsies - not because of how they are outlined.)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

from cv_summary import bootstrap_ci
from data import load_mask, load_rows, mask_path
from stain import BETA

HERE = P.INDEX
W = 256                                   # window side, um (= pixels at 1 um/px)
RADII = list(range(0, 85, 5))
PROBES = [10, 20, 40]                     # gap-fill probe radii, um
TRAINED_RADII = [40, 80]                  # SICAPv2 relabellings that are trained (prepare_sicap_closed.py)
SOURCES = {
    "ours": dict(data=Path(str(P.DATASET47)), index=HERE / "tiles.csv", um=0.5),
    "Karolinska": dict(data=Path(str(P.DATA_ROOT / "karolinska_all")), index=HERE / "karolinska_all_tiles.csv", um=0.503),
    "SICAPv2": dict(data=Path(str(P.DATA_ROOT / "sicapv2")), index=HERE / "sicap_tiles.csv", um=0.875),
}


def close(mask, r_px):
    """Binary closing with a Euclidean disk of radius r_px, reflect padding."""
    if r_px <= 0 or not mask.any():
        return mask
    p = int(np.ceil(r_px)) + 2
    m = np.pad(mask, p, mode="reflect")
    dil = ndimage.distance_transform_edt(~m) <= r_px
    if dil.all():
        return np.ones_like(mask)
    return (ndimage.distance_transform_edt(dil) > r_px)[p:-p, p:-p]


def empty_pixels(rgb):
    """Glass, lumen or fat: optical density below stain.BETA in every channel."""
    return (-np.log((rgb.astype(np.float64) + 1) / 256) < BETA).all(-1)


GLASS_UM2, SPECK_UM2 = 2500, 200


def glass(rgb, um):
    """Empty areas of at least GLASS_UM2 (~50 x 50 um, larger than most gland lumens) plus specks of debris
    under SPECK_UM2 enclosed by them."""
    lab, _ = ndimage.label(empty_pixels(rgb), structure=np.ones((3, 3)))
    big = (np.bincount(lab.ravel()) * um * um >= GLASS_UM2)[lab] & (lab > 0)
    hl, _ = ndimage.label(ndimage.binary_fill_holes(big) & ~big)
    return big | ((np.bincount(hl.ravel()) * um * um < SPECK_UM2)[hl] & (hl > 0))


def relabel(mask, rgb, r_um, um):
    """Region-style SICAPv2 label: the closing of the mask with a disk of r_um, except on glass(); small
    lumens and pinholes inside tumour are filled like tissue."""
    if r_um <= 0 or not mask.any():
        return mask
    return mask | (close(mask, r_um / um) & ~glass(rgb, um))


def window(arr, um, resample):
    n = round(W / um)
    o0, o1 = (arr.shape[0] - n) // 2, (arr.shape[1] - n) // 2
    crop = arr[o0:o0 + n, o1:o1 + n]
    if n == W:
        return crop
    img = Image.fromarray(crop if crop.dtype == np.uint8 and crop.ndim == 3 else crop.astype(np.uint8) * 255)
    out = np.array(img.resize((W, W), resample))
    return out if crop.ndim == 3 else out > 127


def geometry(t):
    """Outline length (um) and region statistics of a W x W boolean tumour window."""
    outline = int((t[:, 1:] != t[:, :-1]).sum() + (t[1:] != t[:-1]).sum())
    lab, n = ndimage.label(t, structure=np.ones((3, 3)))
    regions = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        area = int(comp.sum())
        cut = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == W or sl[1].stop == W
        solidity = np.nan
        if area >= 20:
            yy, xx = np.nonzero(comp)
            pts = np.concatenate([np.stack([yy + dy, xx + dx], 1) for dy in (0, 1) for dx in (0, 1)])   # pixel corners
            try:
                solidity = area / ConvexHull(pts).volume
            except Exception:
                pass
        regions.append((area, cut, solidity))
    return outline, regions


def measure(job):
    source, data, um, image, unit, radii = job
    rgb = np.array(Image.open(data / image).convert("RGB"))
    mask = load_mask(data / mask_path(image)).astype(bool)
    empty = empty_pixels(window(rgb, um, Image.BOX))
    rows = []
    for r in radii:
        t = window(relabel(mask, rgb, r, um), um, Image.NEAREST)
        if not t.any():
            continue
        outline, regions = geometry(t)
        a = np.array([x[0] for x in regions]), np.array([x[1] for x in regions]), np.array([x[2] for x in regions])
        rows.append(dict(source=source, unit=unit, image=image, radius=r, tumour=int(t.sum()), empty_in_tumour=int((t & empty).sum()),
                         outline=outline, regions=len(regions), regions_cut=int(a[1].sum()),
                         median_area_uncut=float(np.median(a[0][~a[1]])) if (~a[1]).any() else np.nan,
                         median_solidity=float(np.nanmedian(a[2])) if np.isfinite(a[2]).any() else np.nan,
                         **{f"closed_{p}": int(close(t, p).sum()) for p in PROBES}))
    return rows


def gap_fill(d, p):
    return d[f"closed_{p}"].sum() / d.tumour.sum() - 1


def sample():
    jobs = []
    t = pd.read_csv(SOURCES["ours"]["index"])
    t = t[(t.tumour_frac > 0) & (t.tissue_frac >= 0.25)].assign(unit=lambda d: d.slide.str.split(".").str[0].astype(int))
    for _, g in t.groupby("unit"):
        jobs += [("ours", r.image, r.unit) for r in g.sample(min(150, len(g)), random_state=0).itertuples()]
    k = pd.read_csv(SOURCES["Karolinska"]["index"])
    test = set(json.loads((HERE / "karolinska_all_split.json").read_text())["splits"]["test"])
    k = k[(k.tumour_frac > 0) & (k.tissue_frac >= 0.25) & k.slide.isin(test)]
    jobs += [("Karolinska", r.image, r.slide) for r in k.sample(2000, random_state=0).itertuples()]
    s = load_rows(SOURCES["SICAPv2"]["index"], HERE / "sicap_split.json", "train")
    s = s[s.tumour_frac > 0]
    jobs += [("SICAPv2", r.image, r.slide) for r in s.sample(2000, random_state=0).itertuples()]
    return [(src, SOURCES[src]["data"], SOURCES[src]["um"], img, unit, RADII if src == "SICAPv2" else [0]) for src, img, unit in jobs]


def per_unit(w):
    g = w.groupby("unit")
    return pd.DataFrame({"empty_in_tumour": g.empty_in_tumour.sum() / g.tumour.sum(),
                         **{f"gap_fill_{p}": g[f"closed_{p}"].sum() / g.tumour.sum() - 1 for p in PROBES},
                         "outline_density": 1000 * g.outline.sum() / g.tumour.sum(),
                         "regions_per_window": g.regions.mean(),
                         "share_cut": g.regions_cut.sum() / g.regions.sum(),
                         "median_area_uncut": g.median_area_uncut.median(),
                         "median_solidity": g.median_solidity.median()})


def main():
    with Pool(8) as pool:
        w = pd.DataFrame([r for rows in pool.imap_unordered(measure, sample(), chunksize=8) for r in rows])
    w.to_csv(HERE / "cv" / "granularity_windows.csv", index=False)

    ours = w[(w.source == "ours") & (w.radius == 0)]
    target = {p: gap_fill(ours, p) for p in PROBES}
    sic = w[w.source == "SICAPv2"]
    sweep = sic.groupby("radius").apply(lambda d: pd.Series({
        **{f"gap_fill_{p}": gap_fill(d, p) for p in PROBES},
        "outline_density": 1000 * d.outline.sum() / d.tumour.sum(), "empty_in_tumour": d.empty_in_tumour.sum() / d.tumour.sum(),
        "regions_per_window": d.regions.mean(), "tumour_share_of_window": d.tumour.sum() / (len(d) * W * W)}), include_groups=False)
    sweep["distance_to_ours"] = sum((sweep[f"gap_fill_{p}"] - target[p]) ** 2 for p in PROBES)
    r_best = int(sweep.distance_to_ours.idxmin())
    (HERE / "cv" / "sicap_closing.json").write_text(json.dumps(
        {"radius_um": r_best, "um_per_px": SOURCES["SICAPv2"]["um"], "radius_px": r_best / SOURCES["SICAPv2"]["um"],
         "criterion": "pooled gap fill at 10/20/40 um of SICAPv2 training windows closest to ours (sum of squares)",
         "ours_gap_fill": target, "sicap_gap_fill_at_radius": {p: float(sweep[f"gap_fill_{p}"][r_best]) for p in PROBES}}, indent=2))

    sets = {"ours": ours, "Karolinska": w[w.source == "Karolinska"], "SICAPv2": sic[sic.radius == 0],
            **{f"SICAPv2 relabelled, r = {r} um": sic[sic.radius == r] for r in TRAINED_RADII}}
    rows = []
    for name, d in sets.items():
        u = per_unit(d)
        r = {"labels": name, "units": len(u), "windows": len(d)}
        for col, fmt in [("empty_in_tumour", "{:.3f}"), *[(f"gap_fill_{p}", "{:.3f}") for p in PROBES], ("outline_density", "{:.1f}"),
                         ("regions_per_window", "{:.2f}"), ("share_cut", "{:.2f}"), ("median_area_uncut", "{:.0f}"), ("median_solidity", "{:.2f}")]:
            v = u[col].dropna()
            if len(v):
                lo, hi = bootstrap_ci(v)
                r[col] = f"{fmt.format(v.mean())} [{fmt.format(lo)}, {fmt.format(hi)}]"
            else:
                r[col] = "- (no region inside a window)"
        rows.append(r)
    table = pd.DataFrame(rows).rename(columns={
        "empty_in_tumour": "empty (glass/lumen) inside tumour", **{f"gap_fill_{p}": f"gap fill at {p} um" for p in PROBES},
        "outline_density": "outline um per 1,000 um^2",
        "regions_per_window": "regions per window", "share_cut": "regions cut by window",
        "median_area_uncut": "median area of uncut regions, um^2", "median_solidity": "median solidity"})
    lines = ["# Annotation granularity: how finely each source outlines tumour", "",
             "256 x 256 um windows at 1 um/px from tiles with tumour; mean over units (patients / slides / SICAPv2 regions) [95% CI]. "
             "Geometry and brightness only (a stain-based epithelium / stroma split was unreliable on our slides).", "",
             table.to_markdown(index=False), "",
             f"## Closing radius for the SICAPv2 relabelling (chosen before training)", "",
             "Target: pooled gap fill of ours at " + ", ".join(f"{p} um {target[p]:.3f}" for p in PROBES)
             + f". Closest by sum of squares: r = {r_best} um; trained: r = "
             + " and ".join(f"{r} um ({r / SOURCES['SICAPv2']['um']:.1f} SICAPv2 pixels)" for r in TRAINED_RADII)
             + ". Relabelling closes over tissue and small lumens, not over glass.", "",
             sweep.reset_index().to_markdown(index=False, floatfmt=".3f"), ""]
    (HERE / "cv" / "granularity.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
