"""
Detection-delineation protocol: every model's result split into finding the tumour (detection) and
drawing it (delineation), from the predictions saved by protocol_predict.py (run_protocol.py).

    python protocol.py [<prediction folder> ...]     # analyse folders (default: every finished one not yet
                                                     # analysed; cached in <folder>/protocol_*.csv)

Fixed before any result was seen (2026-10-29):
  regions     connected components (8-connected) of the annotated tumour mask across the whole slide
              (tiles stitched on their grid; a missing tile counts as glass, as in inference), in tile mode
              as well: the tile-mode predictions are stitched for the analysis only. SICAPv2 tiles have no
              grid position (and overlap), so there regions are per tile and the tile border is not an outline.
              Changed from 'per tile in tile mode' after a one-slide smoke test of our own model (7.5, before
              any model comparison): per tile, 44% of its missed pixels fell in 'missed regions' that were
              slivers of large regions cut by tile borders (239 tile pieces vs 10 slide regions, all found).
              The per-tile variant is kept as a sensitivity check (protocol_*_tile.csv).
  detected    the model marks at least DETECT = 10% of the region's area
  band        pixels within BAND_UM = 50 um of the annotated outline: outside the tumour the distance to the nearest tumour pixel, inside the distance to the
              outline pixels (tumour pixels with a non-tumour 8-neighbour), by EUCLIDEAN distance in um (distance transform with the
              pixel size as sampling; the paper's definition). Until 2026-11-08 this was a square neighbourhood (Chebyshev distance) and every
              result made with it is deprecated (files deprecated_square_band_*); breakdown.py / granularity.py still use squares
  counting    region counts (recall, false regions) use components of at least MIN_UM2 = 2,500 um^2
              (about one small gland, 50 x 50 um); every pixel counts in the pixel measures. As in the reference, a region below MIN_UM2 is never
              'found': its unmarked pixels are missed (changed 2026-11-08 after the parity check)
Detection measures
  clean-tile AUC   karolinska_all_report.per_unit: tiles >= 50% under the outline vs tiles without tumour,
                   tiles with >= 25% tissue, score = predicted tumour share of the tile
  recall           share of annotated regions (>= MIN_UM2) detected; recall_area = share of tumour area
                   in detected regions
  false regions    predicted components with no pixel within 50 um of annotated tumour: number
                   (>= MIN_UM2) per mm^2 of tumour-free tissue, and their area as a share of that tissue
Delineation measures
  dice_found       Dice between the detected regions and the predicted components that overlap them
  dice_tolerant    Dice with the band left out
  fill             share of the detected regions' area that is marked
Error bins: every wrong pixel goes to exactly one bin; the bins add up to fp + fn
  missed     tumour pixels of regions that were not detected                   (detection)
  fill       unmarked tumour pixels of detected regions, outside the band       (delineation: interior)
  boundary   unmarked tumour pixels of detected regions inside the band, and
             false tumour within 50 um of annotated tumour                      (delineation: outline)
  false      false tumour farther than 50 um from any annotated tumour          (detection)
Divided by 2 tp + fp + fn, the bins add up to 1 - Dice of the unit (err_<bin>); as shares of fp + fn: share_<bin>.
Ranking scores: detection = clean-tile AUC, delineation = dice_found.
SICAPv2: most regions lack tumour-free (or mostly tumour) tiles, so its clean-tile AUC is one value pooled over
all tiles (per model, then the mean of the models).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from data import grid_position
from karolinska_all_report import CLEAN, auc

HERE = P.INDEX
PRED = Path(str(P.PRED))
DETECT, BAND_UM, MIN_UM2 = 0.10, 50, 2500
EIGHT = np.ones((3, 3), bool)
SUMS = ["tp", "fp", "fn", "tumour_px", "benign_px", "fp_tissue", "tp_st", "fp_st", "fn_st",
        "missed", "fill_px", "fn_band", "fp_band", "false",
        "regions", "regions_found", "region_px", "region_px_found", "tp_found", "pred_px_matched",
        "false_regions", "false_region_px"]
BINS = ["missed", "fill", "boundary", "false"]
UM_PER_PX = {"ours": 0.5, "karolinska": 0.503, "sicap": 1 / 1.75}
INDEX = {"ours": "tiles.csv", "karolinska": "karolinska_all_tiles.csv", "sicap": "sicap_tiles.csv"}


def dilate(x, r):
    """Square dilation of a bool array by r pixels; outside the array counts as False."""
    y = ndi.maximum_filter1d(x.view(np.uint8), 2 * r + 1, axis=0, mode="constant")
    return ndi.maximum_filter1d(y, 2 * r + 1, axis=1, mode="constant").view(bool)


def within_band(src, um_per_px):
    """True where the Euclidean distance (um) to the nearest True pixel of src is at most BAND_UM. Outside the array nothing is known (no pixels)."""
    if not src.any():
        return np.zeros(src.shape, bool)
    if src.all():
        return np.ones(src.shape, bool)
    return ndi.distance_transform_edt(~src, sampling=um_per_px) <= BAND_UM


def count(labels, sel, n):
    return np.bincount(labels.ravel() if sel is None else labels[sel], minlength=n + 1)[1:]


def analyse_slide(path, um_per_px, stitch):
    """Sums (SUMS) of one slide's saved predictions and per-tile scores (image, score, tumour share)."""
    z = np.load(path)
    S, images = int(z["size"]), z["image"]
    n = len(images)
    r, min_px = int(np.ceil(BAND_UM / um_per_px)) + 1, MIN_UM2 / um_per_px ** 2     # r: padding around a tile (band radius + 1 px: the outline pixels at the band's edge need their neighbours); the band itself is 50 um exactly

    packed = {k: z[k] for k in ("pred", "truth", "tissue")}     # read once: each z[k] decompresses the file

    def unpack(a, i):
        return np.unpackbits(packed[a][i], count=S * S).reshape(S, S).view(bool)

    @lru_cache(maxsize=1024)                                  # tiles come row by row: a few rows of neighbours
    def truth(i):
        return unpack("truth", i)

    pos = [grid_position(p) for p in images] if stitch else None
    at = {p: i for i, p in enumerate(pos)} if stitch else {}
    s = dict.fromkeys(SUMS, 0)
    score, frac = np.zeros(n), np.zeros(n)
    t_parts, p_parts, pairs, borders = [], [], [], []
    t_off = p_off = 0
    for i in range(n):
        T, P, tissue = truth(i), unpack("pred", i), unpack("tissue", i)
        # truth around the tile (r pixels of the neighbours when stitched), and what is known there
        canvas, known = np.zeros((S + 2 * r,) * 2, bool), np.zeros((S + 2 * r,) * 2, bool)
        canvas[r:r + S, r:r + S], known[r:r + S, r:r + S] = T, True
        if stitch:
            known[:] = True                                       # a missing tile is glass
            c0, r0 = pos[i]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = at.get((c0 + dc, r0 + dr))
                    if j is None or j == i:
                        continue
                    y0, x0 = r + dr * S, r + dc * S              # neighbour's corner in the canvas
                    ys, ye, xs, xe = max(y0, 0), min(y0 + S, S + 2 * r), max(x0, 0), min(x0 + S, S + 2 * r)
                    canvas[ys:ye, xs:xe] = truth(j)[ys - y0:ye - y0, xs - x0:xe - x0]
        # the reference's band (attribution.py): outside the tumour, within 50 um of any tumour pixel; inside, within 50 um of the OUTLINE = tumour
        # pixels with a known non-tumour 8-neighbour (unknown pixels and the canvas border do not create an outline)
        tumour_k = canvas & known
        outline = tumour_k & ~ndi.binary_erosion(tumour_k | ~known, EIGHT, border_value=1)
        near_tumour = within_band(tumour_k, um_per_px)[r:r + S, r:r + S]
        near_outline = within_band(outline, um_per_px)[r:r + S, r:r + S]
        band = np.where(T, near_outline, near_tumour)
        TP, FP, FN = P & T, P & ~T, ~P & T
        fp_band = int((FP & near_tumour).sum())
        s["tp"] += int(TP.sum()); s["fp"] += int(FP.sum()); s["fn"] += int(FN.sum()); s["tumour_px"] += int(T.sum())
        s["benign_px"] += int((tissue & ~T).sum()); s["fp_tissue"] += int((FP & tissue).sum())
        s["tp_st"] += int((TP & ~band).sum()); s["fn_st"] += int((FN & ~band).sum())
        s["fp_band"] += fp_band; s["false"] += int(FP.sum()) - fp_band
        score[i], frac[i] = P.mean(), T.mean()

        lt, nt = ndi.label(T, EIGHT)
        lp, np_ = ndi.label(P, EIGHT)
        t_parts.append(np.stack([count(lt, None, nt), count(lt, TP, nt), count(lt, FN & band, nt),
                                 count(lt, FN & ~band, nt)], 1))
        p_parts.append(np.stack([count(lp, None, np_), count(lp, P & (near_tumour | T), np_)], 1))
        both = lp[TP] * (nt + 1) + lt[TP]
        u = np.unique(both)
        pairs.append(np.stack([u // (nt + 1) - 1 + p_off, u % (nt + 1) - 1 + t_off], 1))
        if stitch:
            borders.append((i, [np.where(e > 0, e - 1 + t_off, -1).astype(np.int32) for e in (lt[0], lt[-1], lt[:, 0], lt[:, -1])],
                            [np.where(e > 0, e - 1 + p_off, -1).astype(np.int32) for e in (lp[0], lp[-1], lp[:, 0], lp[:, -1])]))
        t_off += nt
        p_off += np_
    s["fp_st"] = s["false"]

    tc = np.concatenate(t_parts) if t_off else np.zeros((0, 4), int)
    pc = np.concatenate(p_parts) if p_off else np.zeros((0, 2), int)
    pr = np.concatenate(pairs) if pairs else np.zeros((0, 2), int)
    t_group, p_group = np.arange(t_off), np.arange(p_off)
    if stitch and borders:
        side = {i: (tb, pb) for i, tb, pb in borders}            # top, bottom, left, right label lines
        t_edges, p_edges = [], []
        for i in side:
            c0, r0 = pos[i]
            for which, edges in ((0, t_edges), (1, p_edges)):
                top, bottom, left, right = side[i][which]
                j = at.get((c0 + 1, r0))                          # right neighbour: column to column, 8-connected
                if j is not None:
                    other = side[j][which][2]
                    for d in (-1, 0, 1):
                        a_, b_ = right[max(0, -d):S - max(0, d)], other[max(0, d):S - max(0, -d)]
                        edges.append(np.stack([a_, b_], 1))
                j = at.get((c0, r0 + 1))                          # neighbour below: row to row
                if j is not None:
                    other = side[j][which][0]
                    for d in (-1, 0, 1):
                        a_, b_ = bottom[max(0, -d):S - max(0, d)], other[max(0, d):S - max(0, -d)]
                        edges.append(np.stack([a_, b_], 1))
                j = at.get((c0 + 1, r0 + 1))                      # corners
                if j is not None:
                    edges.append(np.array([[bottom[-1], side[j][which][0][0]]]))
                j = at.get((c0 - 1, r0 + 1))
                if j is not None:
                    edges.append(np.array([[bottom[0], side[j][which][0][-1]]]))

        def groups(edges, n_nodes):
            if not n_nodes:
                return np.arange(0)
            e = np.concatenate(edges) if edges else np.zeros((0, 2), int)
            e = e[(e >= 0).all(1)]
            g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes))
            return connected_components(g, directed=False)[1]

        t_group, p_group = groups(t_edges, t_off), groups(p_edges, p_off)

    nt_g = t_group.max() + 1 if t_off else 0
    area, tp_r, fnb, fno = (np.bincount(t_group, weights=tc[:, k], minlength=nt_g) for k in range(4))
    found = tp_r >= DETECT * area
    big = area >= min_px
    counted = found & big                                      # reference: only detected regions of at least MIN_UM2 are "found"; the rest is missed
    s["missed"] = int((fnb + fno)[~counted].sum()); s["fill_px"] = int(fno[counted].sum()); s["fn_band"] = int(fnb[counted].sum())
    s["regions"] = int(big.sum()); s["regions_found"] = int((big & found).sum())
    s["region_px"] = int(area.sum()); s["region_px_found"] = int(area[found].sum()); s["tp_found"] = int(tp_r[found].sum())
    np_g = p_group.max() + 1 if p_off else 0
    p_area = np.bincount(p_group, weights=pc[:, 0], minlength=np_g)
    touch = np.bincount(p_group, weights=pc[:, 1], minlength=np_g) > 0
    matched = np.zeros(np_g, bool)
    if len(pr):
        matched[p_group[pr[:, 0]][found[t_group[pr[:, 1]]]]] = True
    s["pred_px_matched"] = int(p_area[matched].sum())
    s["false_regions"] = int((~touch & (p_area >= min_px)).sum()); s["false_region_px"] = int(p_area[~touch].sum())
    assert s["missed"] + s["fill_px"] + s["fn_band"] == s["fn"], path
    return str(z["slide"]), s, pd.DataFrame({"image": images, "score": score, "tumour_share": frac})


def cache(folder, per_tile):
    sfx = "_tile" if per_tile else ""
    return Path(folder) / f"protocol_slides_euclid{sfx}.csv", Path(folder) / f"protocol_tiles_euclid{sfx}.csv"     # "_euclid": never reuse a square-band cache


def is_cached(folder, per_tile):
    out_s, out_t = cache(folder, per_tile)
    return out_s.exists() and out_t.exists() and out_s.stat().st_mtime >= (Path(folder) / "done.json").stat().st_mtime


def analyse(folder, per_tile=False, workers=8):
    """Per-slide sums and per-tile scores of one prediction folder, cached next to the predictions."""
    folder = Path(folder)
    out_s, out_t = cache(folder, per_tile)
    if is_cached(folder, per_tile):
        return pd.read_csv(out_s), pd.read_csv(out_t)
    done = json.loads((folder / "done.json").read_text())
    dataset = next(k for k, v in INDEX.items() if Path(done["index"]).name == v)
    stitch = dataset != "sicap" and not per_tile
    files = sorted(folder.glob("*.npz"))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(analyse_slide, files, [UM_PER_PX[dataset]] * len(files), [stitch] * len(files)))
    slides = pd.DataFrame([dict(slide=k, **s) for k, s, _ in res])
    tiles = pd.concat([t.assign(slide=k) for k, _, t in res])
    slides.to_csv(out_s, index=False)
    tiles.to_csv(out_t, index=False)
    return slides, tiles


def unit_key(slides, dataset):
    """Unit of the slides: patient for our data (slide name before the first dot), else the slide itself."""
    return slides.str.split(".").str[0].astype(int) if dataset == "ours" else slides


def measures(slides, tiles, dataset, tissue):
    """Per-unit measures from one prediction folder's sums and tile scores."""
    g = slides.groupby(unit_key(slides.slide, dataset))[SUMS].sum()
    cancer = g.tumour_px > 0
    d = (2 * g.tp + g.fp + g.fn).clip(lower=1)
    wrong = (g.fp + g.fn).clip(lower=1)
    mm2 = g.benign_px * UM_PER_PX[dataset] ** 2 / 1e6
    bins = {"missed": g.missed, "fill": g.fill_px, "boundary": g.fn_band + g.fp_band, "false": g["false"]}
    out = pd.DataFrame({
        "dice": (2 * g.tp / d).where(cancer),
        "dice_tolerant": (2 * g.tp_st / (2 * g.tp_st + g.fp_st + g.fn_st).clip(lower=1)).where(cancer),
        "sensitivity": (g.tp / g.tumour_px.clip(lower=1)).where(cancer),
        "false_tumour": g.fp_tissue / g.benign_px.clip(lower=1),
        "recall": (g.regions_found / g.regions.clip(lower=1)).where(g.regions > 0),
        "recall_area": (g.region_px_found / g.region_px.clip(lower=1)).where(cancer),
        "false_regions_mm2": g.false_regions / mm2.where(mm2 > 0),
        "false_area": g.false_region_px / g.benign_px.clip(lower=1),
        "fill": (g.tp_found / g.region_px_found.clip(lower=1)).where(g.region_px_found > 0),
        "dice_found": (2 * g.tp_found / (g.pred_px_matched + g.region_px_found).clip(lower=1)).where(g.region_px_found > 0),
        **{f"err_{k}": (v / d).where(cancer) for k, v in bins.items()},
        **{f"share_{k}": (v / wrong).where(g.fp + g.fn > 0) for k, v in bins.items()},
        # post hoc (2026-10-30): the false bin = isolated false regions + spill of components that reach tumour
        "err_false_isolated": (g.false_region_px / d).where(cancer),
        "err_false_attached": ((g["false"] - g.false_region_px) / d).where(cancer),
    })
    t = clean_tiles(tiles, tissue)
    out["clean_auc"] = t.groupby(unit_key(t.slide, dataset)).apply(lambda x: auc(x.score, x.tumour_share > 0),
                                                                     include_groups=False)
    return out


def clean_tiles(tiles, tissue):
    """Tiles of the clean-tile AUC: >= 25% tissue, and either no tumour or at least CLEAN under the outline."""
    t = tiles.merge(tissue, on="image")
    return t[(t.tissue_frac >= 0.25) & ((t.tumour_share == 0) | (t.tumour_share >= CLEAN))]


def pooled_clean_auc(tiles, tissue):
    """One clean-tile AUC over all tiles (SICAPv2 regions are too small to have both kinds of tile each)."""
    t = clean_tiles(tiles, tissue)
    return auc(t.score, t.tumour_share > 0)


def main():
    folders = sys.argv[1:] or [d.parent for d in sorted(PRED.rglob("done.json"))]
    for f in folders:
        for per_tile in (False, True):
            if is_cached(f, per_tile):
                continue
            t0 = time.time()
            slides, _ = analyse(f, per_tile)
            print(f"{f}{' (per tile)' if per_tile else ''}: {len(slides)} slides, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
