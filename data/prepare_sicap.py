"""
Convert the SICAPv2 Kaggle copy into the layout train.py reads, for pretraining.

    python prepare_sicap.py --src str(P.SICAP_RAW)

Writes:
    $SITESHIFT_DATA/sicapv2/images/<n>.jpg   the original JPEG tiles, byte-copied
    $SITESHIFT_DATA/sicapv2/masks/<n>.png    0 = not cancer, 255 = cancer (any Gleason grade)
    sicap_tiles.csv     one row per tile: slide (= overlap region, see below), image, tissue_frac,
                        tumour_frac, and the share of the tile under each grey level of the
                        original mask (level_50 ... level_255), kept for a later Gleason-grade model
    sicap_split.json    train / val regions for pretraining; "guard" regions are used by neither.
                        Never overwritten without --force.

What the Kaggle copy is (checked on all 18,783 tiles, 2026-10-19):
- The 18,783 SICAPv2 patches (512 x 512 RGB JPEG), renamed 0 ... 18782; Train/ holds 0 ... 18764,
  Val/ the last 18. The original names (slide id, coordinates) are gone.
- Masks are greyscale JPEGs. Labelled pixels sit at about 11 grey levels (50, 65, 75, 85, 100,
  150, 165, 175, 185, 200, 255); some levels only occur in parts of the numbering, so the grey
  value is not one fixed Gleason code. JPEG noise adds values 1-4 around 0 and +-4 around each
  level. Every pixel >= 25 counts as cancer here; unlabelled tissue (benign glands, stroma) is 0.
- Neighbouring patches overlap by half a tile, pixel for pixel, and the numbering keeps the
  original row-by-row order (80% of tiles continue at index + 1). Tiles linked by such overlaps
  form regions (238); a tile and the tiles it overlaps always end up in the same split.
- The Kaggle Val/ tiles overlap Train/ tiles, so they are not a held-out set; this script treats
  them like every other tile.

Split: the numbering follows the original file names, which start with the slide id, so one
slide's tiles sit in one stretch of the numbering. The tiles are cut into --chunks stretches,
only at points no region crosses; val is one stretch from each third of the numbering (the
combination whose tile and cancer-pixel shares are closest to --val), and the regions within
--guard positions of a val stretch are left out of training, so that a slide cut by a stretch
border cannot sit on both sides.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import itertools
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prepare import tissue_fraction

HERE = P.INDEX
LEVELS = np.array([50, 65, 75, 85, 100, 150, 165, 175, 185, 200, 255])
CANCER_FROM = 25            # mask grey value from which a pixel counts as cancer
THUMB = 64                  # tile thumbnail side for finding overlaps


def convert(job):
    src, dst, sub, n = job
    img_src, mask_src = src / sub / "Images" / f"{n}.jpg", src / sub / "Masks" / f"{n}.jpg"
    img_dst, mask_dst = dst / "images" / f"{n}.jpg", dst / "masks" / f"{n}.png"
    if not (img_dst.exists() and img_dst.stat().st_size == img_src.stat().st_size):
        shutil.copyfile(img_src, img_dst)
    raw = np.array(Image.open(mask_src))
    if raw.ndim == 3:
        raw = raw[..., 0]
    cancer = raw >= CANCER_FROM
    if not mask_dst.exists():
        Image.fromarray((cancer * 255).astype(np.uint8)).save(mask_dst)
    nearest = np.abs(raw[cancer].astype(np.int16)[:, None] - LEVELS[None]).argmin(1)
    level = np.bincount(nearest, minlength=len(LEVELS)) / raw.size
    im = Image.open(img_src).convert("RGB")
    rgb = np.array(im)
    row = dict(index=n, source=sub, image=f"images/{n}.jpg", tissue_frac=round(tissue_fraction(rgb), 4),
               tumour_frac=float(cancer.mean()))
    row.update({f"level_{v}": round(float(f), 5) for v, f in zip(LEVELS, level)})
    return row, np.array(im.resize((THUMB, THUMB), Image.BILINEAR))


def overlap_regions(thumbs, max_diff=1.0, min_texture=8.0):
    """Label each tile with its region: tiles joined by a half-tile overlap (right half of one =
    left half of another, or bottom half = top half). Halves without texture (glass) are not
    used to join tiles, since any two blank halves look alike."""
    import torch                    # here, not at the top: the converting workers do not need it
    dev ="cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor(thumbs, dtype=torch.float32, device=dev)            # N, 64, 64, 3
    n, h = len(t), THUMB // 2
    inner = slice(1, THUMB - 1)                                          # skip the outer pixel line
    halves = {"right": t[:, inner, h + 1:THUMB - 1], "left": t[:, inner, 1:h - 1],
              "bottom": t[:, h + 1:THUMB - 1, inner], "top": t[:, 1:h - 1, inner]}
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for first, second in [("right", "left"), ("bottom", "top")]:
        a, b = halves[first].reshape(n, -1), halves[second].reshape(n, -1)
        za = a - a.mean(1, keepdim=True)
        za = za / (za.norm(dim=1, keepdim=True) + 1e-6)
        zb = b - b.mean(1, keepdim=True)
        zb = zb / (zb.norm(dim=1, keepdim=True) + 1e-6)
        match = torch.empty(n, dtype=torch.long, device=dev)
        for s in range(0, n, 2048):                                      # most similar other tile
            c = za[s:s + 2048] @ zb.T
            c[torch.arange(len(c)), torch.arange(s, s + len(c))] = -2
            match[s:s + 2048] = c.argmax(1)
        diff = (a - b[match]).abs().mean(1)
        ok = ((diff < max_diff) & (a.std(1) > min_texture)).cpu().numpy()
        for i, j in zip(np.where(ok)[0], match.cpu().numpy()[ok]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    first_tile = {r: i for i, r in reversed(list(enumerate(roots)))}     # number regions in tile order
    order = {r: k for k, r in enumerate(sorted(first_tile, key=first_tile.get))}
    return np.array([order[r] for r in roots])


def make_split(df, chunks, val_share, guard):
    n = len(df)
    region = df.region.values
    lo = df.groupby("region")["pos"].min().reindex(range(region.max() + 1)).values
    hi = df.groupby("region")["pos"].max().reindex(range(region.max() + 1)).values
    crossing = np.zeros(n + 1, int)                  # regions spanning the cut between pos-1 and pos
    np.add.at(crossing, lo + 1, 1)
    np.add.at(crossing, hi + 1, -1)
    safe = np.where(np.cumsum(crossing)[:n] == 0)[0]                     # cut before these positions
    cuts = sorted({int(safe[np.abs(safe - round(k * n / chunks)).argmin()]) for k in range(1, chunks)})
    edges = [0] + cuts + [n]
    chunk = np.searchsorted(np.array(cuts), np.arange(n), side="right")
    tumour_px = df.tumour_frac.values
    tile_share = np.bincount(chunk, minlength=len(edges) - 1) / n
    tumour_share = np.bincount(chunk, weights=tumour_px, minlength=len(edges) - 1) / tumour_px.sum()
    thirds = np.array_split(np.arange(len(edges) - 1), 3)
    best = min(itertools.product(*thirds), key=lambda c: abs(tile_share[list(c)].sum() - val_share)
               + abs(tumour_share[list(c)].sum() - val_share))
    val_regions = set(region[np.isin(chunk, best)])
    near = np.zeros(n, bool)
    for c in best:
        a, b = edges[c], edges[c + 1]
        near[max(0, a - guard):a] = True
        near[b:min(n, b + guard)] = True
    guard_regions = set(region[near]) - val_regions
    split = np.where(np.isin(region, list(val_regions)), "val",
                     np.where(np.isin(region, list(guard_regions)), "guard", "train"))
    return split, chunk, [(edges[c], edges[c + 1] - 1) for c in best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(str(P.SICAP_RAW)))
    ap.add_argument("--out", type=Path, default=Path(str(P.DATA_ROOT / "sicapv2")))
    ap.add_argument("--chunks", type=int, default=30, help="stretches of the numbering to choose val from")
    ap.add_argument("--val", type=float, default=0.10, help="target share of tiles and cancer pixels in val")
    ap.add_argument("--guard", type=int, default=150,
                    help="regions within this many tiles of a val stretch are used by neither split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="overwrite sicap_split.json")
    a = ap.parse_args()

    split_json, index_csv = HERE / "sicap_split.json", HERE / "sicap_tiles.csv"
    if split_json.exists() and not a.force:
        raise SystemExit(f"{split_json} exists; pass --force to rebuild it")
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "masks").mkdir(parents=True, exist_ok=True)
    jobs = [(a.src, a.out, sub, n) for sub in ["Train", "Val"]
            for n in sorted(int(p.stem) for p in (a.src / sub / "Images").glob("*.jpg"))]
    missing = [f"{s}/Masks/{n}.jpg" for _, _, s, n in jobs if not (a.src / s / "Masks" / f"{n}.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} image(s) without a mask, e.g. {missing[0]}")
    print(f"converting {len(jobs)} tiles from {a.src} -> {a.out} ...", flush=True)
    with Pool(a.workers) as pool:
        out = pool.map(convert, jobs, chunksize=64)
    df = pd.DataFrame([r for r, _ in out]).sort_values("index").reset_index(drop=True)
    thumbs = np.stack([t for _, t in out])[np.argsort([r["index"] for r, _ in out])]
    if not (df["index"].values == np.arange(len(df))).all():
        raise SystemExit("tile numbers are not 0 ... N-1 without gaps; the overlap order cannot be trusted")

    print("finding overlapping tiles ...", flush=True)
    df["pos"] = np.arange(len(df))
    df["region"] = overlap_regions(thumbs)
    df["slide"] = [f"region_{r:03d}" for r in df.region]
    df["group"] = df.slide
    df["split"], df["chunk"], val_ranges = make_split(df, a.chunks, a.val, a.guard)

    cols = ["slide", "group", "image", "tissue_frac", "tumour_frac", "split", "chunk", "source"] + \
           [c for c in df.columns if c.startswith("level_")]
    df[cols].to_csv(index_csv, index=False)
    summary = {}
    for s in ["train", "val", "guard"]:
        d = df[df.split == s]
        summary[s] = dict(tiles=len(d), regions=int(d.region.nunique()),
                          tile_share=round(len(d) / len(df), 4),
                          cancer_pixel_share=round(d.tumour_frac.sum() / df.tumour_frac.sum(), 4),
                          tiles_with_cancer=int((d.tumour_frac > 0).sum()))
    split_json.write_text(json.dumps(dict(
        source=str(a.src), chunks=a.chunks, val=a.val, guard=a.guard, val_tile_ranges=val_ranges,
        splits={s: sorted(df.slide[df.split == s].unique().tolist()) for s in ["train", "val", "guard"]},
        summary=summary), indent=2))

    print(f"{len(df)} tiles, {df.region.nunique()} regions, {(df.tumour_frac > 0).sum()} with cancer, "
          f"{df.tumour_frac.mean():.1%} of pixels cancer")
    for s, v in summary.items():
        print(f"  {s:5s}: {v['tiles']:5d} tiles ({v['tile_share']:.1%}), {v['regions']:3d} regions, "
              f"{v['cancer_pixel_share']:.1%} of cancer pixels")
    print(f"val stretches (tile numbers): {val_ranges}")
    print(f"-> {index_csv}, {split_json}")


if __name__ == "__main__":
    main()
