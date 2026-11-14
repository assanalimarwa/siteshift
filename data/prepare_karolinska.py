"""
Convert a sample of the PANDA Karolinska biopsies into the tile layout train.py and evaluate.py read.

    python prepare_karolinska.py          # fmgrid env (needs tifffile + imagecodecs)

Source: <src>/Karolinska/images/<id>.tiff and masks/<id>_mask.tiff (PANDA, Karolinska provider),
<src>/Karolinska/labels.csv. Level 0 is 0.503 um/px (TIFF resolution tags), the scale of our tiles,
so tiles are taken as they are: the TIFF's own 512 x 512 tiles, decoded one by one.
Mask red channel: 0 background / unknown, 1 benign tissue, 2 cancer -> binary 255 = cancer.

Fixed choices (2026-10-24): --slides slides drawn at random with the natural mix of ISUP grades
(seed 0), split by slide 70 / 10 / 20 into train / val / test within each grade. PANDA has no
patient ids, so biopsies of one patient can fall on both sides of the split. From each slide every
512-px tile whose mask marks at least --min-tissue of it as tissue (label 1 or 2) is kept, so
neighbouring tiles exist for shifted crops and context windows.

Writes:
  $SITESHIFT_DATA/karolinska/<id>/images/<id> [d=1,x=..,y=..,w=512,h=512].jpg   (JPEG quality 95)
  $SITESHIFT_DATA/karolinska/<id>/masks/<same name>.png                          (0 / 255)
  karolinska_tiles.csv   slide, group, image, tissue_frac, tumour_frac, mask_tissue_frac, isup
  karolinska_split.json  train / val / test slide lists (never overwritten without --force)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prepare import tissue_fraction

HERE = P.INDEX
T = 512


def read_tiles(path, cells):
    """Decode the TIFF's own level-0 tiles at (row, col) cells, one at a time."""
    import tifffile
    out = {}
    with tifffile.TiffFile(path) as tf, open(path, "rb") as raw:
        page = tf.pages[0]
        if page.tilewidth != T or page.tilelength != T:
            raise ValueError(f"{path}: TIFF tiles are {page.tilewidth} px, expected {T}")
        # load the tile table first: reading these tags lazily moves tifffile's file position, so the
        # tile bytes are read through a separate handle
        offsets, counts, tables = list(page.dataoffsets), list(page.databytecounts), page.jpegtables
        across = -(-page.imagewidth // T)
        for r, c in cells:
            i = r * across + c
            raw.seek(offsets[i])
            tile, _, _ = page.decode(raw.read(counts[i]), i, jpegtables=tables)
            out[(r, c)] = tile.reshape(tile.shape[-3:])
    return out


def convert(job):
    import tifffile
    src, out, iid, isup, min_tissue, quality = job
    img_path, mask_path = src / "images" / f"{iid}.tiff", src / "masks" / f"{iid}_mask.tiff"
    m2 = tifffile.imread(mask_path, level=2)[..., 0]
    with tifffile.TiffFile(img_path) as tf:
        H, W = tf.pages[0].imagelength, tf.pages[0].imagewidth
    s = T // 16                                                    # a level-0 tile is 32 px at level 2
    h, w = min(m2.shape[0] // s, H // T), min(m2.shape[1] // s, W // T)
    b = m2[:h * s, :w * s].reshape(h, s, w, s)
    tissue = (b > 0).mean((1, 3))
    cells = [tuple(x) for x in np.argwhere(tissue >= min_tissue)]
    if not cells:
        return []
    imgs, masks = read_tiles(img_path, cells), read_tiles(mask_path, cells)
    (out / iid / "images").mkdir(parents=True, exist_ok=True)
    (out / iid / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for r, c in cells:
        name = f"{iid} [d=1,x={c * T},y={r * T},w={T},h={T}]"
        rgb, lab = imgs[(r, c)][..., :3], masks[(r, c)][..., 0]
        cancer = lab == 2
        jpg, png = out / iid / "images" / f"{name}.jpg", out / iid / "masks" / f"{name}.png"
        if not jpg.exists():
            Image.fromarray(rgb).save(jpg, quality=quality)
        if not png.exists():
            Image.fromarray((cancer * 255).astype(np.uint8)).save(png)
        rows.append(dict(slide=iid, group=iid, image=f"{iid}/images/{name}.jpg", tissue_frac=round(tissue_fraction(rgb), 4),
                         tumour_frac=float(cancer.mean()), mask_tissue_frac=round(float((lab > 0).mean()), 4), isup=int(isup)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(str(P.PANDA_RAW / "Karolinska")))
    ap.add_argument("--out", type=Path, default=Path(str(P.DATA_ROOT / "karolinska")))
    ap.add_argument("--slides", type=int, default=600)
    ap.add_argument("--min-tissue", type=float, default=0.10)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the image tiles")
    ap.add_argument("--all", action="store_true",
                    help="every slide with a mask (--slides ignored); slides of --keep-split keep their split")
    ap.add_argument("--keep-split", type=Path, help="earlier split.json whose slides keep their train / val / test role")
    ap.add_argument("--name", default="karolinska", help="output names: <name>_tiles.csv, <name>_split.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    split_json, index_csv = HERE / f"{a.name}_split.json", HERE / f"{a.name}_tiles.csv"
    if split_json.exists() and not a.force:
        raise SystemExit(f"{split_json} exists; pass --force to rebuild it")

    labels = pd.read_csv(a.src / "labels.csv")
    labels = labels[labels.has_mask == 1]
    if a.all:
        chosen = labels.reset_index(drop=True)
    else:
        share = a.slides / len(labels)
        chosen = labels.groupby("isup_grade", group_keys=False).apply(
            lambda g: g.sample(int(round(len(g) * share)), random_state=a.seed)).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    split = {}
    if a.keep_split:
        for k, ids in json.loads(a.keep_split.read_text())["splits"].items():
            split.update({i: k for i in ids})
    for _, g in chosen[~chosen.image_id.isin(list(split))].groupby("isup_grade"):
        ids = rng.permutation(g.image_id.values)
        n_train, n_val = int(round(len(ids) * a.fractions[0])), int(round(len(ids) * a.fractions[1]))
        split.update({i: "train" for i in ids[:n_train]})
        split.update({i: "val" for i in ids[n_train:n_train + n_val]})
        split.update({i: "test" for i in ids[n_train + n_val:]})
    print(f"{len(chosen)} slides: " + ", ".join(f"{k} {sum(v == k for v in split.values())}" for k in ["train", "val", "test"]), flush=True)

    jobs = [(a.src, a.out, iid, isup, a.min_tissue, a.quality) for iid, isup in zip(chosen.image_id, chosen.isup_grade)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(convert, jobs, chunksize=4) for r in part]
    df = pd.DataFrame(rows).sort_values(["slide", "image"]).reset_index(drop=True)
    df.to_csv(index_csv, index=False)
    summary = {}
    for k in ["train", "val", "test"]:
        d = df[df.slide.map(split) == k]
        summary[k] = dict(slides=int(d.slide.nunique()), tiles=len(d), tumour_tiles=int((d.tumour_frac > 0).sum()),
                          tumour_pixel_share=round(float(d.tumour_frac.mean()), 4))
    split_json.write_text(json.dumps(dict(source=str(a.src), seed=a.seed, slides=len(chosen), min_tissue=a.min_tissue,
                                          quality=a.quality, kept_split=str(a.keep_split) if a.keep_split else None,
                                          splits={k: sorted(i for i, v in split.items() if v == k and i in set(df.slide))
                                                  for k in ["train", "val", "test"]}, summary=summary), indent=1))
    print(f"{len(df)} tiles from {df.slide.nunique()} slides, {(df.tumour_frac > 0).sum()} with cancer")
    for k, v in summary.items():
        print(f"  {k:5s}: {v['slides']} slides, {v['tiles']} tiles, {v['tumour_tiles']} with cancer, {v['tumour_pixel_share']:.1%} cancer pixels")
    print(f"-> {a.out}, {index_csv}, {split_json}")


if __name__ == "__main__":
    main()
