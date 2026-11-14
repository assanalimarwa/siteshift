"""
Check the Karolinska tiles (prepare_karolinska.py) against the original TIFFs, slide by slide.

    python verify_karolinska.py           # fmgrid env; cv/karolinska_verify.csv + printed summary

Per slide:
  completeness  tiles expected (level-2 mask: >= min_tissue tissue, full 512-px tiles) vs tiles in the
                index vs image and mask files on disk (missing or extra files)
  coverage      share of the slide's tissue and cancer pixels (level-2 mask) inside the kept tiles;
                the rest lies in tiles below min_tissue or in the partial strip at the right / bottom edge
  positions     the slide rebuilt from its tiles, shrunk 16x, against the TIFF's own level-2 image
                (mean absolute difference over kept tiles; a misplaced tile would stand out)
  alignment     each tile's cancer share vs the cancer share of its level-2 mask block
  masks         only 0 / 255 values
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

HERE = P.INDEX
SRC = Path(str(P.PANDA_RAW / "Karolinska"))
OUT = Path(str(P.DATA_ROOT / "karolinska"))       # main() may change it (--out)
T, S = 512, 32                         # a 512-px level-0 tile is 32 px at level 2


def check(job):
    import tifffile
    iid, rows, min_tissue, out = job
    global OUT
    OUT = out
    m2 = tifffile.imread(SRC / "masks" / f"{iid}_mask.tiff", level=2)[..., 0]
    img2 = tifffile.imread(SRC / "images" / f"{iid}.tiff", level=2)[..., :3]
    with tifffile.TiffFile(SRC / "images" / f"{iid}.tiff") as tf:
        H, W = tf.pages[0].imagelength, tf.pages[0].imagewidth
    h, w = min(m2.shape[0] // S, H // T), min(m2.shape[1] // S, W // T)
    blocks = m2[:h * S, :w * S].reshape(h, S, w, S)
    tissue, cancer = (blocks > 0).mean((1, 3)), (blocks == 2).mean((1, 3))
    expected = {(int(r), int(c)) for r, c in np.argwhere(tissue >= min_tissue)}
    indexed = {(int(y) // T, int(x) // T) for x, y in rows[["x", "y"]].values}
    img_files = {p.stem for p in (OUT / iid / "images").glob("*.jpg")}
    mask_files = {p.stem for p in (OUT / iid / "masks").glob("*.png")}
    names = set(rows.name_stem)
    kept = np.zeros((h, w), bool)
    for r, c in indexed & expected:
        kept[r, c] = True
    total_tissue, total_cancer = (m2 > 0).sum(), (m2 == 2).sum()
    in_tissue = ((blocks > 0).sum((1, 3)) * kept).sum()
    in_cancer = ((blocks == 2).sum((1, 3)) * kept).sum()
    # positions and alignment: rebuild the kept tiles at level-2 size and compare
    mad, align, bad_masks = [], [], 0
    for (r, c), stem in zip([(int(y) // T, int(x) // T) for x, y in rows[["x", "y"]].values], rows.name_stem):
        tile = np.array(Image.open(OUT / iid / "images" / f"{stem}.jpg").convert("RGB").resize((S, S), Image.BOX)).astype(np.int16)
        ref = img2[r * S:(r + 1) * S, c * S:(c + 1) * S].astype(np.int16)
        if ref.shape == tile.shape:
            mad.append(np.abs(tile - ref).mean())
        m = np.array(Image.open(OUT / iid / "masks" / f"{stem}.png"))
        bad_masks += bool(set(np.unique(m).tolist()) - {0, 255})
        align.append(((m > 0).mean(), cancer[r, c]))
    align = np.array(align)
    return dict(slide=iid, expected=len(expected), indexed=len(indexed), missing=len(expected - indexed), extra=len(indexed - expected),
                missing_files=len(names - img_files) + len(names - mask_files), extra_files=len(img_files - names) + len(mask_files - names),
                tissue_covered=in_tissue / max(total_tissue, 1), cancer_covered=in_cancer / total_cancer if total_cancer else np.nan,
                cancer_px_level2=int(total_cancer), position_mad=float(np.mean(mad)) if mad else np.nan,
                position_mad_max=float(np.max(mad)) if mad else np.nan,
                align_corr=float(np.corrcoef(align[:, 0], align[:, 1])[0, 1]) if len(align) > 2 and align[:, 1].std() > 0 else np.nan,
                align_max_diff=float(np.abs(align[:, 0] - align[:, 1]).max()) if len(align) else np.nan, bad_masks=bad_masks)


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="karolinska", help="<name>_tiles.csv and <name>_split.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT = a.out
    split = json.loads((HERE / f"{a.name}_split.json").read_text())
    t = pd.read_csv(HERE / f"{a.name}_tiles.csv")
    xy = t.image.str.extract(r"\[d=1,x=(\d+),y=(\d+),w=512,h=512\]").astype(int)
    t["x"], t["y"] = xy[0], xy[1]
    t["name_stem"] = t.image.str.split("/").str[-1].str.replace(".jpg", "", regex=False)
    jobs = [(iid, g, split["min_tissue"], OUT) for iid, g in t.groupby("slide")]
    with Pool(a.workers) as pool:
        r = pd.DataFrame(pool.map(check, jobs, chunksize=4))
    r.to_csv(HERE / "cv" / f"{a.name}_verify.csv", index=False)

    sets = {k: set(v) for k, v in split["splits"].items()}
    overlap = sum(len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
    labels = pd.read_csv(SRC / "labels.csv").set_index("image_id")
    print(f"slides checked: {len(r)} (index {t.slide.nunique()}, split lists {sum(len(v) for v in sets.values())}); slides in two splits: {overlap}")
    print(f"completeness: expected tiles {r.expected.sum()}, in index {r.indexed.sum()}, missing {r.missing.sum()}, extra {r.extra.sum()}, "
          f"missing files {r.missing_files.sum()}, extra files {r.extra_files.sum()}, duplicate index rows {t.image.duplicated().sum()}")
    print(f"coverage of tissue: mean {r.tissue_covered.mean():.4f}, min {r.tissue_covered.min():.4f}; "
          f"coverage of cancer: mean {r.cancer_covered.mean():.4f}, min {r.cancer_covered.min():.4f}, "
          f"pooled {(r.cancer_covered * r.cancer_px_level2).sum() / r.cancer_px_level2.sum():.4f}")
    low = r[r.cancer_covered < 0.95]
    print(f"slides with < 95% of their cancer kept: {len(low)}" + (f" (lowest {low.cancer_covered.min():.3f}, cancer pixels {low.cancer_px_level2.tolist()[:8]})" if len(low) else ""))
    print(f"positions: mean abs. difference rebuilt vs TIFF thumbnail {r.position_mad.mean():.2f} grey levels (worst slide mean {r.position_mad.max():.2f}, "
          f"worst single tile {r.position_mad_max.max():.2f})")
    print(f"alignment: tile cancer share vs mask block, correlation mean {r.align_corr.mean():.4f} (min {r.align_corr.min():.4f}), "
          f"max difference {r.align_max_diff.max():.4f}; masks with values other than 0/255: {r.bad_masks.sum()}")
    for k, v in sets.items():
        print(f"  {k:5s}: ISUP " + str(labels.loc[sorted(v), "isup_grade"].value_counts().sort_index().to_dict()))
    print(f"-> {HERE / 'cv' / (a.name + '_verify.csv')}")


if __name__ == "__main__":
    main()
