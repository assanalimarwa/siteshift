"""
Index the tiles and split the slide folders into train / val / test.

    python prepare.py                         # default: $SITESHIFT_DATA/dataset47
    python prepare.py --group slide           # split every slide folder on its own

Writes next to this script:
    tiles.csv   one row per tile: slide, group, image (relative to --data), tissue_frac,
                tumour_frac. Reused on later runs unless --reindex.
    split.json  which slide folders are train, val and test. Never overwritten without --force,
                so the test set cannot change silently.

Whole slide folders go to one split, never single tiles: tiles of one slide share stain,
scanner and patient, and splitting them would leak that into the test score. With the
default --group patient, slides whose names share the number before the first dot
(10.1, 10.2, 10.3) also stay together, since that number is most likely the patient.

The split is a seeded random search for the assignment whose tile share and tumour-pixel
share come closest to --fractions, with at least one small slide group (needle biopsies,
< 1000 tiles per slide) and one large one in both val and test.
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

HERE = P.INDEX
SPLITS = ["train", "val", "test"]
SMALL_SLIDE = 1000


def tissue_mask(rgb):
    """Pixels that are tissue: neither glass (bright) nor padding (dark), and coloured."""
    a = rgb.astype(np.int16)
    mean = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    return (mean > 25) & (mean < 220) & (sat > 12)


def tissue_fraction(rgb):
    return float(tissue_mask(rgb).mean())


def measure(job):
    root, slide, name = job
    img = np.array(Image.open(root / slide / "images" / name).convert("RGB"))
    raw = np.array(Image.open(root / slide / "masks" / name))
    return dict(slide=slide, group=slide.split(".")[0], image=f"{slide}/images/{name}",
                tissue_frac=round(tissue_fraction(img), 4),
                tumour_frac=float((raw > 0).mean()),          # unrounded: a few pixels must stay > 0
                unexpected=sorted(set(np.unique(raw).tolist()) - {0, 255}))


def build_index(data, out_csv, workers):
    jobs = [(data, s.name, f.name) for s in sorted(data.iterdir()) if (s / "images").is_dir()
            for f in sorted((s / "images").glob("*.png"))]
    missing = [f"{s}/masks/{n}" for _, s, n in jobs if not (data / s / "masks" / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} image(s) without a mask, e.g. {missing[0]}")
    print(f"measuring {len(jobs)} tiles in {data} ...", flush=True)
    with Pool(workers) as pool:
        df = pd.DataFrame(pool.map(measure, jobs, chunksize=64))
    odd = df[df.unexpected.str.len() > 0]
    if len(odd):
        raise SystemExit(f"{len(odd)} mask(s) have values other than 0/255, e.g. {odd.image.iloc[0]}: "
                         f"{odd.unexpected.iloc[0]}")
    df = df.drop(columns="unexpected")
    df.to_csv(out_csv, index=False)
    return df


def search_split(groups, fractions, seed, trials):
    f = np.array([fractions[k] for k in SPLITS])
    tiles, tumour, small = groups.tiles.values, groups.tumour.values, groups.small.values
    rng = np.random.default_rng(seed)
    onehot = rng.choice(len(SPLITS), size=(trials, len(groups)), p=f)[..., None] == np.arange(len(SPLITS))
    tile_share = (onehot * tiles[None, :, None]).sum(1) / tiles.sum()
    tumour_share = (onehot * tumour[None, :, None]).sum(1) / tumour.sum()
    score = np.abs(tile_share - f).sum(1) + np.abs(tumour_share - f).sum(1)
    has_small = (onehot & small[None, :, None]).any(1)[:, 1:].all(1)
    has_large = (onehot & ~small[None, :, None]).any(1)[:, 1:].all(1)
    score[~(has_small & has_large)] = np.inf
    best = int(np.argmin(score))
    if not np.isfinite(score[best]):
        raise SystemExit("no split satisfies the constraints; raise --trials or change --fractions")
    return onehot[best].argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(str(P.DATASET47)))
    ap.add_argument("--group", choices=["patient", "slide"], default="patient",
                    help="patient: keep slides with the same number before the dot together")
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reindex", action="store_true", help="re-measure tiles even if tiles.csv exists")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split.json")
    ap.add_argument("--index", type=Path, default=HERE / "tiles.csv", help="tile index to write / reuse")
    ap.add_argument("--split", type=Path, default=HERE / "split.json")
    ap.add_argument("--index-only", action="store_true", help="measure the tiles, make no split (for CV: make_folds.py)")
    a = ap.parse_args()

    index_csv, split_json = a.index, a.split
    if index_csv.exists() and not a.reindex:
        df = pd.read_csv(index_csv)
        print(f"using {index_csv} ({len(df)} tiles)")
    else:
        df = build_index(a.data, index_csv, a.workers)
    print(f"{len(df)} tiles, {df.slide.nunique()} slides, {(df.tumour_frac > 0).sum()} with tumour, "
          f"{df.tumour_frac.mean():.1%} of pixels tumour")
    if a.index_only:
        return

    if split_json.exists() and not a.force:
        raise SystemExit(f"{split_json} exists; pass --force to replace it (this changes the test set)")

    key = "group" if a.group == "patient" else "slide"
    g = df.groupby(key).agg(tiles=("image", "size"), tumour=("tumour_frac", "sum"), n_slides=("slide", "nunique"))
    g["small"] = g.tiles / g.n_slides < SMALL_SLIDE
    fractions = dict(zip(SPLITS, (np.array(a.fractions) / sum(a.fractions)).tolist()))
    assign = search_split(g, fractions, a.seed, a.trials)
    slide_split = {s: SPLITS[assign[g.index.get_loc(k)]] for s, k in df.groupby("slide")[key].first().items()}

    df["split"] = df.slide.map(slide_split)
    summary = {}
    for k in SPLITS:
        d = df[df.split == k]
        summary[k] = dict(slides=sorted(d.slide.unique(), key=lambda s: [int(n) for n in s.split(".")[:2]]),
                          tiles=len(d), tumour_tiles=int((d.tumour_frac > 0).sum()),
                          tile_share=round(len(d) / len(df), 4),
                          tumour_pixel_share=round(d.tumour_frac.sum() / df.tumour_frac.sum(), 4))
    split = dict(seed=a.seed, group_by=a.group, fractions=fractions,
                 splits={k: summary[k]["slides"] for k in SPLITS}, summary=summary)
    split_json.write_text(json.dumps(split, indent=2))

    print(f"\nsplit by {a.group} -> {split_json}")
    for k in SPLITS:
        s = summary[k]
        print(f"{k:5}  {len(s['slides']):2d} slides  {s['tiles']:6d} tiles ({s['tile_share']:.1%})  "
              f"{s['tumour_tiles']:5d} tumour tiles  {s['tumour_pixel_share']:.1%} of tumour pixels")
        print("       " + ", ".join(x.replace(".bif - Series 0", "") for x in s["slides"]))


if __name__ == "__main__":
    main()
