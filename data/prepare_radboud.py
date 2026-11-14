"""
Convert PANDA Radboud biopsies into the tile layout train.py reads (fmgrid env: tifffile + imagecodecs).

    python prepare_radboud.py --measure 40                  # tiles and MB per slide on 40 random slides, no split written
    python prepare_radboud.py --train 2000 --val 300        # the set used by run_all_public.py

Source: <src>/images/<id>.tiff and masks/<id>_mask.tiff (PANDA, Radboud provider, Panda/ folder), labels.csv.
Level 0 is ~0.486 um/px (TIFF tags), close to our 0.5, so tiles are taken as they are: the TIFF's own
512 x 512 tiles. Mask red channel: 0 background / unknown, 1 stroma, 2 benign epithelium, 3-5 Gleason 3-5.
Label mapping (user's spec 2026-10-28): Gleason 3-5 -> tumour (255), stroma and benign epithelium -> not
tumour (0), background / unknown -> ignored in training (IGNORE = 128, see data.load_mask). No closing of
the gland-level cancer outlines.

Split: every Radboud slide with a mask is assigned train / val / test (70 / 10 / 20 within each ISUP grade,
seed 0) and recorded in radboud_split.json; only random subsets of --train train and --val val slides are
converted (disk: all slides would need ~45 GB), test slides are left out (we never test on Radboud).
From each converted slide every 512-px tile whose level-2 mask marks at least --min-tissue of it as labelled
(1-5) is kept, as for Karolinska.

Writes $SITESHIFT_DATA/radboud/<id>/images|masks/<id> [d=1,x=..,y=..,w=512,h=512].jpg|png (JPEG quality 85),
radboud_tiles.csv (slide, group, image, tissue_frac, tumour_frac, ignore_frac, mask_tissue_frac, isup),
radboud_split.json (converted slides only in "splits", the full assignment in "all_slides").
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data import IGNORE
from prepare import tissue_fraction
from prepare_karolinska import T, read_tiles

HERE = P.INDEX


def convert(job):
    import tifffile
    src, out, iid, isup, min_tissue, quality = job
    img_path, mask_path = src / "images" / f"{iid}.tiff", src / "masks" / f"{iid}_mask.tiff"
    m2 = tifffile.imread(mask_path, level=2)
    m2 = m2[..., 0] if m2.ndim == 3 else m2
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
        rgb, lab = imgs[(r, c)][..., :3], masks[(r, c)]
        lab = lab[..., 0] if lab.ndim == 3 else lab
        cancer, unknown = lab >= 3, lab == 0
        target = np.where(cancer, 255, np.where(unknown, IGNORE, 0)).astype(np.uint8)
        jpg, png = out / iid / "images" / f"{name}.jpg", out / iid / "masks" / f"{name}.png"
        if not jpg.exists():
            Image.fromarray(rgb).save(jpg, quality=quality)
        if not png.exists():
            Image.fromarray(target).save(png)
        rows.append(dict(slide=iid, group=iid, image=f"{iid}/images/{name}.jpg", tissue_frac=round(tissue_fraction(rgb), 4),
                         tumour_frac=float(cancer.mean()), ignore_frac=round(float(unknown.mean()), 4),
                         mask_tissue_frac=round(float((lab > 0).mean()), 4), isup=int(isup)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(str(P.PANDA_RAW / "Panda")))
    ap.add_argument("--out", type=Path, default=Path(str(P.DATA_ROOT / "radboud")))
    ap.add_argument("--train", type=int, default=2000, help="train slides converted (random subset of the train split)")
    ap.add_argument("--val", type=int, default=300, help="val slides converted")
    ap.add_argument("--train-take", type=int, default=0,
                    help="keep only the first N of the --train slides (ids sorted; PANDA ids are hashes, so still a random "
                         "subset). 2026-10-28: the 2,500-slide conversion was stopped at 1,839 for disk space -> 1,800")
    ap.add_argument("--measure", type=int, default=0, help="only convert this many random slides into a scratch folder and report size")
    ap.add_argument("--min-tissue", type=float, default=0.10)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality of the image tiles (as karolinska_all)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    labels = pd.read_csv(a.src / "labels.csv")
    labels = labels[labels.has_mask == 1].reset_index(drop=True)
    if a.measure:
        pick = labels.sample(a.measure, random_state=a.seed)
        out = a.out.parent / "_radboud_measure"
        t0 = time.time()
        with Pool(a.workers) as pool:
            rows = [r for part in pool.map(convert, [(a.src, out, i, g, a.min_tissue, a.quality)
                                                     for i, g in zip(pick.image_id, pick.isup_grade)]) for r in part]
        mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
        df = pd.DataFrame(rows)
        print(f"{a.measure} slides: {len(df)} tiles ({len(df) / a.measure:.0f} per slide), {mb:.0f} MB "
              f"({mb / a.measure:.1f} MB per slide), {(time.time() - t0) / a.measure:.2f} s per slide; "
              f"tumour pixels {df.tumour_frac.mean():.1%}, ignored {df.ignore_frac.mean():.1%} -> {out}")
        return

    split_json, index_csv = HERE / "radboud_split.json", HERE / "radboud_tiles.csv"
    if split_json.exists() and not a.force:
        raise SystemExit(f"{split_json} exists; pass --force to rebuild it")
    rng = np.random.default_rng(a.seed)
    split = {}
    for _, g in labels.groupby("isup_grade"):
        ids = rng.permutation(g.image_id.values)
        n_train, n_val = int(round(len(ids) * a.fractions[0])), int(round(len(ids) * a.fractions[1]))
        split.update({i: "train" for i in ids[:n_train]})
        split.update({i: "val" for i in ids[n_train:n_train + n_val]})
        split.update({i: "test" for i in ids[n_train + n_val:]})
    chosen = {k: sorted(rng.permutation(sorted(i for i, v in split.items() if v == k))[:n].tolist())
              for k, n in [("train", a.train), ("val", a.val)]}
    if a.train_take:
        chosen["train"] = chosen["train"][:a.train_take]
    isup =labels.set_index("image_id").isup_grade
    jobs = [(a.src, a.out, i, int(isup[i]), a.min_tissue, a.quality) for k in chosen for i in chosen[k]]
    print(f"converting {len(chosen['train'])} train + {len(chosen['val'])} val of {len(labels)} Radboud slides", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(convert, jobs, chunksize=4) for r in part]
    df = pd.DataFrame(rows).sort_values(["slide", "image"]).reset_index(drop=True)
    df.to_csv(index_csv, index=False)
    kept = set(df.slide)
    summary = {}
    for k in ["train", "val"]:
        d = df[df.slide.isin(chosen[k])]
        summary[k] = dict(slides=int(d.slide.nunique()), tiles=len(d), tumour_tiles=int((d.tumour_frac > 0).sum()),
                          tumour_pixel_share=round(float(d.tumour_frac.mean()), 4), ignored_pixel_share=round(float(d.ignore_frac.mean()), 4))
    split_json.write_text(json.dumps(dict(source=str(a.src), seed=a.seed, min_tissue=a.min_tissue, quality=a.quality,
                                          mapping="3-5 -> 255 tumour; 1-2 -> 0; 0 -> 128 ignore",
                                          splits={k: [i for i in chosen[k] if i in kept] for k in chosen} | {"test": []},
                                          all_slides={k: sorted(i for i, v in split.items() if v == k) for k in ["train", "val", "test"]},
                                          summary=summary), indent=1))
    mb = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file()) / 2 ** 20
    print(f"{len(df)} tiles from {len(kept)} slides, {(df.tumour_frac > 0).sum()} with cancer, {mb / 1024:.1f} GB")
    for k, v in summary.items():
        print(f"  {k:5s}: {v}")
    print(f"-> {a.out}, {index_csv}, {split_json}")


if __name__ == "__main__":
    main()
