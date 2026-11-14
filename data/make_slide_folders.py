"""
Copy the QuPath tile export into one folder per slide, each with images/ and masks/.

    python make_slide_folders.py --export <QuPath export folder> --out $SITESHIFT_DATA/dataset47

    dataset/36.4.bif - Series 0/images/36.4.bif - Series 0 [d=2,x=...,y=...,w=1024,h=1024].png
    dataset/36.4.bif - Series 0/masks/36.4.bif - Series 0 [d=2,x=...,y=...,w=1024,h=1024].png

A mask has the same file name as its image ("-labelled" is dropped). Files are copied byte for
byte: images stay RGB 512x512, masks stay palette PNGs with 0 = not tumour, 255 = tumour.
Only the tiles directly inside each slide folder are taken; 26.3 also holds data/ and
stitched/ with copies of its tiles, which would otherwise be counted twice.

The number before the first dot of a slide is the patient (36.4 and 36.5 are both patient 36),
so when splitting, keep slides of one patient on the same side.

The export is never modified. Re-running skips files already copied with the same size.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TILE = re.compile(r"^(?P<slide>\d+\.\d+\.bif - Series 0) \[d=\d+,x=\d+,y=\d+,w=\d+,h=\d+\]\.png$")
MASK_SUFFIX = "-labelled.png"


def collect(export: Path):
    """(image, mask, slide) for every tile; fails loudly on anything unexpected."""
    pairs, problems = [], []
    for slide_dir in sorted(p for p in export.iterdir() if p.is_dir() and p.name.endswith(".bif - Series 0")):
        names = {f.name for f in slide_dir.iterdir() if f.is_file() and f.suffix == ".png"}
        masks = {n for n in names if n.endswith(MASK_SUFFIX)}
        images = names - masks
        for img in sorted(images):
            m = TILE.match(img)
            if not m or m["slide"] != slide_dir.name:
                problems.append(f"unexpected name: {slide_dir.name}/{img}")
                continue
            mask = img[:-4] + MASK_SUFFIX
            if mask not in masks:
                problems.append(f"no mask for: {slide_dir.name}/{img}")
                continue
            pairs.append((slide_dir / img, slide_dir / mask, slide_dir.name))
        for mask in masks - {i[:-4] + MASK_SUFFIX for i in images}:
            problems.append(f"no image for: {slide_dir.name}/{mask}")
    return pairs, problems


def copy(src: Path, dst: Path, link=False):
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=Path(str(P.DATA_ROOT / "export")))
    ap.add_argument("--out", type=Path, default=Path(str(P.DATASET47)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--link", action="store_true",
                    help="hard-link instead of copying (same drive only; no extra space, the export is still never modified)")
    a = ap.parse_args()

    pairs, problems = collect(a.export)
    if problems:
        raise SystemExit("\n".join(problems[:20]) + f"\n{len(problems)} problem(s), nothing copied")
    print(f"{len(pairs)} image/mask pairs from {a.export}", flush=True)

    jobs = []
    for img, mask, slide in pairs:
        dest = a.out / slide
        (dest / "images").mkdir(parents=True, exist_ok=True)
        (dest / "masks").mkdir(parents=True, exist_ok=True)
        jobs += [(img, dest / "images" / img.name), (mask, dest / "masks" / img.name)]

    with ThreadPoolExecutor(a.workers) as pool:
        copied = sum(pool.map(lambda j: copy(*j, link=a.link), jobs, chunksize=64))
    print(f"copied {copied} files, {len(jobs) - copied} already present")

    per_slide = Counter(s for _, _, s in pairs)
    for slide in sorted(per_slide, key=lambda s: [int(n) for n in s.split(".")[:2]]):
        d = a.out / slide
        n_img, n_msk = len(list((d / "images").glob("*.png"))), len(list((d / "masks").glob("*.png")))
        flag = "" if n_img == n_msk == per_slide[slide] else "  <-- MISMATCH"
        print(f"{slide}: {n_img:6d} images  {n_msk:6d} masks{flag}")
    print(f"{len(per_slide)} slide folders")


if __name__ == "__main__":
    main()
