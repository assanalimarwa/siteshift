"""
Region-style SICAPv2 labels for the label-harmonization test: every SICAPv2 mask closed with a disk of
r um over tissue and small lumens but not over glass (granularity.relabel), r = 40 and 80 um
(granularity.py: the gaps between labelled glands are filled from ~40 um on; the "closest gap fill to
ours" rule gives 60 um).

    python prepare_sicap_closed.py     # $SITESHIFT_DATA/sicapv2_r40, sicapv2_r80 (images hard-linked)

Same tiles, split and index (sicap_tiles.csv, sicap_split.json); only the masks differ, so a run with
--data $SITESHIFT_DATA/sicapv2_r<r> and otherwise the sicap_pretrain recipe isolates the labels.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data import load_mask
from granularity import SOURCES, TRAINED_RADII, relabel

HERE = P.INDEX
SRC = Path(str(P.DATA_ROOT / "sicapv2"))


def job(args):
    image, radii = args
    rgb = np.array(Image.open(SRC / image).convert("RGB"))
    name = Path(image).stem + ".png"
    mask = load_mask(SRC / "masks" / name).astype(bool)
    out = []
    for r in radii:
        dst = Path(str(P.DATA_ROOT / f"sicapv2_r{r}"))
        new = relabel(mask, rgb, r, SOURCES["SICAPv2"]["um"])
        Image.fromarray(new.astype(np.uint8) * 255).save(dst / "masks" / name)
        link = dst / image
        if not link.exists():
            os.link(SRC / image, link)
        out.append((r, int(mask.sum()), int(new.sum())))
    return out


def main():
    rows = pd.read_csv(HERE / "sicap_tiles.csv")
    for r in TRAINED_RADII:
        for sub in ["images", "masks"]:
            (Path(str(P.DATA_ROOT / f"sicapv2_r{r}")) / sub).mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        res = [x for out in pool.imap_unordered(job, [(i, TRAINED_RADII) for i in rows.image], chunksize=32) for x in out]
    for r in TRAINED_RADII:
        before = sum(b for rr, b, _ in res if rr == r)
        after = sum(a for rr, _, a in res if rr == r)
        n = len(list(Path(str(P.DATA_ROOT / f"sicapv2_r{r}/masks")).glob("*.png")))
        print(f"r = {r} um: {n} masks, tumour pixels {before:,} -> {after:,} (x{after / before:.3f})")


if __name__ == "__main__":
    main()
