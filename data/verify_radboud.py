"""
Check the Radboud tile set against the PANDA source (fmgrid env):

    python verify_radboud.py [--tiles 300]

Every listed tile and mask exists; for random tiles the mask equals the label mapping of the source mask
TIFF tile (3-5 -> 255, 1-2 -> 0, 0 -> IGNORE) and the image is close to the source tile (JPEG quality 85).
Exits with an error if anything is off.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data import IGNORE, mask_path
from prepare_karolinska import T, read_tiles

HERE = P.INDEX
SRC = Path(str(P.PANDA_RAW / "Panda"))
ROOT = Path(str(P.DATA_ROOT / "radboud"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=300)
    a = ap.parse_args()
    t = pd.read_csv(HERE / "radboud_tiles.csv")
    split = json.loads((HERE / "radboud_split.json").read_text())["splits"]
    missing = [p for p in t.image if not (ROOT / p).exists() or not (ROOT / mask_path(p)).exists()]
    print(f"{len(t)} tiles, {t.slide.nunique()} slides (train {len(split['train'])}, val {len(split['val'])}); "
          f"missing image or mask: {len(missing)}")
    bad, psnr = 0, []
    for p in t.sample(a.tiles, random_state=0).image:
        iid = p.split("/")[0]
        x, y = map(int, re.search(r"x=(\d+),y=(\d+)", p).groups())
        cell = (y // T, x // T)
        lab = read_tiles(SRC / "masks" / f"{iid}_mask.tiff", [cell])[cell]
        lab = lab[..., 0] if lab.ndim == 3 else lab
        want = np.where(lab >= 3, 255, np.where(lab == 0, IGNORE, 0))
        got = np.array(Image.open(ROOT / mask_path(p)))
        bad += int(not np.array_equal(got, want))
        src = read_tiles(SRC / "images" / f"{iid}.tiff", [cell])[cell][..., :3].astype(float)
        mse = ((np.array(Image.open(ROOT / p).convert("RGB")).astype(float) - src) ** 2).mean()
        psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    print(f"{a.tiles} random tiles: masks differing from the source mapping {bad}; image PSNR vs source "
          f"median {np.median(psnr):.1f} dB (min {np.min(psnr):.1f})")
    print(f"tumour pixels {t.tumour_frac.mean():.1%}, ignored {t.ignore_frac.mean():.1%}, tiles with tumour "
          f"{(t.tumour_frac > 0).mean():.1%}")
    if missing or bad or np.min(psnr) < 25:
        raise SystemExit("Radboud tile set FAILED verification")
    print("OK")


if __name__ == "__main__":
    main()
