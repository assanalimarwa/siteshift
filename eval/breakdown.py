"""
Where the wrong pixels are: each false-tumour and missed-tumour pixel of a tile is put in one place
class, from the image brightness and the distance to the annotated outline (a stain-based epithelium /
stroma split was unreliable on our slides, see granularity.py):

  glass   bright pixel (optical density below stain.BETA in every channel): glass, lumen, fat
  near    tissue within 50 um of the annotated outline: false tumour just outside a labelled region,
          missed tumour along its edge - where outlining conventions differ
  far     tissue farther than 50 um from the outline: false tumour regions away from any label, missed
          tumour deep inside a labelled region
Distances use square (Chebyshev) neighbourhoods inside the tile; the tile border is not an outline.
evaluate.py --breakdown and evaluate_sicap.py --breakdown add COLUMNS to per_tile.csv: the false (fp_*)
and missed (fn_*) pixels per class, and all non-tumour (benign_*) and tumour (tumour_*) pixels per class.
"""

import torch
import torch.nn.functional as F

from stain import BETA

BAND_UM = 50
CLASSES = ["glass", "near", "far"]
COLUMNS = [f"{k}_{c}" for k in ["fp", "fn", "benign", "tumour"] for c in CLASSES]


def _dilate(x, r):
    """Square dilation of a bool (B,1,H,W) by r pixels; outside the tile counts as False."""
    k = 2 * r + 1
    y = F.max_pool2d(F.pad(x.float(), (r, r, 0, 0)), (1, k), 1)
    return F.max_pool2d(F.pad(y, (0, 0, r, r)), (k, 1), 1) > 0


def counts(img, truth, pred, um_per_px):
    """img uint8 (B,3,H,W); truth, pred bool (B,1,H,W) -> int64 (B, len(COLUMNS)) in COLUMNS order."""
    r = round(BAND_UM / um_per_px)
    glass = (-torch.log((img.float() + 1) / 256) < BETA).all(1, keepdim=True)
    near_tumour = _dilate(truth, r)            # for non-tumour pixels: within r of tumour
    near_benign = _dilate(~truth, r)           # for tumour pixels: within r of non-tumour
    out = []
    for sel, near in [(pred & ~truth, near_tumour), (~pred & truth, near_benign), (~truth, near_tumour), (truth, near_benign)]:
        out += [sel & glass, sel & ~glass & near, sel & ~glass & ~near]
    return torch.stack([m.sum((1, 2, 3)) for m in out], 1)
