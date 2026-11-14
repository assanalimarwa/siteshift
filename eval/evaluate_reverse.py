"""
Score one checkpoint on SICAPv2 tiles for the reverse-direction experiments (run_reverse.py), with
the counts reverse_report.py needs.

    python evaluate_reverse.py --ckpt X/best.pt --out X/eval_reverse_test --split-file cv/sicap_fold0.json --split test
    python evaluate_reverse.py --ckpt X/best.pt --out X/eval_reverse_all --split all       # every SICAPv2 tile

Style-tolerant counts (tp_st, fp_st, fn_st): the same, ignoring every pixel within 50 um of an
annotated tumour boundary (on either side), so a coarser or tighter outlining style is not penalised.

Tiles are enlarged x1.75 to our pixel size (896 px, as in every SICAPv2 evaluation) and each is
predicted alone at threshold 0.5. Per tile (per_tile.csv): tp, fp, fn, tumour_px, the tumour-free
tissue (benign_tissue_px) and false tumour on it (fp_tissue), both split into "near" (within
NEAR_UM = 50 um of annotated tumour in the same tile) and "far", and the predicted tumour share of
the tile (score, for the tile AUC). SICAPv2 is taken as 1 um/px, so 50 um = 87.5 px at x1.75.
Tumour in a neighbouring tile is not seen, so false tumour at a tile edge next to tumour counts as far.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from data import TileDataset, load_rows, to_input
from evaluate import tumour_probability
from metrics import tissue_mask
from model import build_model

HERE = P.INDEX
ZOOM, NEAR_UM, UM_PER_PX = (1.5, 2.0), 50.0, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="guard", help="a split of --split-file, or 'all' for every SICAPv2 tile")
    ap.add_argument("--split-file", type=Path, default=HERE / "sicap_split.json")
    ap.add_argument("--limit", type=int, default=0, help="first N tiles only (tests)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["arch"], pretrained=False).to(dev, memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()

    if a.split == "all":
        rows = pd.read_csv(HERE / "sicap_tiles.csv")
    else:
        rows = load_rows(HERE / "sicap_tiles.csv", a.split_file, a.split)
    if a.limit:
        rows = rows.iloc[:a.limit].reset_index(drop=True)
    ds = TileDataset(str(P.DATA_ROOT / "sicapv2"), rows, zoom=ZOOM)
    near_px = NEAR_UM / UM_PER_PX * ds.size / ds.T                     # 50 um at the evaluation scale
    out = []
    with torch.no_grad():
        for img, mask, idx in DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True):
            img = img.to(dev, non_blocking=True)
            pred = (tumour_probability(model, to_input(img), tta=False) > 0.5).cpu().numpy()[:, 0]
            truth = mask.numpy()[:, 0] > 0
            tissue = tissue_mask(img).cpu().numpy()[:, 0]
            for j, i in enumerate(idx.tolist()):
                p, t, s = pred[j], truth[j], tissue[j]
                if t.any():
                    # edt(x) = distance of each True pixel of x to the nearest False pixel
                    near = ~t & (ndi.distance_transform_edt(~t) <= near_px)               # outside, close to tumour
                    band = near | (t & (ndi.distance_transform_edt(t) <= near_px))       # either side of the boundary
                else:
                    near = band = np.zeros_like(t)
                keep = ~band
                benign = s & ~t
                out.append(dict(slide=rows.slide[i], image=rows.image[i],
                                tp=int((p & t).sum()), fp=int((p & ~t).sum()), fn=int((~p & t).sum()), tumour_px=int(t.sum()),
                                fp_tissue=int((p & benign).sum()), benign_tissue_px=int(benign.sum()),
                                fp_near=int((p & benign & near).sum()), benign_near_px=int((benign & near).sum()),
                                fp_far=int((p & benign & ~near).sum()), benign_far_px=int((benign & ~near).sum()),
                                tp_st=int((p & t & keep).sum()), fp_st=int((p & ~t & keep).sum()),
                                fn_st=int((~p & t & keep).sum()), tumour_st=int((t & keep).sum()),
                                score=float(p.mean())))
    t = pd.DataFrame(out)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "per_tile.csv", index=False)
    s = t.drop(columns=["slide", "image", "score"]).sum()
    summary = dict(checkpoint=str(a.ckpt), arch=ck["arch"], epoch=ck.get("epoch"), split=a.split, split_file=str(a.split_file),
                   tiles=len(t), size=ds.size, near_px=near_px,
                   dice=float(2 * s.tp / max(2 * s.tp + s.fp + s.fn, 1)), sensitivity=float(s.tp / max(s.tumour_px, 1)),
                   dice_style_tolerant=float(2 * s.tp_st / max(2 * s.tp_st + s.fp_st + s.fn_st, 1)),
                   false_tumour_rate=float(s.fp_tissue / max(s.benign_tissue_px, 1)),
                   false_tumour_near=float(s.fp_near / max(s.benign_near_px, 1)), false_tumour_far=float(s.fp_far / max(s.benign_far_px, 1)))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
