"""
Save one model's thresholded predictions for the detection-delineation protocol (protocol.py), so the
region analysis can be redone without the GPU:

    python protocol_predict.py --ckpt runs/X/best.pt --out $SITESHIFT_PRED/X/ours_tile_all_test \
        --split-file cv/all.json --context 0 --no-tta

Inference exactly as evaluate.py (same windows, Hann blending of the context windows, flips, threshold 0.5);
--zoom 1.5 2.0 enlarges SICAPv2 tiles x1.75 as evaluate_reverse.py does; --split all takes every tile of
--index. Per slide <out>/<slide>.npz: image (tile paths), pred / truth / tissue (np.packbits of each S x S
mask, one row per tile; tissue as metrics.tissue_mask), size S. <out>/done.json is written last.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import TileDataset, grid_position, load_rows, to_input
from evaluate import Blender, tumour_probability
from metrics import tissue_mask
from model import build_model

HERE = P.INDEX


def slide_file(out, slide):
    return out / (re.sub(r"[^\w.-]+", "_", slide) + ".npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(str(P.DATASET47)))
    ap.add_argument("--index", type=Path, default=HERE / "tiles.csv")
    ap.add_argument("--split-file", type=Path, default=HERE / "cv" / "all.json")
    ap.add_argument("--split", default="test", help="a split of --split-file, or 'all' for every tile of --index")
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--zoom", type=float, nargs=2, help="SICAPv2: 1.5 2.0 (tiles enlarged x1.75)")
    ap.add_argument("--slides", nargs="+", help="only these slides (tests)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if (a.out / "done.json").exists():
        print(f"{a.out} already done")
        return

    t0 = time.time()
    dev = torch.device(a.device)
    if a.device == "cuda" and os.environ.get("QP_GPU_MEM_GB"):      # shared GPU: never take more than this (run_queue.py sets it)
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["QP_GPU_MEM_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["arch"], pretrained=False).to(dev, memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = pd.read_csv(a.index) if a.split == "all" else load_rows(a.index, a.split_file, a.split)
    if a.slides:
        rows = rows[rows.slide.isin(a.slides)]
    c = a.context
    if c:                                                     # row by row, so blending keeps few tiles pending
        pos = np.array([grid_position(p) for p in rows.image])
        rows = rows.assign(_c=pos[:, 0], _r=pos[:, 1]).sort_values(["slide", "_r", "_c"]).drop(columns=["_c", "_r"])
    else:
        rows = rows.sort_values("slide", kind="stable")
    rows = rows.reset_index(drop=True)
    ds = TileDataset(a.data, rows, context=c, zoom=tuple(a.zoom) if a.zoom else None)
    size = ds.size - 2 * c
    print(f"{a.ckpt} (epoch {ck['epoch']}): {len(rows)} tiles, {rows.slide.nunique()} slides, "
          f"{'tile' if not c else f'context {c}'}{', flips' if a.tta else ''}, tiles of {size} px -> {a.out}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    left = rows.slide.value_counts().to_dict()                # tiles of each slide not yet final
    ready, pending = {}, {}                                   # slide -> {tile: packed masks}; tile -> (image, truth)
    blender = Blender(rows, c, dev) if c else None

    def finish(probs):
        if not probs:
            return
        idx = np.array(sorted(probs))
        img = torch.stack([pending[i][0] for i in idx])
        truth = torch.stack([pending.pop(i)[1] for i in idx])
        pred = torch.stack([probs[i] for i in idx]) > 0.5
        masks = [np.packbits(m.reshape(len(idx), -1).cpu().numpy(), axis=1) for m in (pred, truth, tissue_mask(img))]
        for j, i in enumerate(idx):
            s = rows.slide[i]
            ready.setdefault(s, {})[i] = [m[j] for m in masks]
            left[s] -= 1
            if not left[s]:
                got = ready.pop(s)
                order = sorted(got)
                np.savez_compressed(slide_file(a.out, s), slide=s, size=size, image=rows.image[order].values.astype(str),
                                    **{k: np.stack([got[i][n] for i in order]) for n, k in enumerate(["pred", "truth", "tissue"])})

    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=a.device == "cuda")
    for img, mask, idx in dl:
        img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
        prob = tumour_probability(model, to_input(img), a.tta)
        idx = idx.tolist()
        for j, i in enumerate(idx):
            pending[i] = (img[j, :, c:c + size, c:c + size].clone(), mask[j, :, c:c + size, c:c + size].clone() > 0)
        finish(blender.add(idx, prob) if blender else {i: prob[j, :, c:c + size, c:c + size] for j, i in enumerate(idx)})
    assert not pending and not ready, f"{len(pending)} tiles / {len(ready)} slides never finished"

    (a.out / "done.json").write_text(json.dumps(dict(
        ckpt=str(a.ckpt), epoch=ck["epoch"], tiles=len(rows), slides=int(rows.slide.nunique()), size=size, context=c,
        tta=a.tta, zoom=a.zoom, index=str(a.index), split_file=str(a.split_file), split=a.split,
        minutes=round((time.time() - t0) / 60, 1)), indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
