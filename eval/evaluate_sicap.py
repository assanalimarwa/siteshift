"""
Score a checkpoint on the held-out SICAPv2 tiles and put the result next to the same metrics on
our test slides, to measure how much a model loses when it moves from one dataset to the other.

    python evaluate_sicap.py --name sicap_pretrain
    python evaluate_sicap.py --name resnet34_v2

SICAPv2's official test partition is defined by its original file names, which the Kaggle copy
replaced by numbers, so the held-out sets here come from prepare_sicap.py:
    val     1,883 tiles never trained on; sicap_pretrain used them to choose its best epoch
    guard   1,077 tiles never used at all; they border the val stretches and may share a slide
            with training tiles
Tiles are enlarged to our pixel size (x1.75, 896 px, as during pretraining validation) and each
is predicted alone at threshold 0.5, the "tile" mode of evaluate.py.

Writes runs/<name>/eval_sicap_<split>_tile/ (summary.json, per_tile.csv, per_region.csv) and
prints one table with the same metrics on SICAPv2 and, if evaluate.py has scored it, on our test
slides (runs/<name>/eval_test_tile/per_tile.csv):
    tumour share        share of pixels that are tumour (Dice rises with it)
    dice                over all pixels
    dice_tumour_tiles   over the pixels of tiles that contain tumour only (no false alarms on
                        tumour-free tiles)
    tumour_tile_dice    mean per-tile Dice over tiles with tumour
    group_dice          mean Dice per slide (ours) / per overlap region (SICAPv2), groups with tumour
    clean_empty_tiles   share of tumour-free tiles with < 1% of pixels predicted tumour
    missed, extra       missed / falsely predicted tumour pixels, as a share of true tumour pixels
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
from torch.utils.data import DataLoader

import breakdown
from data import TileDataset, load_rows, to_input
from evaluate import tumour_probability
from metrics import dice, pixel_counts, summarize
from model import build_model

HERE = P.INDEX
ZOOM = (1.5, 2.0)


def comparable(per_tile, tile_pixels):
    t = per_tile
    tp, fp, fn, gt = (t[c].sum() for c in ["tp", "fp", "fn", "tumour_px"])
    tum = t[t.tumour_px > 0]
    groups = t.groupby("slide")[["tp", "fp", "fn", "tumour_px"]].sum()
    groups = groups[groups.tumour_px > 0]
    empty = t[t.tumour_px == 0]
    return dict(
        tiles=len(t),
        tumour_share=gt / (len(t) * tile_pixels),
        dice=float(dice(tp, fp, fn)),
        dice_tumour_tiles=float(dice(tum.tp.sum(), tum.fp.sum(), tum.fn.sum())),
        tumour_tile_dice=float(tum.dice.mean()),
        group_dice=float(dice(groups.tp, groups.fp, groups.fn).mean()),
        clean_empty_tiles=float((empty.fp < 0.01 * tile_pixels).mean()),
        missed=fn / gt,
        extra=fp / gt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run folder under runs/")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--splits", nargs="+", default=["val", "guard"])
    ap.add_argument("--data", type=Path, default=Path(str(P.DATA_ROOT / "sicapv2")))
    ap.add_argument("--index", type=Path, default=HERE / "sicap_tiles.csv")
    ap.add_argument("--split-file", type=Path, default=HERE / "sicap_split.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--breakdown", action="store_true",
                    help="add breakdown.COLUMNS (tiles enlarged to 0.5 um/px); output folder eval_sicap_<split>_tile_breakdown")
    a = ap.parse_args()

    run = P.RUNS / a.name
    dev = torch.device("cuda")
    ck = torch.load(run / a.checkpoint, map_location=dev, weights_only=False)
    model = build_model(ck["arch"], pretrained=False).to(dev, memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()

    table = {}
    torch.set_grad_enabled(False)
    for split in a.splits:
        rows = load_rows(a.index, a.split_file, split)
        ds = TileDataset(a.data, rows, zoom=ZOOM)
        counts = np.zeros((len(rows), 4), np.int64)
        places = np.zeros((len(rows), len(breakdown.COLUMNS)), np.int64)
        for img, mask, idx in DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True):
            img = img.to(dev, non_blocking=True)
            prob = tumour_probability(model, to_input(img), tta=False)
            pred, truth = prob > 0.5, mask.to(dev) > 0
            counts[idx.numpy()] = pixel_counts(pred, truth).cpu().numpy()
            if a.breakdown:
                places[idx.numpy()] = breakdown.counts(img, truth, pred, 0.5).cpu().numpy()
        summary, per_region, per_tile = summarize(rows, counts, tile_pixels=ds.size ** 2)
        if a.breakdown:
            per_tile[breakdown.COLUMNS] = places
        out = run / (f"eval_sicap_{split}_tile" + ("_breakdown" if a.breakdown else ""))
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(
            dict(checkpoint=a.checkpoint, epoch=ck["epoch"], split=split, zoom=ZOOM, tile_px=ds.size, **summary), indent=2))
        per_region.to_csv(out / "per_region.csv", index=False)
        per_tile.to_csv(out / "per_tile.csv", index=False)
        table[f"SICAPv2 {split}"] = comparable(per_tile, ds.size ** 2)
        print(f"-> {out}", flush=True)

    ours = run / "eval_test_tile" / "per_tile.csv"
    if ours.exists():
        table["our test"] = comparable(pd.read_csv(ours), 512 * 512)
    print(f"\n{a.name}/{a.checkpoint} (epoch {ck['epoch']}), each tile alone, threshold 0.5")
    print(pd.DataFrame(table).to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
