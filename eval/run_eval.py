"""
Score a folder of predicted tumour masks on our patients: per-patient Dice, clean-tile AUC and the detection / delineation attribution.

    python eval/run_eval.py --pred <PRED_DIR> --track {A,B,C} [--data <DATA>/dataset47] [--index data/tiles47.csv] [--out scores]
    python eval/run_eval.py --pred <PRED_DIR> [--split-file data/cv47/fold0.json --split test]        # any other tile set
                            [--out scores] [--workers 8]

Submission format (see eval/SUBMISSION_FORMAT.md): one PNG per tile, same name as the tile image of the dataset,
    <PRED_DIR>/<slide>/<tile image name>.png        512 x 512, 8 bit, 0 = not tumour, 255 = tumour (any value > 0 counts as tumour)
The tiles to score are the rows of the tile index: with --track, the test slides of the withheld fold 2 (data/cv47/fold2.json; Tracks A
and C: its 10 patients; Track B: its 5 large-section patients, data/cv47/folds.json); otherwise all of them, or one split of a split file. A missing file is an error, never an
implicit "no tumour".

The scoring is the one of the paper and uses the verified code unchanged: the tile masks are stitched on the slide's tile grid and passed to
protocol.analyse_slide (regions of >= 2,500 um^2, detected = >= 10% marked, Euclidean 50 um band); the per-patient measures are those of
protocol.measures. The clean-tile AUC scores each tile by its predicted tumour share (a binary submission therefore gives the share of
marked pixels; submit nothing but 0 / 255).

Writes <out>_per_patient.csv (one row per patient), <out>_per_slide.csv (pixel sums per slide) and prints the means.
Per-patient Dice is the Dice of the patient's slides pooled (the "patient-pooled Dice" of the paper's secondary measure) from the pixel sums;
the paper's headline per-patient Dice (mean of the patient's slide Dice) is reported as `dice_mean_of_slides`.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import siteshift_paths as P
import protocol
from data import grid_position, load_mask, mask_path

TILE = 512


def tissue_mask(rgb):
    a = rgb.astype(np.int16)
    mean = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    return (mean > 25) & (mean < 220) & (sat > 12)


def build_npz(args):
    """Write the protocol-format file of one slide (pred / truth / tissue bit-packed per tile) from the submission PNGs."""
    slide, images, data, pred, tmp = args
    packed = {"pred": [], "truth": [], "tissue": []}
    frac = []
    for rel in images:
        p = Path(pred) / slide / (Path(rel).stem + ".png")
        if not p.exists():
            raise FileNotFoundError(f"missing prediction: {p}")
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (TILE, TILE):
            raise ValueError(f"{p}: expected {TILE} x {TILE}, got {m.shape}")
        rgb = np.array(Image.open(Path(data) / rel).convert("RGB"))
        truth = load_mask(Path(data) / mask_path(rel)) > 0
        tis = tissue_mask(rgb)
        frac.append(float(tis.mean()))
        for k, a in (("pred", m > 0), ("truth", truth), ("tissue", tis)):
            packed[k].append(np.packbits(a.reshape(-1)))
    f = Path(tmp) / (slide.replace(" ", "_") + ".npz")
    np.savez_compressed(f, slide=slide, size=TILE, image=np.array(list(images), dtype=str), **{k: np.stack(v) for k, v in packed.items()})
    return f, list(zip(images, frac))


TEST_FOLD = 2          # the withheld leaderboard fold: the test fold of every track (README, "Benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="folder of predicted masks, <slide>/<tile>.png")
    ap.add_argument("--data", type=Path, default=P.DATASET47)
    ap.add_argument("--index", type=Path, default=P.INDEX / "tiles47.csv")
    ap.add_argument("--track", choices=["A", "B", "C"], help="benchmark track: score the test patients of fold 2 (B: its large-section patients only)")
    ap.add_argument("--split-file", type=Path, help="e.g. data/cv47/fold0.json; omit to score every tile of the index")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slides", nargs="+", help="only these slides")
    ap.add_argument("--out", type=Path, default=Path("scores"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    rows = pd.read_csv(a.index)
    if a.track:
        if a.split_file or a.slides:
            raise SystemExit("--track selects the tiles itself (fold 2, test); do not combine it with --split-file / --slides")
        a.split_file, a.split = P.INDEX / "cv47" / f"fold{TEST_FOLD}.json", "test"
    if a.split_file:
        import json
        rows = rows[rows.slide.isin(json.loads(a.split_file.read_text())["splits"][a.split])]
    if a.slides:
        rows = rows[rows.slide.isin(a.slides)]
    if a.track == "B":
        import json
        large = {int(x) for x in json.loads((P.INDEX / "cv47" / "folds.json").read_text())["large_section_patients"]}
        rows = rows[rows.slide.str.split(".").str[0].astype(int).isin(large)]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise SystemExit("no tiles selected")
    groups = [(s, sorted(g.image, key=lambda p: grid_position(p)[::-1])) for s, g in rows.groupby("slide")]
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(s, imgs, str(a.data), str(a.pred), tmp) for s, imgs in groups]
        with ProcessPoolExecutor(a.workers) as ex:
            built = list(ex.map(build_npz, jobs))
            files = [f for f, _ in built]
            res = list(ex.map(protocol.analyse_slide, files, [0.5] * len(files), [True] * len(files)))
    slides = pd.DataFrame([dict(slide=k, **s) for k, s, _ in res])
    tiles = pd.concat([t.assign(slide=k) for k, _, t in res])
    tissue = pd.DataFrame([x for _, fr in built for x in fr], columns=["image", "tissue_frac"])
    m = protocol.measures(slides, tiles, "ours", tissue)
    pooled = slides.assign(patient=slides.slide.str.split(".").str[0].astype(int))
    slide_dice = (2 * slides.tp / (2 * slides.tp + slides.fp + slides.fn).clip(lower=1)).where(slides.tumour_px > 0)
    m["dice_mean_of_slides"] = slide_dice.groupby(pooled.patient.values).mean()
    m.index.name = "patient"
    m.to_csv(f"{a.out}_per_patient.csv")
    slides.to_csv(f"{a.out}_per_slide.csv", index=False)
    show = ["dice", "dice_mean_of_slides", "clean_auc", "recall", "dice_found", "err_missed", "err_false", "err_fill", "err_boundary"]
    print(f"{len(m)} patients, {len(slides)} slides, {len(tiles)} tiles")
    print(m[show].mean(numeric_only=True).round(4).to_string())
    det, dele = (m.err_missed + m.err_false).mean(), (m.err_fill + m.err_boundary).mean()
    print(f"attribution (mean over patients): detection {det / (det + dele):.1%}, delineation {dele / (det + dele):.1%} of 1 - Dice")


if __name__ == "__main__":
    main()
