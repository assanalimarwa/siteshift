"""
Score a trained U-Net on the test (or val) slides and draw original vs predicted masks.

    python evaluate.py --name resnet34_v2                        # test slides, context + blending + flips
    python evaluate.py --name resnet34_v2 --context 0 --no-tta   # each tile alone, as in training
    python evaluate.py --name resnet34_v2 --split val

Each tile is predicted inside a window that adds --context pixels of its neighbours on every
side (default 256, i.e. a 1024 x 1024 window), so decisions near a tile's edge see what lies
beyond it. The overlapping windows are then blended (default): each is weighted by a Hann window
and averaged with the windows of the neighbouring tiles, so adjacent tiles agree at their shared
border; --no-blend keeps only each tile's own window. With TTA (default) every window is
averaged over its four flips. Every tile of the chosen slides is scored, glass included, with
the probability thresholded at 0.5.

Writes runs/<name>/eval_<split>_<mode>/ (mode e.g. ctx256_blend_tta, ctx256_tta or tile):
    summary.json        Dice and IoU over all pixels, mean per-slide Dice, mean Dice on tumour
                        tiles, share of tumour-free tiles left (almost) clean, seam ratios
    per_slide.csv       pixel counts and Dice per slide
    per_tile.csv        pixel counts and Dice per tile
    slides/<slide>.png  the whole slide stitched from its tiles: image, comparison,
                        original mask, predicted mask (cells without a tile are hatched)
    tiles/<slide>.png   a few tiles of the slide at full resolution, same panels
    seams/<slide>.png   3 x 3 neighbouring tiles stitched at full resolution with the tile
                        grid drawn in, to see whether the prediction runs on across tile borders

Seams: at every border between two adjacent tiles, count the pixels where the two pixel lines that
meet there disagree; compare with the same count across a virtual border through the middle of
each tile (columns / rows 255 and 256), where the mask comes from one prediction and can only
change by its natural drift.
    seam_ratio      mean mismatch at real borders / mean mismatch at the virtual ones
                    (about 1 = the mask runs on smoothly across tile borders)
    seam_jumps      share of borders touched by tumour that show a jump of >= 8 mismatched pixels
    mid_jumps       the same share at the virtual mid-tile borders, as a baseline
Reported for the prediction and, as a reference, for the original mask.

In the comparison, yellow = tumour in both, green = tumour only in the original mask (missed),
red = tumour only in the prediction (false alarm).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

import breakdown
from data import TileDataset, grid_position, load_rows, to_input
from metrics import pixel_counts, summarize, tissue_mask
from model import build_model
from stain import StainNormalizer

HERE = P.INDEX
BOTH, MISSED, EXTRA = (255, 200, 0), (0, 190, 0), (230, 0, 0)
LEGEND = "comparison\nyellow both, green missed, red extra"


def compare(rgb, truth, pred):
    out = rgb.astype(np.float32)
    for m, colour in ((truth & pred, BOTH), (truth & ~pred, MISSED), (~truth & pred, EXTRA)):
        out[m] = 0.45 * out[m] + 0.55 * np.array(colour, np.float32)
    return out.astype(np.uint8)


def grid(panels, titles, ncols, font_size=18, gap=8):
    """Equal-sized RGB panels in a grid, each with its (up to two-line) title above it."""
    font = ImageFont.load_default(size=font_size)
    h, w = panels[0].shape[:2]
    bar = 2 * (font_size + 4) + 6
    nrows = -(-len(panels) // ncols)
    canvas = Image.new("RGB", (ncols * w + (ncols - 1) * gap, nrows * (h + bar) + (nrows - 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (p, t) in enumerate(zip(panels, titles)):
        x, y = (k % ncols) * (w + gap), (k // ncols) * (h + bar + gap)
        draw.text((x + 4, y + 4), t, fill="black", font=font)
        canvas.paste(Image.fromarray(p), (x, y + bar))
    return canvas


def as_rgb(mask):
    return np.repeat((mask * 255).astype(np.uint8)[..., None], 3, 2)


def hatch(h, w, step=6):
    """Light diagonal hatching for cells that have no tile, so they are not mistaken for glass."""
    yy, xx = np.mgrid[:h, :w]
    out = np.full((h, w, 3), 246, np.uint8)
    out[(xx + yy) % step == 0] = (200, 200, 205)
    return out


JUMP = 8   # mismatched pixels along a border that make a visible step


def border_lines(mask):
    """(B,1,T,T) bool -> (B, 8, T): left, right, top, bottom border lines, then the two middle
    columns and the two middle rows (a virtual border through the tile)."""
    m, h = mask[:, 0], mask.shape[-1] // 2
    return torch.stack([m[:, :, 0], m[:, :, -1], m[:, 0, :], m[:, -1, :],
                        m[:, :, h - 1], m[:, :, h], m[:, h - 1, :], m[:, h, :]], 1)


def seam_stats(rows, lines, prefix):
    """Mismatch across real tile borders against mismatch across virtual mid-tile borders."""
    pos = {(s, *grid_position(p)): i for i, (s, p) in enumerate(zip(rows.slide, rows.image))}
    pairs = [(i, j, k) for (s, c, r), i in pos.items()
             for k, nb in enumerate(((s, c + 1, r), (s, c, r + 1))) if (j := pos.get(nb)) is not None]
    if not pairs:
        return {}
    i, j, k = map(np.array, zip(*pairs))
    a_line, b_line = np.where(k == 0, 1, 3), np.where(k == 0, 0, 2)        # right|left, bottom|top
    a, b = lines[i, a_line], lines[j, b_line]
    seam = (a != b).sum(1)
    seam_touched = (a | b).any(1)
    mid_a, mid_b = np.concatenate([lines[:, 4], lines[:, 6]]), np.concatenate([lines[:, 5], lines[:, 7]])
    mid = (mid_a != mid_b).sum(1)
    mid_touched = (mid_a | mid_b).any(1)
    return {f"seam_ratio_{prefix}": float(seam.mean() / max(mid.mean(), 1e-9)),
            f"seam_jumps_{prefix}": float((seam[seam_touched] >= JUMP).mean()) if seam_touched.any() else 0.0,
            f"mid_jumps_{prefix}": float((mid[mid_touched] >= JUMP).mean()) if mid_touched.any() else 0.0}


def pick_blocks(rows, seed=0):
    """Per slide: a tile on a tumour border whose 8 neighbours all exist (with tissue), as a 3 x 3 block."""
    rng = np.random.default_rng(seed)
    pos = {(s, *grid_position(p)): i for i, (s, p) in enumerate(zip(rows.slide, rows.image))}
    blocks = {}
    for slide, r in rows.groupby("slide"):
        cands = []
        for i in r.index[(r.tumour_frac > 0.2) & (r.tumour_frac < 0.8)]:
            c, rr = grid_position(rows.image[i])
            block = [pos.get((slide, c + dc, rr + dr)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            if None not in block and rows.tissue_frac.values[block].min() >= 0.2:
                cands.append(block)
        if cands:
            blocks[slide] = cands[rng.integers(len(cands))]
    return blocks


def save_blocks(blocks, kept, folder, panel=1024):
    """Stitch each 3 x 3 block at full resolution and draw the tile grid on it."""
    folder.mkdir(parents=True, exist_ok=True)
    for slide, block in blocks.items():
        rgb, truth, pred = (np.concatenate([np.concatenate([kept[block[3 * r + c]][k] for c in range(3)], 1)
                                            for r in range(3)], 0) for k in range(3))
        t = rgb.shape[0] // 3
        out = []
        for p in (rgb, compare(rgb, truth, pred), as_rgb(truth), as_rgb(pred)):
            p = p.copy()
            for g in (t, 2 * t):                                # tile borders: dashed cyan
                dash = (np.arange(p.shape[0]) // 12) % 2 == 0
                p[dash, g - 1:g + 1] = (0, 200, 255)
                p[g - 1:g + 1, dash] = (0, 200, 255)
            out.append(np.array(Image.fromarray(p).resize((panel, panel), Image.LANCZOS)))
        name = slide.replace(".bif - Series 0", "")
        grid(out, [f"{name}: 3 x 3 tiles at full resolution (cyan = tile borders)", LEGEND,
                   "original mask", "predicted mask"], ncols=2).save(folder / f"{name}.png")


@torch.no_grad()
def tumour_probability(model, x, tta):
    """Sigmoid output for a normalised batch, averaged over the four flips if tta."""
    flips = ([], [-1], [-2], [-2, -1]) if tta else ([],)
    p = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        for dims in flips:
            out = model(x.flip(dims) if dims else x).float()
            p = p + torch.sigmoid(out.flip(dims) if dims else out)
    return p / len(flips)


def window_weight(size, device):
    """2-D Hann window: 1 in the centre, falling to ~0 at the window's edges."""
    w = torch.sin(torch.pi * (torch.arange(size, device=device) + 0.5) / size) ** 2
    return (w[:, None] * w[None, :]).clamp_min(1e-4)


class Blender:
    """Averages the overlapping window predictions into one probability per tile.

    Every tile's window (the tile plus `context` pixels of its neighbours) is weighted by a Hann
    window and added to each tile it covers: the tile itself and the margins of its eight
    neighbours. At a border between two tiles both windows weigh exactly the same, so the two
    tiles agree there and the merged mask runs on without a step. A tile is returned as soon as
    every existing tile whose window reaches it has been predicted (process rows in slide order to
    keep few tiles pending).
    """

    def __init__(self, rows, context, device, tile=512):
        self.c, self.t = context, tile
        self.reach = -(-context // tile)
        self.pos = [grid_position(p) for p in rows.image]
        self.slides = rows.slide.values
        self.index = {(s, c, r): i for i, (s, (c, r)) in enumerate(zip(self.slides, self.pos))}
        self.weight = window_weight(tile + 2 * context, device)
        self.num, self.den, self.got = {}, {}, {}

    def around(self, i):
        (c, r), s, n = self.pos[i], self.slides[i], self.reach
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                k = self.index.get((s, c + dc, r + dr))
                if k is not None:
                    yield k, dr, dc

    def add(self, idx, prob):
        """idx: dataset indices of the windows in prob (B,1,S,S). Returns {tile: (1,T,T) probability}."""
        ready, size, t = {}, prob.shape[-1], self.t
        for j, i in enumerate(idx):
            for k, dr, dc in self.around(i):
                y0, x0 = self.c + dr * t, self.c + dc * t          # tile k's corner inside window i
                ys, ye, xs, xe = max(y0, 0), min(y0 + t, size), max(x0, 0), min(x0 + t, size)
                if ys >= ye or xs >= xe:
                    continue
                if k not in self.num:
                    self.num[k] = torch.zeros(t, t, device=prob.device)
                    self.den[k] = torch.zeros(t, t, device=prob.device)
                    self.got[k] = 0
                w = self.weight[ys:ye, xs:xe]
                self.num[k][ys - y0:ye - y0, xs - x0:xe - x0] += prob[j, 0, ys:ye, xs:xe] * w
                self.den[k][ys - y0:ye - y0, xs - x0:xe - x0] += w
                self.got[k] += 1
                if self.got[k] == sum(1 for _ in self.around(k)):
                    ready[k] = (self.num.pop(k) / self.den.pop(k))[None]
                    del self.got[k]
        return ready


class SlideMaps:
    """Each slide stitched back together from its tiles at a reduced scale."""

    def __init__(self, rows, long_side=1400):
        self.rows = rows
        self.pos = np.array([grid_position(p) for p in rows.image])
        self.maps = {}
        for slide, idx in rows.groupby("slide").indices.items():
            (c0, r0), (c1, r1) = self.pos[idx].min(0), self.pos[idx].max(0)
            px = int(2 ** np.clip(np.floor(np.log2(long_side / max(c1 - c0 + 1, r1 - r0 + 1))), 3, 7))
            shape = ((r1 - r0 + 1) * px, (c1 - c0 + 1) * px)
            self.maps[slide] = dict(px=px, c0=c0, r0=r0, img=hatch(*shape),
                                    truth=np.zeros(shape, bool), pred=np.zeros(shape, bool))

    def add(self, idx, img, truth, pred):
        """idx: dataset indices; img (B,3,T,T) uint8, truth / pred (B,1,T,T) bool, on the GPU."""
        slides = self.rows.slide.values[idx]
        for px in {self.maps[s]["px"] for s in slides}:
            k = img.shape[-1] // px
            small = dict(img=F.avg_pool2d(img.float(), k).round().byte().permute(0, 2, 3, 1).cpu().numpy(),
                         truth=(F.avg_pool2d(truth.float(), k) > 0.5)[:, 0].cpu().numpy(),
                         pred=(F.avg_pool2d(pred.float(), k) > 0.5)[:, 0].cpu().numpy())
            for j, i in enumerate(idx):
                m = self.maps[slides[j]]
                if m["px"] != px:
                    continue
                y0, x0 = (self.pos[i, 1] - m["r0"]) * px, (self.pos[i, 0] - m["c0"]) * px
                for key in ("img", "truth", "pred"):
                    m[key][y0:y0 + px, x0:x0 + px] = small[key][j]

    def save(self, folder, per_slide):
        folder.mkdir(parents=True, exist_ok=True)
        slide_dice = per_slide.set_index("slide").dice
        for slide, m in self.maps.items():
            name = slide.replace(".bif - Series 0", "")
            tall = m["img"].shape[0] >= 1.5 * m["img"].shape[1]
            grid([m["img"], compare(m["img"], m["truth"], m["pred"]), as_rgb(m["truth"]), as_rgb(m["pred"])],
                 [f"{name}  (slide Dice {slide_dice[slide]:.3f})", LEGEND, "original mask", "predicted mask"],
                 ncols=4 if tall else 2).save(folder / f"{name}.png")


def pick_examples(rows, per_slide=3, seed=0):
    """Per slide: two tiles with tumour (preferably on a tumour border) and one without."""
    rng = np.random.default_rng(seed)
    picks = {}
    for slide, r in rows.groupby("slide"):
        border = r.index[(r.tumour_frac > 0.1) & (r.tumour_frac < 0.9)]
        tumour = border if len(border) >= per_slide - 1 else r.index[r.tumour_frac > 0]
        clear = r.index[(r.tumour_frac == 0) & (r.tissue_frac >= 0.3)]
        picks[slide] = [int(i) for i in rng.choice(tumour, min(per_slide - 1, len(tumour)), replace=False)] + \
                       [int(i) for i in rng.choice(clear, min(1, len(clear)), replace=False)]
    return picks


def save_examples(picks, kept, rows, per_tile, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for slide, idx in picks.items():
        panels, titles = [], []
        for i in idx:
            rgb, truth, pred = kept[i]
            x, y = (v * 1024 for v in grid_position(rows.image[i]))
            panels += [rgb, as_rgb(truth), as_rgb(pred), compare(rgb, truth, pred)]
            titles += [f"x={x} y={y}", f"original mask ({truth.mean():.0%} tumour)",
                       f"predicted ({pred.mean():.0%}, tile Dice {per_tile.dice[i]:.2f})", LEGEND]
        grid(panels, titles, ncols=4).save(folder / f"{slide.replace('.bif - Series 0', '')}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run folder under runs/")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--context", type=int, default=256, help="pixels of neighbouring tiles around each tile")
    ap.add_argument("--no-tta", dest="tta", action="store_false", help="skip averaging over flips")
    ap.add_argument("--no-blend", dest="blend", action="store_false",
                    help="keep only each tile's own window instead of averaging the overlapping windows")
    ap.add_argument("--data", type=Path, default=Path(str(P.DATASET47)))
    ap.add_argument("--index", type=Path, default=HERE / "tiles.csv")
    ap.add_argument("--split-file", type=Path, default=HERE / "split.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu: only for tests")
    ap.add_argument("--no-pictures", action="store_true", help="metrics only, skip slides/ and tiles/")
    ap.add_argument("--ensemble", nargs="+", type=Path, default=[],
                    help="more checkpoints; their probabilities are averaged with --name's model (equal weights)")
    ap.add_argument("--ensemble-name", default="ens", help="added to the output folder name, e.g. eval_test_tile_ens-<name>")
    ap.add_argument("--stain", type=Path, help="stain normalisation parameters (stain.py fit), e.g. cv/stain_macenko_sicap.json; "
                                               "output folder gets _stain-<file stem>")
    ap.add_argument("--breakdown", action="store_true",
                    help="add breakdown.COLUMNS (wrong pixels on glass / near / far from the outline); output folder gets _breakdown")
    ap.add_argument("--um-per-px", type=float, default=0.5, help="pixel size of the tiles, for --breakdown and --tolerant")
    ap.add_argument("--save-pred", type=Path, help="also save the thresholded predictions per slide in protocol_predict.py's "
                         "format (<dir>/<slide>.npz + done.json) for protocol.py, from the same pass")
    ap.add_argument("--save-prob", type=Path, help="also save the tumour probability per slide as uint8 (round(255 p)) with the "
                         "packed truth and tissue masks (<dir>/<slide>.npz: prob, truth, tissue, image + done.json; threshold47.py); "
                         "the output folder gets _prob, so an earlier evaluation is never overwritten")
    ap.add_argument("--tolerant", action="store_true",
                    help="add boundary-tolerant counts tp_st, fp_st, fn_st: the same, leaving out every pixel within "
                         "breakdown.BAND_UM (50 um) of the annotated outline on either side (square neighbourhoods "
                         "inside the tile, as --breakdown); summary gets dice_tolerant; output folder gets _tol")
    a = ap.parse_args()

    run = P.RUNS / a.name
    dev = torch.device(a.device)
    if a.device == "cuda" and os.environ.get("QP_GPU_MEM_GB"):                       # shared GPU: never take more than this (queue.py sets it)
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["QP_GPU_MEM_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    def load(path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        m = build_model(ck["arch"], pretrained=False).to(dev, memory_format=torch.channels_last)
        m.load_state_dict(ck["model"])
        return m.eval(), ck

    model, ck = load(run / a.checkpoint)
    extra = [load(p)[0] for p in a.ensemble]

    rows = load_rows(a.index, a.split_file, a.split)
    pos = np.array([grid_position(p) for p in rows.image])        # row by row, so blending keeps few tiles pending
    rows = rows.assign(_c=pos[:, 0], _r=pos[:, 1]).sort_values(["slide", "_r", "_c"]) \
               .drop(columns=["_c", "_r"]).reset_index(drop=True)
    c = a.context
    blend = bool(c) and a.blend
    mode = (f"ctx{c}" if c else "tile") + ("_blend" if blend else "") + ("_tta" if a.tta else "") \
        + (f"_ens-{a.ensemble_name}" if extra else "") + (f"_stain-{a.stain.stem}" if a.stain else "") \
        + ("_breakdown" if a.breakdown else "") + ("_tol" if a.tolerant else "")
    print(f"{a.name}/{a.checkpoint} (epoch {ck['epoch']}) on {a.split}, {mode}: {len(rows)} tiles, "
          f"{rows.slide.nunique()} slides", flush=True)

    maps = None if a.no_pictures else SlideMaps(rows)
    picks = {} if a.no_pictures else pick_examples(rows)
    blocks = {} if a.no_pictures else pick_blocks(rows)
    wanted = {i for idx in [*picks.values(), *blocks.values()] for i in idx}
    kept, counts = {}, np.zeros((len(rows), 4), np.int64)
    benign_counts = np.zeros((len(rows), 2), np.int64)        # false tumour on tumour-free tissue, that tissue
    place_counts = np.zeros((len(rows), len(breakdown.COLUMNS)), np.int64) if a.breakdown else None
    tol_counts = np.zeros((len(rows), 3), np.int64) if a.tolerant else None
    band_px = round(breakdown.BAND_UM / a.um_per_px)
    pred_lines = np.zeros((len(rows), 8, 512), bool)
    truth_lines = np.zeros((len(rows), 8, 512), bool)
    blender = Blender(rows, c, dev) if blend else None
    pending = {}                                             # tile -> (image, truth) until its prediction is final
    left, ready = rows.slide.value_counts().to_dict(), {}     # --save-pred: tiles of each slide not yet final, packed masks
    if a.save_pred:
        if (a.save_pred / "done.json").exists():
            raise SystemExit(f"{a.save_pred} already holds finished predictions")
        a.save_pred.mkdir(parents=True, exist_ok=True)
    left_p, ready_p, writes = rows.slide.value_counts().to_dict(), {}, []
    writer = ThreadPoolExecutor(1) if a.save_prob else None       # zlib releases the GIL: compression beside the GPU
    if a.save_prob:
        if (a.save_prob / "done.json").exists():
            raise SystemExit(f"{a.save_prob} already holds finished probabilities")
        a.save_prob.mkdir(parents=True, exist_ok=True)

    def finish(probs):
        if not probs:
            return
        idx = np.array(sorted(probs))
        img = torch.stack([pending[i][0] for i in idx])
        truth = torch.stack([pending.pop(i)[1] for i in idx])
        prob = torch.stack([probs[i] for i in idx])
        pred = prob > 0.5
        if a.save_prob:
            q = (prob * 255).round().to(torch.uint8)[:, 0].cpu().numpy()
            tt = [np.packbits(m.reshape(len(idx), -1).cpu().numpy(), axis=1) for m in (truth, tissue_mask(img))]
            for j, i in enumerate(idx):
                s = rows.slide[i]
                ready_p.setdefault(s, {})[i] = (q[j], tt[0][j], tt[1][j])
                left_p[s] -= 1
                if not left_p[s]:
                    got = ready_p.pop(s)
                    order = sorted(got)
                    for w in writes[:-2]:                          # at most 3 slides waiting for the disk
                        w.result()
                    writes.append(writer.submit(
                        np.savez_compressed, a.save_prob / (re.sub(r"[^\w.-]+", "_", s) + ".npz"), slide=s, size=pred.shape[-1],
                        image=rows.image[order].values.astype(str), prob=np.stack([got[i][0] for i in order]),
                        truth=np.stack([got[i][1] for i in order]), tissue=np.stack([got[i][2] for i in order])))
        counts[idx] = pixel_counts(pred, truth).cpu().numpy()
        benign = tissue_mask(img) & ~truth
        benign_counts[idx] = torch.stack([(pred & benign).sum((1, 2, 3)), benign.sum((1, 2, 3))], 1).cpu().numpy()
        if place_counts is not None:
            place_counts[idx] = breakdown.counts(img, truth, pred, a.um_per_px).cpu().numpy()
        if tol_counts is not None:
            keep = ~(breakdown._dilate(truth, band_px) & breakdown._dilate(~truth, band_px))   # outside the band
            tol_counts[idx] = torch.stack([(pred & truth & keep).sum((1, 2, 3)), (pred & ~truth & keep).sum((1, 2, 3)),
                                           (~pred & truth & keep).sum((1, 2, 3))], 1).cpu().numpy()
        pred_lines[idx] = border_lines(pred).cpu().numpy()
        if a.save_pred:
            packed = [np.packbits(m.reshape(len(idx), -1).cpu().numpy(), axis=1) for m in (pred, truth, tissue_mask(img))]
            for j, i in enumerate(idx):
                s = rows.slide[i]
                ready.setdefault(s, {})[i] = [m[j] for m in packed]
                left[s] -= 1
                if not left[s]:
                    got = ready.pop(s)
                    order = sorted(got)
                    np.savez_compressed(a.save_pred / (re.sub(r"[^\w.-]+", "_", s) + ".npz"), slide=s, size=pred.shape[-1], image=rows.image[order].values.astype(str),
                                        **{k: np.stack([got[i][n] for i in order]) for n, k in enumerate(["pred", "truth", "tissue"])})
        truth_lines[idx] = border_lines(truth).cpu().numpy()
        if maps:
            maps.add(idx, img, truth, pred)
        for j, i in enumerate(idx):
            if i in wanted:
                kept[i] = (img[j].permute(1, 2, 0).cpu().numpy(), truth[j, 0].cpu().numpy(), pred[j, 0].cpu().numpy())

    stain = StainNormalizer.load(a.stain) if a.stain else None
    dl = DataLoader(TileDataset(a.data, rows, context=c, stain=stain), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    for img, mask, idx in dl:
        img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
        x = to_input(img)
        prob = sum(tumour_probability(m, x, a.tta) for m in [model, *extra]) / (1 + len(extra))
        t = img.shape[-1] - 2 * c
        idx = idx.tolist()
        for j, i in enumerate(idx):
            pending[i] = (img[j, :, c:c + t, c:c + t].clone(), mask[j, :, c:c + t, c:c + t].clone() > 0)
        finish(blender.add(idx, prob) if blender else
               {i: prob[j, :, c:c + t, c:c + t] for j, i in enumerate(idx)})
    assert not pending, f"{len(pending)} tiles never got a final prediction"
    if a.save_prob:
        assert not ready_p, f"{len(ready_p)} slides never finished"
        for w in writes:
            w.result()
        writer.shutdown()
        (a.save_prob / "done.json").write_text(json.dumps(dict(
            ckpt=str(run / a.checkpoint), epoch=ck["epoch"], tiles=len(rows), slides=int(rows.slide.nunique()), size=512,
            context=c, tta=a.tta, index=str(a.index), split_file=str(a.split_file), split=a.split, prob="uint8 round(255 p)",
            ensemble=[str(e) for e in a.ensemble], by="evaluate.py --save-prob"), indent=2))
    if a.save_pred:
        assert not ready, f"{len(ready)} slides never finished"
        (a.save_pred / "done.json").write_text(json.dumps(dict(
            ckpt=str(run / a.checkpoint), epoch=ck["epoch"], tiles=len(rows), slides=int(rows.slide.nunique()), size=512,
            context=c, tta=a.tta, zoom=None, index=str(a.index), split_file=str(a.split_file), split=a.split,
            ensemble=[str(e) for e in a.ensemble], by="evaluate.py --save-pred"), indent=2))

    summary, per_slide, per_tile = summarize(rows, counts)
    per_tile[["fp_tissue", "benign_tissue_px"]] = benign_counts
    if place_counts is not None:
        per_tile[breakdown.COLUMNS] = place_counts
    if tol_counts is not None:
        per_tile[["tp_st", "fp_st", "fn_st"]] = tol_counts
        tp, fp, fn = tol_counts.sum(0)
        summary["dice_tolerant"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    benign = per_tile.groupby("slide")[["fp_tissue", "benign_tissue_px"]].sum()
    per_slide = per_slide.join(benign, on="slide")
    per_slide["false_tumour_rate"] = per_slide.fp_tissue / per_slide.benign_tissue_px.clip(lower=1)
    summary["false_tumour_rate"] = float(benign.fp_tissue.sum() / max(benign.benign_tissue_px.sum(), 1))
    summary = dict(checkpoint=a.checkpoint, epoch=ck["epoch"], split=a.split, mode=mode, **summary,
                   **seam_stats(rows, pred_lines, "pred"), **seam_stats(rows, truth_lines, "truth"))
    tag = "" if a.split_file.resolve() == (HERE / "split.json").resolve() else a.split_file.stem + "_"
    out = run / f"eval_{tag}{a.split}_{mode}{'_prob' if a.save_prob else ''}"                  # e.g. eval_test_tile, or eval_fold0_test_tile for cv/fold0.json
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    per_slide.to_csv(out / "per_slide.csv", index=False)
    per_tile.to_csv(out / "per_tile.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(per_slide.assign(slide=per_slide.slide.str.replace(".bif - Series 0", ""))
          [["slide", "tiles", "tumour_px", "dice"]].to_string(index=False, float_format="%.4f"), flush=True)
    if maps:
        maps.save(out / "slides", per_slide)
        save_examples(picks, kept, rows, per_tile, out / "tiles")
        save_blocks(blocks, kept, out / "seams")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
