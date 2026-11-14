"""Shared helpers for the paper figures: style, tile stitching, outlines, scale bars."""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

U = P.INDEX
FIGS = P.RESULTS                              # outputs: results/datasets/
sys.path.insert(0, str(U))
D47 = P.DATASET47
DPI = 300
FULL_W, COL_W = 6.9, 3.3
# Okabe-Ito (colour-blind safe)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", grey="#999999")
OUTLINE = OI["yellow"]          # tumour outline: Okabe-Ito yellow over a thin black line (readable on glass and tissue)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "lines.linewidth": 0.8, "savefig.dpi": DPI})


def grid_pos(image):
    """(col, row) of a tile in its slide grid from '[d=..,x=..,y=..,w=..,h=..]' (ours: w=1024 at d=2; PANDA: w=512)."""
    x, y, w = map(int, re.search(r"x=(\d+),y=(\d+),w=(\d+)", image).groups())
    return x // w, y // w


def mask_rel(rel):
    from data import mask_path
    return str(mask_path(rel))


def stitch(root, rows, c0, r0, cols, nrows, out_tile=512, mask=True, fill=(235, 235, 235)):
    """Stitch a window of a slide grid into RGB (+ mask); tiles not exported (no tissue) = glass colour `fill` / 0.
    Each tile is resized to out_tile px (downsampling for thumbnails)."""
    rgb = np.empty((nrows * out_tile, cols * out_tile, 3), np.uint8)
    rgb[:] = fill
    m = np.zeros((nrows * out_tile, cols * out_tile), np.uint8)
    for rel in rows.image:
        c, r = grid_pos(rel)
        if not (c0 <= c < c0 + cols and r0 <= r < r0 + nrows):
            continue
        y, x = (r - r0) * out_tile, (c - c0) * out_tile
        im = Image.open(root / rel).convert("RGB")
        if im.width != out_tile:
            im = im.resize((out_tile, out_tile), Image.BILINEAR)
        rgb[y:y + out_tile, x:x + out_tile] = np.asarray(im)
        if mask:
            mk = Image.open(root / mask_rel(rel))
            if mk.width != out_tile:
                mk = mk.resize((out_tile, out_tile), Image.NEAREST)
            m[y:y + out_tile, x:x + out_tile] = np.asarray(mk)
    return rgb, m


def to_panel(img, w_in, h_in=None, nearest=False):
    """Resample an image to exactly 300 dpi for a panel of w_in x h_in inches (embedded as is in the PDF)."""
    h_in = h_in if h_in is not None else w_in * img.shape[0] / img.shape[1]
    size = (round(w_in * DPI), round(h_in * DPI))
    return np.asarray(Image.fromarray(img).resize(size, Image.NEAREST if nearest else Image.LANCZOS))


def show(ax, img, mask=None, lw=0.9, tumour_values=(255,)):
    """Image panel with the tumour outline of `mask` (any resolution, same field) drawn as vector contours."""
    ax.imshow(img, interpolation="none")
    if mask is not None and np.isin(mask, tumour_values).any():
        t = np.isin(mask, tumour_values).astype(float)
        sy, sx = img.shape[0] / t.shape[0], img.shape[1] / t.shape[1]
        ext = (-0.5, t.shape[1] * sx - 0.5, t.shape[0] * sy - 0.5, -0.5)
        ax.contour(t, levels=[0.5], colors=OI["black"], linewidths=lw + 0.7, extent=ext, origin="upper")
        ax.contour(t, levels=[0.5], colors=OUTLINE, linewidths=lw, extent=ext, origin="upper")
    ax.set_xlim(-0.5, img.shape[1] - 0.5), ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def scale_bar(ax, img, um_per_px, length_um, loc="lower right"):
    """Scale bar of length_um on an image panel; um_per_px = microns per pixel of the DISPLAYED image."""
    h, w = img.shape[:2]
    px = length_um / um_per_px
    pad = 0.05 * min(w, h)
    x1 = w - pad if "right" in loc else pad + px
    y = h - pad if "lower" in loc else pad + 0.08 * h
    ax.plot([x1 - px, x1], [y, y], color="white", lw=3.0, solid_capstyle="butt")
    ax.plot([x1 - px, x1], [y, y], color="black", lw=1.5, solid_capstyle="butt")
    txt = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} \u00b5m"
    ax.text(x1 - px / 2, y - 0.025 * h, txt, ha="center", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))


def nice_bar(field_um):
    """A round scale-bar length of at most 30% of the field."""
    for v in [5000, 2000, 1000, 500, 250, 200, 100, 50, 25, 20, 10]:
        if v <= field_um * 0.3:
            return v
    return 10
