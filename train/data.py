"""Tile loading and augmentation shared by prepare.py, train.py and evaluate.py."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE_XY = re.compile(r"\[d=\d+,x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]")

# stain vectors of haematoxylin, eosin and DAB (Ruifrok & Johnston, as in skimage.color.rgb2hed)
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)

# colour / blur / noise augmentation strengths; "light" is what the first run (runs/resnet34) used
AUG = {
    "light": dict(hed=0.05, contrast=0.10, brightness=0.10, hue=0.0, saturation=0.0, blur=0.0, noise=0.0),
    "strong": dict(hed=0.10, contrast=0.15, brightness=0.15, hue=0.04, saturation=0.25, blur=0.25, noise=0.25),
}


IGNORE = 128    # mask value of unlabelled pixels (PANDA Radboud background / unknown, prepare_radboud.py)


def load_mask(path):
    """Tumour mask as uint8: 0 not tumour, 1 tumour, 2 unlabelled (mask value IGNORE; left out of the
    loss and the validation counts, metrics.py).

    QuPath writes palette PNGs whose pixel value is the class (0 = not tumour, 255 = tumour).
    Read that value, never convert("L"): that returns the brightness of the display colour.
    """
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    out = (m > 0).astype(np.uint8)
    out[m == IGNORE] = 2
    return out


def mask_path(image_rel):
    """dataset/<slide>/images/<tile>.png -> dataset/<slide>/masks/<tile>.png
    (masks are always PNG, also for the JPEG tiles of SICAPv2: sicapv2/images/7.jpg -> sicapv2/masks/7.png)"""
    p = Path(image_rel)
    return p.parent.parent / "masks" / (p.stem + ".png")


def grid_position(image_rel):
    """(column, row) of a tile in its slide's tile grid, from the coordinates in its name."""
    x, y, w, h = map(int, TILE_XY.search(str(image_rel)).groups())
    return x // w, y // h


def load_rows(index_csv, split_json, split, min_tissue=0.0):
    """Tiles of one split from tiles.csv; min_tissue drops mostly-glass tiles without tumour."""
    tiles = pd.read_csv(index_csv)
    slides = json.loads(Path(split_json).read_text())["splits"][split]
    rows = tiles[tiles.slide.isin(slides)]
    if min_tissue > 0:
        rows = rows[(rows.tissue_frac >= min_tissue) | (rows.tumour_frac > 0)]
    return rows.reset_index(drop=True)


class TileDataset(Dataset):
    """(uint8 image 3xSxS, uint8 mask 1xSxS, index) per row. Training adds random rot90 and flips.

    By default the sample is the tile itself (S = 512). Two options cut a window out of the
    slide instead, assembled from the tile and its neighbours (`tiles` lists every tile that
    may serve as a neighbour; a missing neighbour counts as glass: white, no tumour):

    shift    training only: the window is the tile moved by a random offset of up to +-shift
             pixels, so tile borders land anywhere in it and the model learns to predict
             across them.
    context  a margin of this many pixels around the tile (S = 512 + 2 * context), so the
             prediction for the tile (the centre of the window) sees past its edges.

    For tiles scanned at a coarser pixel size (SICAPv2, see prepare_sicap.py) a third option
    enlarges them to the pixel size of ours:

    zoom     (lo, hi). Training cuts a random square of 512 / f pixels out of the tile, f drawn
             from [lo, hi], and enlarges it to 512. Evaluation enlarges the whole tile by the
             middle factor, rounded to a multiple of 32 pixels (S = 896 for (1.5, 2.0)).

    Colour augmentation and normalisation run on the GPU in to_input(), so workers only
    decode images.
    """

    def __init__(self, root, rows, train=False, shift=0, context=0, tiles=None, tile_size=512, zoom=None, stain=None, mask_root=None):
        self.root = Path(root)
        self.mask_root = Path(mask_root) if mask_root else self.root   # masks from another folder (same layout), e.g. labels47.py
        self.images = rows.image.tolist()
        self.slides = rows.slide.tolist()
        self.train = train
        self.shift = shift if train else 0
        self.context = context
        self.T = tile_size
        self.zoom = zoom
        self.stain = stain              # optional StainNormalizer, applied to every tile read (incl. context)
        if zoom and (self.shift or context):
            raise ValueError("zoom works on single tiles; it cannot be combined with shift or context")
        self.size = 32 * round(tile_size * sum(zoom) / 2 / 32) if zoom and not train else tile_size + 2 * context
        if self.shift or self.context:
            self.pos = [grid_position(p) for p in self.images]
            src = rows if tiles is None else tiles
            self.lookup = {(s, *grid_position(p)): p for s, p in zip(src.slide, src.image)}

    def __len__(self):
        return len(self.images)

    def _read(self, rel):
        img = np.array(Image.open(self.root / rel).convert("RGB"))
        if self.stain is not None:
            img = self.stain(img, Path(rel).parts[0])
        return img, load_mask(self.mask_root / mask_path(rel))

    def _window(self, slide, col, row, dx, dy):
        T, size = self.T, self.T + 2 * self.context
        x0, y0 = col * T - self.context + dx, row * T - self.context + dy   # top-left, tile-grid pixels
        img = np.full((size, size, 3), 255, np.uint8)
        mask = np.zeros((size, size), np.uint8)
        for tr in range(y0 // T, (y0 + size - 1) // T + 1):
            for tc in range(x0 // T, (x0 + size - 1) // T + 1):
                rel = self.lookup.get((slide, tc, tr))
                if rel is None:
                    continue
                ti, tm = self._read(rel)
                ax, bx = max(x0, tc * T), min(x0 + size, (tc + 1) * T)
                ay, by = max(y0, tr * T), min(y0 + size, (tr + 1) * T)
                img[ay - y0:by - y0, ax - x0:bx - x0] = ti[ay - tr * T:by - tr * T, ax - tc * T:bx - tc * T]
                mask[ay - y0:by - y0, ax - x0:bx - x0] = tm[ay - tr * T:by - tr * T, ax - tc * T:bx - tc * T]
        return img, mask

    def _zoom(self, img, mask):
        T = self.T
        if self.train:
            lo, hi = self.zoom
            c = round(T / (lo + (hi - lo) * float(torch.rand(()))))
            x, y = torch.randint(0, T - c + 1, (2,)).tolist()
            box = (x, y, x + c, y + c)
        else:
            box = (0, 0, T, T)
        size = (self.size, self.size)
        return (np.array(Image.fromarray(img).resize(size, Image.BILINEAR, box=box)),
                np.array(Image.fromarray(mask).resize(size, Image.NEAREST, box=box)))

    def __getitem__(self, i):
        if self.shift or self.context:
            dx, dy = torch.randint(-self.shift, self.shift + 1, (2,)).tolist() if self.shift else (0, 0)
            img, mask = self._window(self.slides[i], *self.pos[i], dx, dy)
        else:
            img, mask = self._read(self.images[i])
        if self.zoom:
            img, mask = self._zoom(img, mask)
        if self.train:
            k = int(torch.randint(4, ()))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if torch.rand(()) < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img, mask, i


def stain_jitter(x, sigma=0.05):
    """Scale and shift each image's H, E and DAB optical densities by up to +-sigma
    (Tellez et al. 2018), so the model does not learn one lab's staining. x: (B,3,H,W) in [0,1]."""
    b = x.shape[0]
    od = -torch.log(x.clamp_min(1 / 255))
    hed = torch.einsum("bchw,cd->bdhw", od, HED_FROM_RGB.to(x))
    scale = 1 + (torch.rand(b, 3, 1, 1, device=x.device) * 2 - 1) * sigma
    shift = (torch.rand(b, 3, 1, 1, device=x.device) * 2 - 1) * sigma
    od = torch.einsum("bdhw,dc->bchw", hed * scale + shift, RGB_FROM_HED.to(x))
    return torch.exp(-od).clamp(0, 1)


RGB_TO_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
YIQ_TO_RGB = torch.linalg.inv(RGB_TO_YIQ)
LUMA = torch.tensor([0.299, 0.587, 0.114])


def hue_saturation(x, hue, saturation):
    """Rotate each image's hue by hue[b] turns (in YIQ space) and scale its saturation by saturation[b]."""
    theta = 2 * torch.pi * hue
    cos, sin, one, zero = torch.cos(theta), torch.sin(theta), torch.ones_like(theta), torch.zeros_like(theta)
    rot = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos], 1).view(-1, 3, 3)
    m = YIQ_TO_RGB.to(x) @ rot @ RGB_TO_YIQ.to(x)
    x = torch.einsum("bij,bjhw->bihw", m, x)
    gray = torch.einsum("c,bchw->bhw", LUMA.to(x), x)[:, None]
    return gray + saturation.view(-1, 1, 1, 1) * (x - gray)


def augment(x, strength):
    """Per-image colour, blur and noise augmentation of a (B,3,H,W) batch in [0,1]."""
    p, b, dev = AUG[strength], x.shape[0], x.device
    sym = lambda *shape: torch.rand(*shape, device=dev) * 2 - 1      # uniform in [-1, 1]
    coin = lambda prob: prob > 0 and float(torch.rand(())) < prob    # drawn on the CPU, no sync
    x = stain_jitter(x, p["hed"])
    if p["hue"] or p["saturation"]:
        x = hue_saturation(x, sym(b) * p["hue"], 1 + sym(b) * p["saturation"])
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * (1 + sym(b, 1, 1, 1) * p["contrast"]) + mean) * (1 + sym(b, 1, 1, 1) * p["brightness"])
    for k in range(b):
        if coin(p["blur"]):                                           # out-of-focus scans
            s = 0.3 + 1.2 * float(torch.rand(()))
            x[k] = TF.gaussian_blur(x[k], [7, 7], [s, s])
    if p["noise"]:
        on = (torch.rand(b, 1, 1, 1, device=dev) < p["noise"]).float()
        x = x + on * torch.randn_like(x) * (0.01 + 0.02 * torch.rand(b, 1, 1, 1, device=dev))
    return x.clamp(0, 1)


def to_input(img_u8, aug=None):
    """uint8 batch on the GPU -> normalised float model input; aug = "light" / "strong" for training."""
    x = img_u8.float() / 255
    if aug:
        x = augment(x, aug)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)
