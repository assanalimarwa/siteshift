"""Loss and segmentation metrics shared by train.py and evaluate.py."""

import numpy as np
import torch
import torch.nn.functional as F

from data import to_input


def loss_fn(logits, target):
    """BCE plus soft Dice over the whole batch (per-tile Dice is undefined on tumour-free tiles).
    Unlabelled pixels (target 2, data.load_mask) count in neither term."""
    w = (target < 2).float()
    target = target * w
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum().clamp_min(1)
    p = torch.sigmoid(logits) * w
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def tissue_mask(img_u8):
    """Tissue pixels of a uint8 (B,3,H,W) batch, as prepare.tissue_mask: neither glass (bright)
    nor padding (dark), and coloured. Returns (B,1,H,W) bool."""
    a = img_u8.int()
    mean = a.sum(1, keepdim=True) / 3
    sat = a.amax(1, keepdim=True) - a.amin(1, keepdim=True)
    return (mean > 25) & (mean < 220) & (sat > 12)


def pixel_counts(pred, gt):
    """(B, 4) tp, fp, fn, tumour pixels per image from boolean (B,1,H,W) masks."""
    return torch.stack([(pred & gt).sum((1, 2, 3)), (pred & ~gt).sum((1, 2, 3)),
                        (~pred & gt).sum((1, 2, 3)), gt.sum((1, 2, 3))], 1)


@torch.no_grad()
def predict_counts(model, loader, device):
    """Pixel counts per tile at threshold 0.5, in dataset order: columns tp, fp, fn, tumour.
    Also returns the mean loss."""
    model.eval()
    counts = np.zeros((len(loader.dataset), 4), np.int64)
    loss_sum, n = 0.0, 0
    for img, mask, idx in loader:
        x = to_input(img.to(device, non_blocking=True))
        y = mask.to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss_sum += loss_fn(logits, y).item() * len(idx)
        n += len(idx)
        labelled = y < 2
        counts[idx.numpy()] = pixel_counts((logits > 0) & labelled, y == 1).cpu().numpy()
    return counts, loss_sum / max(n, 1)


def dice(tp, fp, fn):
    """Dice from pixel counts; 1 when neither truth nor prediction has tumour."""
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 1.0)


def summarize(rows, counts, tile_pixels=512 * 512):
    """Split-level metrics, a per-slide table and a per-tile table.

    dice / iou          over all pixels of the split (the main number)
    slide_dice          mean of per-slide Dice, so big slides do not dominate
    tumour_tile_dice    mean per-tile Dice over tiles that contain tumour
    clean_empty_tiles   share of tumour-free tiles with < 1% of pixels predicted tumour
    """
    t = rows[["slide", "image"]].reset_index(drop=True).copy()
    t[["tp", "fp", "fn", "tumour_px"]] = counts
    t["dice"] = dice(t.tp, t.fp, t.fn)
    s = t.groupby("slide")[["tp", "fp", "fn", "tumour_px"]].sum()
    s.insert(0, "tiles", t.groupby("slide").size())
    s["dice"] = dice(s.tp, s.fp, s.fn)
    tp, fp, fn = t.tp.sum(), t.fp.sum(), t.fn.sum()
    empty = t[t.tumour_px == 0]
    summary = dict(
        tiles=len(t),
        dice=float(dice(tp, fp, fn)),
        iou=float(tp / max(tp + fp + fn, 1)),
        slide_dice=float(s.dice.mean()),
        tumour_tile_dice=float(t.dice[t.tumour_px > 0].mean()),
        clean_empty_tiles=float((empty.fp < 0.01 * tile_pixels).mean()) if len(empty) else float("nan"),
    )
    return summary, s.reset_index(), t
