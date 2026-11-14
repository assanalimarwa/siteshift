"""
Train a U-Net to segment tumour in the prostate tiles.

    python train.py --name resnet34_v2                # ResNet-34 U-Net, current recipe
    python train.py --arch unet --name unet_v2        # original U-Net from scratch
    python train.py --name resnet34_v2 --resume       # continue an interrupted run

Needs tiles.csv and split.json from prepare.py. Trains on the train slides (skipping tiles
that are mostly glass and contain no tumour), scores every tile of the val slides after each
epoch, and keeps the checkpoint with the best val Dice:

    runs/<name>/best.pt  last.pt  log.csv  config.json

Against over-fitting the default recipe uses strong colour / blur / noise augmentation, crops
shifted across tile borders, weight decay 0.01, an exponential moving average of the weights
(validated and saved instead of the raw weights) and early stopping. The first run
(runs/resnet34) used: --aug light --shift 0 --ema 0 --weight-decay 1e-4 --epochs 20 --patience 0

Pretraining on SICAPv2 (after prepare_sicap.py), then the same recipe on our tiles starting
from those weights instead of ImageNet:

    python train.py --name sicap_pretrain --data $SITESHIFT_DATA/sicapv2 --index sicap_tiles.csv
                    --split sicap_split.json --shift 0 --zoom 1.5 2.0
    python train.py --name resnet34_v2_sicap --init runs/sicap_pretrain/best.pt

Several public sources at once (run_all_public.py): --sources <json> lists them, each with its data folder,
index, split and optional zoom (shift is turned off for zoomed sources). Every epoch draws the same number of
tiles from each source (--epoch-tiles / number of sources; a small source repeats), and validation scores
each source's val tiles (--limit-val shared evenly) and selects on the mean of the per-source Dice.

    {"sources": [{"name": "SICAPv2", "data": str(P.DATA_ROOT / "sicapv2"), "index": "sicap_tiles.csv",
                  "split": "sicap_split.json", "zoom": [1.5, 2.0]}, ...]}
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Sampler

from data import AUG, TileDataset, load_rows, to_input
from metrics import loss_fn, predict_counts, summarize
from encoders import ENCODERS
from model import build_model

HERE = P.INDEX
LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_slide_dice",
               "val_tumour_tile_dice", "val_clean_empty", "lr", "minutes"]


class EvenSourceSampler(Sampler):
    """Each epoch the same number of tiles from every part of a ConcatDataset: a random subset of a large
    part, a small part passed through several times (each pass in a new random order), all shuffled together."""

    def __init__(self, sizes, per_epoch):
        self.sizes, self.k = sizes, per_epoch // len(sizes)
        self.offsets = np.cumsum([0] + sizes[:-1]).tolist()

    def __len__(self):
        return self.k * len(self.sizes)

    def __iter__(self):
        parts = [torch.cat([torch.randperm(n) for _ in range(-(-self.k // n))])[:self.k] + off
                 for n, off in zip(self.sizes, self.offsets)]
        idx = torch.cat(parts)
        return iter(idx[torch.randperm(len(idx))].tolist())


def make_loader(dataset, train, a, persistent=True):
    sampler = None
    if train and isinstance(dataset, ConcatDataset):
        sampler = EvenSourceSampler([len(d) for d in dataset.datasets], a.epoch_tiles or len(dataset))
    elif train and a.epoch_tiles and a.epoch_tiles != len(dataset):
        # each epoch epoch_tiles random tiles: a fresh subset of a large set, or a small set passed
        # through several times (each pass in a new random order)
        sampler = RandomSampler(dataset, replacement=False, num_samples=a.epoch_tiles)
    return DataLoader(dataset, batch_size=a.batch, shuffle=train and sampler is None, sampler=sampler, drop_last=train,
                      num_workers=a.workers, pin_memory=True, persistent_workers=persistent and a.workers > 0,
                      prefetch_factor=4 if a.workers else None)


def subset(rows, n, seed):
    return rows.sample(n, random_state=seed).reset_index(drop=True) if 0 < n < len(rows) else rows


def save(obj, path):
    """Write to a temporary file, then swap it in, so a shutdown mid-write never leaves a broken checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def deep_supervision_loss(outputs, y):
    """nnU-Net: the loss of every decoder output (full size first, then 1/2, 1/4, ...) against the nearest-resized
    target, weights 1, 1/2, 1/4, ... normalised to sum 1, the lowest resolution left out."""
    w = [1 / 2 ** i for i in range(len(outputs))]
    w[-1] = 0.0
    total = 0.0
    for wi, o in zip(w, outputs):
        if wi:
            t = y if o.shape[-2:] == y.shape[-2:] else torch.nn.functional.interpolate(y, size=o.shape[-2:], mode="nearest")
            total = total + wi * loss_fn(o.float(), t)
    return total / sum(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(str(P.DATASET47)))
    ap.add_argument("--index", type=Path, default=HERE / "tiles.csv")
    ap.add_argument("--split", type=Path, default=HERE / "split.json")
    ap.add_argument("--sources", type=Path,
                    help="JSON list of training sources (replaces --data / --index / --split / --zoom; see above)")
    ap.add_argument("--arch", choices=["resnet34", "unet", "segformer", "transunet", "nnunet"] + [f"fm_{e}" for e in ENCODERS],
                    default="resnet34", help="fm_*: frozen foundation-model encoder + the same trainable head (model.py)")
    ap.add_argument("--name", help="run folder under runs/ (default: the arch)")
    ap.add_argument("--epochs", type=int, default=40, help="maximum; early stopping usually ends sooner")
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many epochs without a better val Dice (0 = never)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500, help="steps of linear learning-rate warm-up")
    ap.add_argument("--aug", choices=list(AUG), default="strong")
    ap.add_argument("--shift", type=int, default=256,
                    help="train on windows shifted up to this many pixels across tile borders (0 = tiles as they are)")
    ap.add_argument("--ema", type=float, default=0.999, help="weight EMA decay (0 = off)")
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("LO", "HI"),
                    help="enlarge tiles by a random factor in [LO, HI] (SICAPv2: 1.5 2.0; needs --shift 0)")
    ap.add_argument("--init", type=Path,
                    help="start from the weights of this checkpoint (e.g. runs/sicap_pretrain/best.pt) instead of ImageNet")
    ap.add_argument("--min-tissue", type=float, default=0.1,
                    help="train only: skip tiles with less tissue than this unless they contain tumour")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-train", type=int, default=0, help="random subset of train tiles (quick tests)")
    ap.add_argument("--epoch-tiles", type=int, default=0,
                    help="tiles per epoch, drawn at random from all train tiles each epoch; more than the train set "
                         "repeats it (0 = every tile, every epoch)")
    ap.add_argument("--keep", choices=["best", "last"], default="best",
                    help="weights saved as best.pt: the epoch with the best val Dice, or the last epoch "
                         "(no model selection on val labels; val is then only logged)")
    ap.add_argument("--limit-val", type=int, default=0, help="random subset of val tiles (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", choices=["default", "nnunet"], default="default",
                    help="nnunet: the nnU-Net optimiser and schedule (SGD, momentum 0.99 Nesterov, poly 0.9, no warm-up) and "
                         "deep supervision; use with --arch nnunet --lr 0.01 --weight-decay 3e-5 --ema 0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mask-root", type=Path, help="read train and val masks from this folder (<slide>/masks/<tile>.png, "
                    "e.g. labels47.py's transformed labels) instead of --data; single source only")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu: only for tests (bfloat16 autocast is off)")
    a = ap.parse_args()

    out = P.RUNS / (a.name or a.arch)
    if (out / "last.pt").exists() and not a.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume or a new --name")
    if a.zoom and a.shift:
        raise SystemExit("--zoom works on single tiles; pass --shift 0 with it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(a.device)
    if a.device == "cuda" and os.environ.get("QP_GPU_MEM_GB"):                       # shared GPU: never take more than this (queue.py sets it)
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["QP_GPU_MEM_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    specs = json.loads(a.sources.read_text())["sources"] if a.sources else \
        [dict(name="", data=a.data, index=a.index, split=a.split, zoom=a.zoom)]
    many = len(specs) > 1
    if a.mask_root and many:
        raise SystemExit("--mask-root works with one source only")
    train_sets, vals, n_train, n_val = [], [], 0, 0
    for sp in specs:
        index, split, zoom = HERE / sp["index"], HERE / sp["split"], sp.get("zoom")
        all_train = load_rows(index, split, "train")           # every tile, for neighbours of shifted crops
        train_rows = subset(load_rows(index, split, "train", min_tissue=a.min_tissue), a.limit_train, a.seed)
        val_rows = subset(load_rows(index, split, "val"), a.limit_val // len(specs), a.seed)
        print(f"{sp['name'] + ' ' if many else ''}train: {len(train_rows)} tiles from {train_rows.slide.nunique()} slides | "
              f"val: {len(val_rows)} tiles from {val_rows.slide.nunique()} slides", flush=True)
        train_sets.append(TileDataset(sp["data"], train_rows, train=True, shift=0 if zoom and many else a.shift,
                                      tiles=all_train, zoom=zoom, mask_root=a.mask_root))
        # several sources: val workers are not kept between epochs (one set of persistent workers per source
        # would exceed the Windows commit limit, unet-pipeline notes)
        vals.append((sp["name"], val_rows, make_loader(TileDataset(sp["data"], val_rows, zoom=zoom, mask_root=a.mask_root), False, a, persistent=not many)))
        n_train, n_val = n_train + len(train_rows), n_val + len(val_rows)
    train_dl = make_loader(ConcatDataset(train_sets) if many else train_sets[0], True, a)

    model = build_model(a.arch)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        if ck.get("arch", a.arch) != a.arch:
            raise SystemExit(f"{a.init} holds a {ck['arch']} model, not {a.arch}")
        model.load_state_dict(ck["model"])
        print(f"weights from {a.init} (epoch {ck.get('epoch')}, its val Dice {ck['val']['dice']:.4f})", flush=True)
    model = model.to(dev, memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True) if a.ema else None
    scored = ema.module if ema else model                      # the weights that are validated and saved
    total = a.epochs * len(train_dl)
    if a.recipe == "nnunet":
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, momentum=0.99, nesterov=True)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - min(s, total - 1) / total) ** 0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then cosine down to 1% of --lr
            opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))

    start, best, since_best = 0, -1.0, 0
    if a.resume:
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema:
            ema.load_state_dict(ck["ema"])
        start, best, since_best = ck["epoch"] + 1, ck["best"], ck.get("since_best", 0)
        print(f"resumed after epoch {start}, best val Dice so far {best:.4f}")
    import importlib.metadata as _md, platform as _pf
    versions = dict(machine=os.environ.get("QP_MACHINE", "unknown"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu_mem_cap_gb=os.environ.get("QP_GPU_MEM_GB"), python=_pf.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for _p in ("torchvision", "numpy", "pandas", "pillow", "scipy", "timm", "transformers", "huggingface_hub"):
        try:
            versions[_p] = _md.version(_p)
        except Exception:
            pass
    (out / "config.json").write_text(json.dumps(
        dict(vars(a), train_tiles=n_train, val_tiles=n_val, sources=specs, versions=versions), indent=2, default=str))
    log = out / "log.csv"
    if not log.exists():
        log.write_text(",".join(LOG_COLUMNS + ([f"val_dice_{name}" for name, _, _ in vals] if many else [])) + "\n")

    for epoch in range(start, a.epochs):
        model.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        t_last, seen_last = t0, 0
        for step, (img, mask, _) in enumerate(train_dl, 1):
            x = to_input(img.to(dev, non_blocking=True), aug=a.aug)
            y = mask.to(dev, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                logits = model(x)
            loss = deep_supervision_loss(logits, y) if isinstance(logits, (list, tuple)) else loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema:
                ema.update_parameters(model)
            loss_sum += loss.item() * len(y)
            seen += len(y)
            if step % 200 == 0 or step == len(train_dl):
                now = time.time()
                print(f"epoch {epoch + 1}/{a.epochs}  step {step}/{len(train_dl)}  loss {loss_sum / seen:.4f}  "
                      f"{(seen - seen_last) / (now - t_last):.0f} img/s", flush=True)
                t_last, seen_last = now, seen

        each = {}
        for name, val_rows, val_dl in vals:
            counts, loss_k = predict_counts(scored, val_dl, dev)
            each[name], _, _ = summarize(val_rows, counts, tile_pixels=val_dl.dataset.size ** 2)
            each[name]["loss"] = loss_k
        if many:                                               # every source weighs the same, as in training
            s = {k: float(np.nanmean([e[k] for e in each.values()])) for k in next(iter(each.values()))}
            s["per_source"] = each
        else:
            s = next(iter(each.values()))
        val_loss = s.pop("loss")
        minutes = (time.time() - t0) / 60
        row = [epoch + 1, loss_sum / seen, val_loss, s["dice"], s["iou"], s["slide_dice"],
               s["tumour_tile_dice"], s["clean_empty_tiles"], sched.get_last_lr()[0], minutes] + \
              ([e["dice"] for e in each.values()] if many else [])
        with log.open("a") as f:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        improved = s["dice"] > best
        since_best = 0 if improved else since_best + 1
        print(f"epoch {epoch + 1} done in {minutes:.1f} min | train loss {loss_sum / seen:.4f} | val loss {val_loss:.4f}  "
              f"Dice {s['dice']:.4f}  IoU {s['iou']:.4f}  slide Dice {s['slide_dice']:.4f}  "
              f"tumour-tile Dice {s['tumour_tile_dice']:.4f}  clean empty {s['clean_empty_tiles']:.3f}"
              + ("  (" + ", ".join(f"{k} {e['dice']:.4f}" for k, e in each.items()) + ")" if many else "")
              + ("  <- best" if improved else ""), flush=True)
        if improved:
            best = s["dice"]
        if improved or a.keep == "last":
            save(dict(model=scored.state_dict(), arch=a.arch, epoch=epoch + 1, val=s), out / "best.pt")
        save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  ema=ema.state_dict() if ema else None, epoch=epoch, best=best, since_best=since_best,
                  arch=a.arch), out / "last.pt")
        if a.patience and since_best >= a.patience:
            print(f"early stop: no better val Dice for {a.patience} epochs")
            break

    print(f"finished: best val Dice {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
