"""
End-to-end test on a tiny synthetic dataset (2 fake slides, 2 fake patients); needs no patient data and no GPU:

    python tests/e2e_test.py [--keep]

It drives the scripts of this repository, as they sit in this layout, through every stage that produces a number of the paper:

  1  fake QuPath export (tile PNGs + "-labelled" masks, the names the real export has)
  2  data/make_slide_folders.py   export -> dataset folders <slide>/images, <slide>/masks   (dataset layout)
  3  data/prepare.py --index-only  tile index (slide, image, tissue_frac, tumour_frac)
  4  train/train.py                one optimisation step (nnU-Net-style network, CPU)
  5  eval/evaluate.py              tile-mode evaluation of the checkpoint -> per_slide.csv
  6  eval/protocol_predict.py      protocol-format predictions (.npz)
  7  eval/protocol.py              detection / delineation analysis of the predictions; checks the identity
                                   missed + fill + boundary(fn) = FN and false + boundary(fp) = FP, and that the Dice equals evaluate.py's
  8  eval/run_eval.py              scoring the same predictions from a submission folder of PNG masks: every pixel sum equals step 7
  9  data/make_provider_labels.py + data/prepare_karolinska.py: a synthetic PANDA-format TIFF pair cut into 512-px tiles (the public-data tiler;
     skipped if tifffile / imagecodecs are not installed)

Everything happens in a temporary folder that is removed at the end (--keep leaves it). Exit code 0 = all stages passed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TILE = 512
fails = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def run(env, script, *args):
    t0 = time.time()
    r = subprocess.run([PY, str(ROOT / script), *map(str, args)], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-2500:])
    check(r.returncode == 0, f"{script} ({time.time() - t0:.0f} s)")
    if r.returncode:
        raise SystemExit(1)
    return r.stdout


def synthetic_export(root, rng):
    """Slide 1: 2 x 2 tiles, slide 2: 3 x 3 tiles of 512 px: pink tissue, a darker 'tumour' disc that crosses tile borders; masks 0 / 255.
    Slide 2 has tumour-free tiles and a tile that is mostly tumour, so the clean-tile AUC exists."""
    root.mkdir(parents=True)
    for slide, (g, (cy, cx, r)) in {"1.1.bif - Series 0": (2, (560, 480, 330)), "2.1.bif - Series 0": (3, (400, 400, 360))}.items():
        d = root / slide
        d.mkdir()
        yy, xx = np.mgrid[:g * TILE, :g * TILE]
        tumour = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        noise = rng.integers(-12, 13, size=(g * TILE, g * TILE, 3))
        rgb = np.where(tumour[..., None], [120, 60, 150], [225, 150, 190]) + noise
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        for ty in range(g):
            for tx in range(g):
                name = f"{slide} [d=2,x={tx * 1024},y={ty * 1024},w=1024,h=1024]"
                sl = (slice(ty * TILE, (ty + 1) * TILE), slice(tx * TILE, (tx + 1) * TILE))
                Image.fromarray(rgb[sl]).save(d / f"{name}.png")
                Image.fromarray((tumour[sl] * 255).astype(np.uint8)).save(d / f"{name}-labelled.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="siteshift_e2e_"))
    print("working folder:", tmp)
    env = dict(os.environ, SITESHIFT_DATA=str(tmp / "datasets"), SITESHIFT_RAW=str(tmp / "raw"), SITESHIFT_RUNS=str(tmp / "runs"),
               SITESHIFT_PRED=str(tmp / "predictions"), SITESHIFT_RESULTS=str(tmp / "results"), SITESHIFT_OUT=str(tmp / "out"),
               SITESHIFT_INDEX=str(tmp / "index"), PYTHONPATH=os.pathsep.join(str(ROOT / d) for d in (".", "data", "train", "eval", "figures")),
               OMP_NUM_THREADS="2")
    (tmp / "index").mkdir()
    rng = np.random.default_rng(0)
    data = tmp / "datasets" / "dataset47"

    # 1 + 2: export -> dataset layout
    synthetic_export(tmp / "export", rng)
    run(env, "data/make_slide_folders.py", "--export", tmp / "export", "--out", data, "--workers", 2)
    n_img = len(list(data.glob("*/images/*.png")))
    n_msk = len(list(data.glob("*/masks/*.png")))
    check(n_img == 13 and n_msk == 13, f"dataset layout: {n_img} images and {n_msk} masks in 2 slide folders")

    # 3: tile index
    index = tmp / "index" / "tiles47.csv"      # the file name protocol.py maps to our patients
    run(env, "data/prepare.py", "--data", data, "--index", index, "--index-only", "--workers", 2)
    import pandas as pd
    t = pd.read_csv(index)
    check(len(t) == 13 and t.slide.nunique() == 2 and (t.tumour_frac > 0).sum() >= 8 and (t.tumour_frac == 0).sum() >= 2 and (t.tissue_frac > 0.9).all(),
          f"tile index: {len(t)} tiles, 2 slides, tissue and tumour fractions measured")

    # split: patient 1 trains, patient 2 validates and tests (a synthetic toy; the real folds are data/cv47/fold<k>.json)
    split = tmp / "index" / "toy_split.json"
    split.write_text(json.dumps({"splits": {"train": ["1.1.bif - Series 0"], "val": ["2.1.bif - Series 0"], "test": ["2.1.bif - Series 0"]}}))

    # 4: one training step
    out = run(env, "train/train.py", "--name", "e2e", "--arch", "nnunet", "--recipe", "nnunet", "--lr", "0.01", "--weight-decay", "3e-5", "--ema", "0",
              "--epochs", "1", "--epoch-tiles", "2", "--batch", "2", "--limit-val", "2", "--workers", "0", "--shift", "0", "--patience", "0",
              "--keep", "last", "--warmup", "0", "--device", "cpu", "--data", data, "--index", index, "--split", split, "--min-tissue", "0")
    ck = tmp / "runs" / "e2e" / "best.pt"
    check(ck.exists() and (tmp / "runs" / "e2e" / "config.json").exists() and (tmp / "runs" / "e2e" / "log.csv").exists(),
          "train.py: one epoch of one step wrote best.pt, config.json, log.csv")

    # 5: tile-mode evaluation
    run(env, "eval/evaluate.py", "--name", "e2e", "--split-file", split, "--data", data, "--index", index, "--context", "0", "--no-tta", "--no-pictures",
        "--workers", "0", "--device", "cpu")
    ev = next((tmp / "runs" / "e2e").glob("eval_*_test_tile"))
    ps = pd.read_csv(ev / "per_slide.csv")
    check(len(ps) == 1 and (ev / "summary.json").exists(), "evaluate.py: tile-mode per_slide.csv and summary.json for the test slide")

    # 6: protocol-format predictions
    pred = tmp / "predictions" / "e2e" / "predictions" / "test"
    run(env, "eval/protocol_predict.py", "--ckpt", ck, "--out", pred, "--data", data, "--index", index, "--split-file", split, "--split", "test",
        "--context", "0", "--no-tta", "--device", "cpu", "--workers", "0")
    check((pred / "done.json").exists() and len(list(pred.glob("*.npz"))) == 1, "protocol_predict.py: one .npz per slide and done.json")

    # 7: protocol analysis (in this process, same folders)
    for k, v in env.items():
        if k.startswith("SITESHIFT"):
            os.environ[k] = v
    sys.path[:0] = [str(ROOT)] + [str(ROOT / d) for d in ("data", "train", "eval")]
    import protocol
    protocol.INDEX["ours"] = "tiles47.csv"      # as eval/protocol47.py does: protocol.py names the pilot index by default
    slides, tiles = protocol.analyse(pred, per_tile=False, workers=1)
    s = slides.iloc[0]
    check(s.missed + s.fill_px + s.fn_band == s.fn and s["false"] + s.fp_band == s.fp, "protocol.py: missed + fill + boundary(fn) = FN and false + boundary(fp) = FP, exactly")
    dice_protocol = 2 * s.tp / max(2 * s.tp + s.fp + s.fn, 1)
    dice_eval = float(ps.dice.iloc[0]) if "dice" in ps else float("nan")
    check(abs(dice_protocol - dice_eval) < 1e-6, f"slide Dice of the protocol analysis ({dice_protocol:.4f}) equals evaluate.py's ({dice_eval:.4f})")
    tissue = t[["image", "tissue_frac"]]
    m = protocol.measures(slides, tiles, "ours", tissue)
    check(len(m) == 1 and np.isfinite(m.dice.iloc[0]), "protocol.measures: per-patient row")

    # 8: submission folder of PNG masks -> run_eval.py
    sub = tmp / "submission"
    z = np.load(next(pred.glob("*.npz")))
    S = int(z["size"])
    for i, img in enumerate(z["image"]):
        mk = np.unpackbits(z["pred"][i], count=S * S).reshape(S, S).astype(np.uint8) * 255
        dst = sub / Path(str(img)).parts[0] / (Path(str(img)).stem + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mk).save(dst)
    run(env, "eval/run_eval.py", "--pred", sub, "--data", data, "--index", index, "--split-file", split, "--split", "test", "--out", tmp / "scores", "--workers", "1")
    a2 = pd.read_csv(tmp / "scores_per_slide.csv").set_index("slide")
    cols = [c for c in a2.columns if c in slides.columns and c != "slide"]
    diff = float((a2[cols].iloc[0] - slides.set_index("slide")[cols].iloc[0]).abs().max())
    check(diff == 0, f"run_eval.py from PNG masks reproduces every pixel sum of the protocol analysis ({len(cols)} sums, max difference {diff})")

    # 9: the public-data path: PANDA-format synthetic TIFFs -> make_provider_labels.py -> prepare_karolinska.py (the tiler of the public data)
    try:
        import imagecodecs  # noqa: F401
        import tifffile
    except Exception:
        print("skip  public-data tiler (pip install tifffile imagecodecs to run stage 9)")
    else:
        panda = tmp / "panda"
        (panda / "train_images").mkdir(parents=True)
        (panda / "train_label_masks").mkdir()
        H = W = 1024
        yy, xx = np.mgrid[:H, :W]
        tum = (yy - 500) ** 2 + (xx - 520) ** 2 <= 250 ** 2
        img = np.clip(np.where(tum[..., None], [120, 60, 150], [225, 150, 190]) + rng.integers(-8, 9, (H, W, 3)), 0, 255).astype(np.uint8)
        lab = np.zeros((H, W, 3), np.uint8)
        lab[..., 0] = np.where(tum, 2, 1)                       # red channel: 1 benign tissue, 2 cancer (PANDA Karolinska masks)
        for path, arr, comp in ((panda / "train_images" / "fake1.tiff", img, "jpeg"), (panda / "train_label_masks" / "fake1_mask.tiff", lab, "zlib")):
            with tifffile.TiffWriter(path) as tf:
                tf.write(arr, tile=(512, 512), compression=comp, photometric="rgb", subifds=2, metadata=None)
                for lv in (1, 2):
                    tf.write(np.ascontiguousarray(arr[::4 ** lv, ::4 ** lv]), tile=(16, 16), compression=comp, photometric="rgb", subfiletype=1, metadata=None)
        (panda / "train.csv").write_text("image_id,data_provider,isup_grade,gleason_score\nfake1,karolinska,1,3+3\nfake2,radboud,0,negative\n")
        run(env, "data/make_provider_labels.py", "--panda", panda, "--out", tmp / "raw" / "prostate-cancer-grade-assessment")
        run(env, "data/prepare_karolinska.py", "--slides", 1, "--workers", 1, "--force")
        tiles = list((tmp / "datasets" / "karolinska").glob("*/images/*.jpg"))
        ki = pd.read_csv(tmp / "index" / "karolinska_tiles.csv")
        check(len(tiles) == 4 and len(ki) == 4 and (ki.tumour_frac > 0).sum() >= 3,
              f"prepare_karolinska.py cut the synthetic PANDA TIFF into {len(tiles)} tiles of 512 px (index: {len(ki)} rows, {(ki.tumour_frac > 0).sum()} with cancer)")

    print("\n" + ("ALL STAGES PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
    if not a.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
