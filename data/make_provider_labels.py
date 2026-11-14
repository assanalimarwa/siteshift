"""
Split the PANDA label file by provider and add the column the converters read.

    python data/make_provider_labels.py --panda <folder with train.csv, train_images/, train_label_masks/> --out $SITESHIFT_RAW/prostate-cancer-grade-assessment

Writes <out>/Karolinska/labels.csv and <out>/Panda/labels.csv (the Radboud provider; folder names as the converters expect), each with the columns of
train.csv plus `has_mask` (1 if train_label_masks/<id>_mask.tiff exists), and links the images and masks next to them:
    <out>/<Provider>/images/<id>.tiff        <out>/<Provider>/masks/<id>_mask.tiff
(links are hard links, or copies if the folders are on different drives; the PANDA files themselves are never modified).
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

PROVIDERS = {"karolinska": "Karolinska", "radboud": "Panda"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.panda / "train.csv")
    for prov, folder in PROVIDERS.items():
        d = df[df.data_provider == prov].copy()
        d["has_mask"] = [int((a.panda / "train_label_masks" / f"{i}_mask.tiff").exists()) for i in d.image_id]
        dst = a.out / folder
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "masks").mkdir(exist_ok=True)
        d.to_csv(dst / "labels.csv", index=False)
        for i, has in zip(d.image_id, d.has_mask):
            for src, to in ((a.panda / "train_images" / f"{i}.tiff", dst / "images" / f"{i}.tiff"),
                            (a.panda / "train_label_masks" / f"{i}_mask.tiff", dst / "masks" / f"{i}_mask.tiff")):
                if src.exists() and not to.exists():
                    try:
                        os.link(src, to)
                    except OSError:
                        shutil.copy2(src, to)
        print(f"{prov}: {len(d)} slides, {int(d.has_mask.sum())} with a mask -> {dst}")


if __name__ == "__main__":
    main()
