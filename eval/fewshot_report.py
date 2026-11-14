"""
cv/fewshot.md: few-shot adaptation to our site (run_fewshot.py first).

    python fewshot_report.py

Our 17 patients in tile mode, each scored by the fold models that never saw it. Rows: the all-Karolinska
U-Net as it is (n = 0); fine-tuned on n of our patients; the same n patients from ImageNet; and our CV
models (ImageNet, all train patients, early stopping on the val patients) as the reference. Every
few-shot run has the same fixed budget and keeps its last weights (no selection on labels). Mean over
patients [95% bootstrap CI]; paired differences on the same patients. Rows with fewer than 17 patients
are incomplete (folds still running).
"""

import json

import pandas as pd

from karolinska_all_report import HERE, RUNS, paired, per_unit, table
from run_fewshot import SIZES


def main():
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    tissue = pd.read_csv(HERE / "tiles.csv")[["image", "tissue_frac"]]
    src = {"Karolinska as it is (n = 0)": [RUNS / "karolinska_all_unet" / "eval_all_test_tile" / "per_tile.csv"]}
    sizes = []
    for n in SIZES:
        for init, label in [("kar", "Karolinska"), ("imagenet", "ImageNet")]:
            runs = [RUNS / "fewshot" / init / f"n{n}" / f"fold{k}" for k in range(5)]
            src[f"{label} + {n} of our patients"] = [r / f"eval_fold{k}_n{n}_test_tile" / "per_tile.csv" for k, r in enumerate(runs)]
            if init == "kar":
                tiles = [json.loads((r / "config.json").read_text())["train_tiles"] for r in runs if (r / "config.json").exists()]
                if tiles:
                    sizes.append({"patients": n, "train tiles per fold": ", ".join(map(str, tiles))})
    src["ours (CV): ImageNet, all train patients, early stopping"] = [
        RUNS / "cv" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_tile" / "per_tile.csv" for k in range(5)]
    found = {}
    for name, paths in src.items():
        paths = [p for p in paths if p.exists()]
        if paths:
            found[name] = per_unit(paths, "patient", tissue)

    pairs = []
    for n in SIZES:
        k, i = f"Karolinska + {n} of our patients", f"ImageNet + {n} of our patients"
        pairs += [(k, i), (k, "Karolinska as it is (n = 0)"), ("ours (CV): ImageNet, all train patients, early stopping", k)]
    lines = ["# Few-shot adaptation: the all-Karolinska U-Net fine-tuned on a few of our patients", "",
             "Tile mode, our 17 patients (each scored only by models that never saw it). Fixed budget per run: "
             "100,000 tile draws, last weights kept.", "",
             table(found, biopsy), "", "Training tiles (the n patients; nested draws per fold):", "",
             pd.DataFrame(sizes).to_markdown(index=False) if sizes else "", "",
             "Paired differences on the same patients:", "",
             paired(found, pairs, metrics=("dice", "false_tumour", "clean_auc")), ""]
    (HERE / "cv" / "fewshot.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
