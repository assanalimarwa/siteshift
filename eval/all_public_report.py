"""
cv/all_public.md: all public data combined, tested on our patients (run_all_public.py first).

    python all_public_report.py

Our 17 patients, unit = patient, each scored only by models that never saw it. Per patient: Dice, sensitivity,
false-tumour rate, tile AUC, clean tile AUC (karolinska_all_report.per_unit), and boundary-tolerant Dice (Dice
leaving out every pixel within 50 um of the annotated outline, evaluate.py --tolerant); per specimen type
(biopsies / large sections, cv/folds.json). Mean over patients [95% bootstrap CI]; paired differences on the
same patients. The plain-language answers at the top come from cv/all_public/answers.md (written after
reading the tables) when it exists.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cv_summary import bootstrap_ci
from karolinska_all_report import HERE, RUNS, paired, per_unit
from run_all_public import CFG, FEWSHOT_N, N_MIX

TOL = ["tp_st", "fp_st", "fn_st"]
METRICS = ["dice", "sensitivity", "false_tumour", "clean_auc", "dice_tolerant"]
SOURCE_NOTES = {
    "SICAPv2": ("grey level >= 25 -> tumour, else not (prepare_sicap.py)", "~1 um/px, enlarged x1.5-2.0 at random (eval x1.75)"),
    "Karolinska": ("2 cancer -> tumour; 1 benign and 0 background -> not tumour", "0.503 um/px, as is"),
    "Radboud": ("3-5 Gleason -> tumour; 1 stroma, 2 benign epithelium -> not; 0 background/unknown -> ignored",
                "0.486 um/px, as is"),
}


def scored(paths, tissue):
    x = per_unit(paths, "patient", tissue)
    t = pd.concat([pd.read_csv(p) for p in paths])
    if set(TOL) <= set(t.columns):
        g = t.groupby(t.slide.str.split(".").str[0].astype(int))[TOL].sum()
        d = 2 * g.tp_st + g.fp_st + g.fn_st
        x["dice_tolerant"] = (2 * g.tp_st / d.clip(lower=1)).where(d > 0)
    return x


def cell(v):
    v = v.dropna()
    if not len(v):
        return ""
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def table(found, biopsy):
    rows = []
    for name, x in found.items():
        r = {"model": name, "patients": len(x), "Dice": cell(x.dice), "sensitivity": cell(x.sensitivity),
             "false-tumour rate": cell(x.false_tumour), "clean tile AUC": cell(x.clean_auc),
             "boundary-tolerant Dice": cell(x.dice_tolerant) if "dice_tolerant" in x else ""}
        rows.append(r)
    return pd.DataFrame(rows).to_markdown(index=False)


def specimen_table(found, biopsy):
    rows = []
    for name, x in found.items():
        for kind, sel in [("biopsies", [p for p in x.index if p in biopsy]), ("large sections", [p for p in x.index if p not in biopsy])]:
            y = x.loc[sel]
            rows.append({"model": name, "specimen": kind, "patients": len(y), "Dice": f"{y.dice.mean():.3f}",
                         "sensitivity": f"{y.sensitivity.mean():.3f}", "false-tumour rate": f"{y.false_tumour.mean():.3f}",
                         "clean tile AUC": f"{y.clean_auc.mean():.3f}",
                         "boundary-tolerant Dice": f"{y.dice_tolerant.mean():.3f}" if "dice_tolerant" in y else ""})
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def paired_by_type(found, pairs, biopsy, metric="dice"):
    rows = []
    for a, b in pairs:
        if a in found and b in found:
            for kind, keep in [("biopsies", lambda p: p in biopsy), ("large sections", lambda p: p not in biopsy)]:
                d = (found[a][metric] - found[b][metric]).dropna()
                d = d[[keep(p) for p in d.index]]
                lo, hi = bootstrap_ci(d)
                rows.append({"difference": f"{a} minus {b}", "specimen": kind, "metric": metric, "patients": len(d),
                             "mean [95% CI]": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", "higher / lower": f"{(d > 0).sum()} / {(d < 0).sum()}"})
    return pd.DataFrame(rows).to_markdown(index=False) if rows else ""


def folder_gb(path, cache):
    if str(path) not in cache:
        total = 0
        for root, _, files in os.walk(path):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        cache[str(path)] = total / 2 ** 30
    return cache[str(path)]


def source_table():
    spec = json.loads((CFG / "sources_A.json").read_text())["sources"]
    cache_file = CFG / "disk_gb.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rows = []
    for s in spec:
        tiles = pd.read_csv(HERE / s["index"])
        split = json.loads((HERE / s["split"]).read_text())["splits"]
        r = {"source": s["name"]}
        for k in ["train", "val"]:
            d = tiles[tiles.slide.isin(split[k])]
            r[f"{k} slides"] = d.slide.nunique()
            r[f"{k} tiles"] = len(d)
        usable = tiles[tiles.slide.isin(split["train"]) & ((tiles.tissue_frac >= 0.1) | (tiles.tumour_frac > 0))]
        r["train tiles used (>= 10% tissue or tumour)"] = len(usable)
        r["tumour pixels"] = f"{tiles.tumour_frac.mean():.1%}"
        r["disk (GB, all tiles on disk)"] = f"{folder_gb(Path(s['data']), cache):.1f}"
        r["label mapping"], r["scale"] = SOURCE_NOTES[s["name"]]
        rows.append(r)
    cache_file.write_text(json.dumps(cache, indent=1))
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def run_note(name):
    run = RUNS / name
    if not (run / "log.csv").exists():
        return f"- {name}: not trained yet"
    log = pd.read_csv(run / "log.csv")
    best = log.loc[log.val_dice.idxmax()]
    extra = ", ".join(f"{c[9:]} {best[c]:.3f}" for c in log.columns if c.startswith("val_dice_"))
    cfg = json.loads((run / "config.json").read_text())
    return (f"- {name}: {len(log)} epochs run, best val Dice {best.val_dice:.3f} at epoch {int(best.epoch)}"
            + (f" (per source: {extra})" if extra else "") + f"; {cfg.get('train_tiles')} train tiles, {cfg.get('epoch_tiles')} draws per epoch")


def models(stage, mode):
    """name -> per_tile.csv paths on our 17 patients (tolerant evaluations)."""
    s = "unet" if stage == "unet" else "uni"
    m = f"{mode}_tol"
    if stage == "unet":
        out = {"ours (CV)": [RUNS / "cv" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_{m}" / "per_tile.csv" for k in range(5)],
               "all-Karolinska (3,814 slides)": [RUNS / "karolinska_all_unet" / f"eval_all_test_{m}" / "per_tile.csv"],
               "SICAPv2": [RUNS / "sicap_pretrain" / f"eval_all_test_{m}" / "per_tile.csv"]}
    else:
        out = {"ours (CV)": [RUNS / "cv" / "fm_uni" / f"fold{k}" / f"eval_fold{k}_test_{m}" / "per_tile.csv" for k in range(5)],
               "SICAPv2": [RUNS / "sicap_fm_uni" / f"eval_all_test_{m}" / "per_tile.csv"]}
    out.update({"A: all public": [RUNS / f"all_public_{s}" / f"eval_all_test_{m}" / "per_tile.csv"],
                f"B: mixed {N_MIX} slides": [RUNS / f"mixed{N_MIX}_{s}" / f"eval_all_test_{m}" / "per_tile.csv"],
                f"B: Karolinska {N_MIX} slides": [RUNS / f"karolinska_n{N_MIX}_{s}" / f"eval_all_test_{m}" / "per_tile.csv"]})
    # a baseline not yet re-scored with --tolerant: its earlier evaluation (same counts, no boundary-tolerant Dice)
    for name, paths in out.items():
        old = [p.parent.with_name(p.parent.name[:-len("_tol")]) / p.name for p in paths]
        if not all(p.exists() for p in paths) and all(p.exists() for p in old):
            out[name] = old
    return out


def fewshot(stage, tissue, biopsy):
    pre = "" if stage == "unet" else "uni_"
    if stage == "unet":
        src = {"A as it is (n = 0)": [RUNS / "all_public_unet" / "eval_all_test_tile_tol" / "per_tile.csv"],
               "all-Karolinska as it is (n = 0)": [RUNS / "karolinska_all_unet" / "eval_all_test_tile" / "per_tile.csv"]}
        inits = [("public", "A"), ("kar", "all-Karolinska"), ("imagenet", "ImageNet")]
        ref = [RUNS / "cv" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_tile" / "per_tile.csv" for k in range(5)]
    else:
        src = {"A as it is (n = 0)": [RUNS / "all_public_uni" / "eval_all_test_tile_tol" / "per_tile.csv"]}
        inits = [("public", "A"), ("scratch", "fresh head")]
        ref = [RUNS / "cv" / "fm_uni" / f"fold{k}" / f"eval_fold{k}_test_tile" / "per_tile.csv" for k in range(5)]
    for n in FEWSHOT_N:
        for init, label in inits:
            src[f"{label} + {n} of our patients"] = [RUNS / "fewshot" / f"{pre}{init}" / f"n{n}" / f"fold{k}" / f"eval_fold{k}_n{n}_test_tile" / "per_tile.csv"
                                                     for k in range(5)]
    src["ours (CV): all train patients, early stopping"] = ref
    found = {k: scored([p for p in v if p.exists()], tissue) for k, v in src.items() if any(p.exists() for p in v)}
    pairs = []
    for n in FEWSHOT_N:
        a = f"A + {n} of our patients"
        pairs += [(a, f"{label} + {n} of our patients") for _, label in inits[1:]] + [(a, "A as it is (n = 0)")]
        pairs += [("ours (CV): all train patients, early stopping", a)]
    return found, pairs


def main():
    biopsy = set(json.loads((HERE / "cv" / "folds.json").read_text())["biopsy_patients"])
    tissue = pd.read_csv(HERE / "tiles.csv")[["image", "tissue_frac"]]
    answers = CFG / "answers.md"
    lines = ["# All public data combined, tested on our patients", ""]
    if answers.exists():
        lines += [answers.read_text(encoding="utf-8"), ""]
    lines += ["## Sources", "", source_table(), "",
              "RINGS (downloaded, 1,000 + 500 patches) left out for now on the user's instruction; AGGC22 needs a data request "
              "(links within 15 working days); PESO excluded. Radboud: 1,800 of the 3,542 Radboud train slides (random, "
              "ISUP-stratified split) because of disk space on C:; all would need ~14 GB more. "
              f"SICAPv2 'slides' are its 238 overlap regions (155 WSIs; slide ids are not in the Kaggle copy).", "",
              "Training: every epoch draws the same number of tiles from each source (20,000 each of 60,000).", ""]
    for stage, title in [("unet", "U-Net (ResNet-34, resnet34_v2 recipe)"), ("uni", "UNI frozen + the FM-grid head")]:
        s = "unet" if stage == "unet" else "uni"
        notes = [run_note(f"all_public_{s}"), run_note(f"mixed{N_MIX}_{s}"), run_note(f"karolinska_n{N_MIX}_{s}")]
        lines += [f"# {title}", "", *notes, ""]
        for mode, mtitle in [("ctx256_blend_tta", "full mode (context + blending + flips)"), ("tile", "tile mode (each tile alone)")]:
            found = {k: scored(p, tissue) for k, p in models(stage, mode).items() if all(x.exists() for x in p)}
            if not found:
                continue
            ref = ["ours (CV)", "all-Karolinska (3,814 slides)", "SICAPv2"]
            pairs = [("A: all public", r) for r in ref]
            pairs += [(f"B: mixed {N_MIX} slides", f"B: Karolinska {N_MIX} slides"), ("A: all public", f"B: mixed {N_MIX} slides"),
                      ("all-Karolinska (3,814 slides)", f"B: Karolinska {N_MIX} slides")]
            lines += [f"## Our 17 patients, {mtitle}", "", table(found, biopsy), "", "Per specimen type (mean over patients):", "",
                      specimen_table(found, biopsy), "", "Paired differences on the same patients:", "",
                      paired(found, pairs, metrics=[m for m in METRICS if all(m in f for f in found.values())]), "", "Paired Dice differences by specimen type:", "",
                      paired_by_type(found, pairs[:4], biopsy), ""]
        found, pairs = fewshot(stage, tissue, biopsy)
        if any(k.startswith("A +") for k in found):
            lines += [f"## Few-shot from A, tile mode ({title})", "",
                      "run_fewshot.py protocol: per CV fold the first n of the fold's train patients (nested, seed = fold), "
                      "100,000 tile draws, last weights; each patient scored by the fold models that never saw it.", "",
                      table(found, biopsy), "", "Paired differences on the same patients:", "",
                      paired(found, pairs, metrics=("dice", "false_tumour", "clean_auc")), ""]
    (HERE / "cv" / "all_public.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
