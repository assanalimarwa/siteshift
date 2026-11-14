"""
Reports of the 47-patient experiments (train/run_cv.py, train/run_fewshot.py, public models), in the form of the baseline report:

    python report47.py seed          # 1  cv47/seed_summary.md
    python report47.py public        # 2  cv47/public_summary.md
    python report47.py fewshot       # 3  cv47/fewshot_summary.md
    python report47.py uni_cv        # 4  cv47/uni_cv_summary.md
    python report47.py confirmation  # 5  cv47/confirmation_summary.md
    python report47.py sample        # 2+6 BiomedParse and SAM on the tile sample: cv47/sample_summary.md
    python report47.py reverse       # 9  cv47/reverse_summary.md (fold U-Nets on Karolinska test and SICAPv2 held-out)
    python report47.py all           # every report whose results exist (cv47/local_summary.md is written by hand from them)

Per patient (the unit; 47 patients, every patient scored by models that never saw it):
  dice          mean of the patient's slide Dice (evaluate.py per_slide.csv) - the headline per-patient Dice
  dice_pooled   Dice over all the patient's pixels
  sensitivity, precision (pixels, pooled per patient), false_tumour (predicted tumour on tumour-free tissue /
  that tissue), clean_auc (tiles with >= 25% tissue; tumour share 0 vs >= 50%; score = predicted tumour share)
Summaries: mean +- SD over patients with a 95% bootstrap CI of the mean (10,000 patient resamples, seed 0), median,
pooled Dice over all pixels, per specimen type and per fold; paired against the U-Net baseline (runs/cv47/imagenet,
same mode): mean difference with bootstrap CI, patients better / worse, Wilcoxon signed-rank p.
Every report also writes cv47/<experiment>_by_patient.csv (one row per patient and model).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = P.INDEX
CV = P.RESULTS / "cv47"                      # result tables (splits: P.INDEX / "cv47")
RUNS = P.RUNS
FULL = "ctx256_blend_tta"
MODES = {"tile": "tile", FULL: "full"}
INFO = pd.read_csv(CV / "dataset_by_patient.csv").set_index("p")[["type", "fold"]]
CLEAN, B, SEED = 0.5, 10_000, 0
_TISSUE = None


def tissue():
    global _TISSUE
    if _TISSUE is None:
        _TISSUE = pd.read_csv(HERE / "tiles47.csv", usecols=["image", "tissue_frac"])
    return _TISSUE


def patient(slide):
    return slide.str.split(".").str[0].astype(int)


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(np.asarray(score)).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_patient(evals):
    """Metrics per patient from evaluate.py output folders (together they cover each patient once)."""
    ps = pd.concat([pd.read_csv(Path(e) / "per_slide.csv") for e in evals])
    pt = pd.concat([pd.read_csv(Path(e) / "per_tile.csv") for e in evals])
    assert not ps.slide.duplicated().any(), "a slide is scored twice"
    ps["p"] = patient(ps.slide)
    g = ps.groupby("p")
    s = g[["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]].sum()
    out = pd.DataFrame({"slides": g.size(), "dice": g.dice.mean(),
                        "dice_pooled": 2 * s.tp / (2 * s.tp + s.fp + s.fn).clip(lower=1),
                        "sensitivity": (s.tp / s.tumour_px.clip(lower=1)).where(s.tumour_px > 0),
                        "precision": (s.tp / (s.tp + s.fp)).where(s.tp + s.fp > 0),
                        "false_tumour": s.fp_tissue / s.benign_tissue_px.clip(lower=1)})
    t = pt.merge(tissue(), on="image")
    t = t[(t.tissue_frac >= 0.25) & ((t.tumour_px == 0) | (t.tumour_px >= CLEAN * 512 ** 2))]
    t = t.assign(score=(t.tp + t.fp) / 512 ** 2, y=t.tumour_px > 0, p=patient(t.slide))
    out["clean_auc"] = t.groupby("p").apply(lambda d: auc(d.score, d.y), include_groups=False)
    tot = s.sum()
    pooled = dict(dice=2 * tot.tp / (2 * tot.tp + tot.fp + tot.fn), sensitivity=tot.tp / tot.tumour_px,
                  precision=tot.tp / (tot.tp + tot.fp), false_tumour=tot.fp_tissue / tot.benign_tissue_px)
    return out, pooled


def combine(parts):
    """Several models scoring the same patients (e.g. the 5 17-patient fold models): per patient the mean of the models."""
    tabs, pooled = zip(*parts)
    t = pd.concat(tabs).groupby(level=0).mean()
    return t, {k: float(np.mean([p[k] for p in pooled])) for k in pooled[0]}


def boot(v):
    v = np.asarray(pd.Series(v).dropna(), float)
    if len(v) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(SEED).choice(v, (B, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(v):
    v = pd.Series(v).dropna()
    lo, hi = boot(v)
    return f"{v.mean():.3f} +- {v.std():.3f} [{lo:.3f}, {hi:.3f}]"


def baseline(mode, patients=None):
    ev = [RUNS / "cv47" / "imagenet" / f"fold{k}" / f"eval_fold{k}_test_{mode}" for k in range(5)]
    t, _ = per_patient(ev)
    return t if patients is None else t.loc[t.index.intersection(patients)]


def paired(t, base):
    d = (t.dice - base.dice.reindex(t.index)).dropna()
    lo, hi = boot(d)
    try:
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    except ValueError:
        p = np.nan
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{(d > 0).sum()}/{(d < 0).sum()}", f"{p:.3g}"


def model_rows(name, mode, t, pooled, base):
    t = t.join(INFO)
    row = {"model": name, "mode": MODES.get(mode, mode), "patients": len(t),
           "Dice per patient (mean +- SD [95% CI])": fmt(t.dice), "median": f"{t.dice.median():.3f}",
           "patient-pooled Dice": fmt(t.dice_pooled), "pooled Dice (all pixels)": f"{pooled['dice']:.3f}",
           "sensitivity": fmt(t.sensitivity), "precision": fmt(t.precision), "false-tumour rate": fmt(t.false_tumour),
           "clean-tile AUC": fmt(t.clean_auc)}
    if base is not None:
        row["vs U-Net baseline (Dice diff [CI])"], row["better/worse"], row["Wilcoxon p"] = paired(t, base)
    spec = {"model": name, "mode": MODES.get(mode, mode)}
    for typ in ["biopsy", "large"]:
        x = t[t.type == typ]
        spec[f"{typ} n"] = len(x)
        spec[f"{typ} Dice mean [CI]"] = f"{x.dice.mean():.3f} [{boot(x.dice)[0]:.3f}, {boot(x.dice)[1]:.3f}]" if len(x) else ""
        spec[f"{typ} median"] = f"{x.dice.median():.3f}" if len(x) else ""
        spec[f"{typ} false-tumour"] = f"{x.false_tumour.mean():.3f}" if len(x) else ""
    fold = {"model": name, "mode": MODES.get(mode, mode),
            **{f"fold {k}": f"{t[t.fold == k].dice.mean():.3f}" if (t.fold == k).any() else "" for k in range(5)}}
    return row, spec, fold


def md_table(rows):
    if not rows:
        return "(no results yet)\n"
    cols = list(dict.fromkeys(c for r in rows for c in r))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + \
        "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows) + "\n"


def write(exp, title, intro, models, with_baseline=True, notes=""):
    """models: list of (name, mode, parts) where parts = list of eval-folder lists (one list per model scoring the patients)."""
    rows, specs, folds, csv = [], [], [], []
    for name, mode, parts in models:
        parts = [p for p in parts if all((Path(e) / "summary.json").exists() for e in p)]
        if not parts:
            continue
        t, pooled = combine([per_patient(p) for p in parts]) if len(parts) > 1 else per_patient(parts[0])
        base = baseline(mode if mode in MODES else "tile", t.index) if with_baseline else None
        r, s, f = model_rows(name, mode, t, pooled, base)
        rows.append(r), specs.append(s), folds.append(f)
        csv.append(t.join(INFO).reset_index().rename(columns={"index": "p"}).assign(model=name, mode=MODES.get(mode, mode), models=len(parts)))
    text = (f"# {title}\n\n_{pd.Timestamp.now():%Y-%m-%d %H:%M}, machine: 4090 (report47.py {exp})_\n\n{intro}\n\n"
            "Per-patient Dice = mean of the patient's slide Dice; CI = 95% bootstrap of the mean over patients.\n\n"
            f"## Results\n\n{md_table(rows)}\n## By specimen type (per-patient Dice)\n\n{md_table(specs)}\n"
            f"## By fold (mean per-patient Dice of each fold's test patients)\n\n{md_table(folds)}\n{notes}")
    (CV / f"{exp}_summary.md").write_text(text, encoding="utf-8")
    if csv:
        pd.concat(csv).to_csv(CV / f"{exp}_by_patient.csv", index=False)
    print(f"-> {CV / f'{exp}_summary.md'} ({len(rows)} rows)")
    return rows


def ev(run, tag, mode):
    return RUNS / run / f"eval_{tag}_test_{mode}"


def report_seed():
    models = []
    for mode in MODES:
        models.append((f"U-Net seed 0 (baseline) fold 0", mode, [[ev("cv47/imagenet/fold0", "fold0", mode)]]))
        models.append((f"U-Net seed 1 fold 0", mode, [[ev("cv47_s1/imagenet/fold0", "fold0", mode)]]))
    notes = ("\nReference given by the user: seed 0 0.7616 (tile) / 0.7800 (full); H100 run 0.7774 / 0.8065 "
             "(per-patient mean Dice, fold 0, 9 patients). The H100 per-patient table is not on this PC.\n\n"
             "The seed-1 run crashed at 13:04 on 2026-11-04 (C: full) and resumed from its epoch-12 checkpoint; its "
             "log.csv lists epoch 13 twice, the second row is the real one. Model and scores are unaffected.\n")
    return write("seed", "Seed check: baseline U-Net fold 0, seed 1 vs seed 0", "Fold 0 of cv47 (9 test patients); "
                 "same data, folds and settings, only --seed differs (it also changes the 20,000 val tiles drawn).",
                 models, notes=notes)


def report_baseline():
    models = [("U-Net site CV (baseline)", m, [[ev(f"cv47/imagenet/fold{k}", f"fold{k}", m) for k in range(5)]]) for m in MODES]
    return write("baseline47", "Baseline U-Net (ImageNet ResNet-34), 47-patient 5-fold patient CV",
                 "runs/cv47/imagenet, finished 2026-11-02 (training details: baseline_summary.md). Same metrics as every "
                 "other report47 report; baseline_summary.md headlines patient-pooled Dice instead.", models, with_baseline=False)


PUBLIC_LABELS = {"sicap_pretrain": "SICAPv2 U-Net", "sicap_pretrain_seed1": "SICAPv2 U-Net seed 1",
                 "sicap_r40": "SICAPv2 relabelled r40 U-Net", "sicap_r80": "SICAPv2 relabelled r80 U-Net",
                 "karolinska_n20_unet": "Karolinska 20 slides U-Net", "karolinska_n155_unet": "Karolinska 155 U-Net",
                 "karolinska_n419_unet": "Karolinska 419 U-Net", "karolinska_n600_unet": "Karolinska 600 U-Net",
                 "karolinska_n1000_unet": "Karolinska 1,000 U-Net", "karolinska_all_unet": "Karolinska 3,814 U-Net",
                 "mixed600_unet": "600 mixed (200 SICAPv2 + 200 Kar + 200 Radboud) U-Net",
                 "all_public_unet": "all public (A) U-Net",
                 "sicap_fm_uni": "SICAPv2 UNI", "sicap_fm_virchow2": "SICAPv2 Virchow2", "sicap_fm_conch": "SICAPv2 CONCH",
                 "sicap_fm_dinov2": "SICAPv2 DINOv2-L", "sicap_fm_unimedclip": "SICAPv2 UniMed-CLIP",
                 "all_public_uni": "all public (A) UNI", "mixed600_uni": "600 mixed UNI", "karolinska_n600_uni": "Karolinska 600 UNI",
                 "radboud_unet": "Radboud U-Net"}
PUBLIC_UNETS = ["sicap_pretrain", "sicap_pretrain_seed1", "sicap_r40", "sicap_r80", "karolinska_n20_unet", "karolinska_n155_unet",
                "karolinska_n419_unet", "karolinska_n600_unet", "karolinska_n1000_unet", "karolinska_all_unet", "mixed600_unet",
                "all_public_unet"]
PUBLIC_FMS = ["sicap_fm_uni", "sicap_fm_virchow2", "sicap_fm_conch", "sicap_fm_dinov2", "sicap_fm_unimedclip",
              "all_public_uni", "mixed600_uni", "karolinska_n600_uni"]
FEWSHOT_N = [2, 4, 8]


def report_public():
    models = [(PUBLIC_LABELS[r], mode, [[ev(r, "all47", mode)]])
              for mode in MODES for r in PUBLIC_UNETS + (PUBLIC_FMS if mode == "tile" else [])]
    models = [("U-Net site CV (baseline)", m, [[ev(f"cv47/imagenet/fold{k}", f"fold{k}", m) for k in range(5)]]) for m in MODES] + models
    models += [(PUBLIC_LABELS["radboud_unet"], m, [[ev("radboud_unet", "all47", m)]]) for m in MODES]
    return write("public", "Public-trained models on all 47 patients", "Models trained only on public data (no tile of "
                 "ours), scored on all 47 patients (cv47/all47.json). U-Nets in tile and full mode, frozen encoders tile mode. "
                 "BiomedParse (tile sample only): sample_summary.md.", models,
                 notes="Scope (user's choice, 2026-11-04, to finish about a day earlier): full mode only for the SICAPv2, "
                       "Karolinska 3,814, 600 mixed and all-public (A) U-Nets; the other 8 public U-Nets in tile mode only. "
                       "Their full-mode scores may be added later from the H100.\n\n"
                       "\"All public combined\" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + "
                       "1,800 Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source.\n\n"
                       "Radboud U-Net (added 2026-11-04): same recipe as the Karolinska 3,814 U-Net, trained on "
                       "all 3,542 Radboud training slides (gland-level Gleason 3-5 = tumour), tile and full mode.")


def report_fewshot():
    names = {"kar": "U-Net from Karolinska 3,814", "imagenet": "U-Net from ImageNet", "uni_scratch": "UNI fresh head",
             "uni_public": "UNI from all-public UNI"}
    models = []
    for n in FEWSHOT_N:
        for init, label in names.items():
            evs = [ev(f"fewshot47/{init}/n{n}/fold{k}", f"fold{k}_n{n}", "tile") for k in range(5)]
            if all((e / "summary.json").exists() for e in evs):
                models.append((f"{label} + {n} patients", "tile", [evs]))
    for init, run in [("U-Net Karolinska 3,814 (n = 0)", "karolinska_all_unet"), ("UNI all public (n = 0)", "all_public_uni")]:
        models.append((init, "tile", [[ev(run, "all47", "tile")]]))
    return write("fewshot", "Few-shot adaptation on 47 patients", "run_fewshot.py protocol on the cv47 folds: per fold the "
                 "train patients in a random order (seed = fold), the first n = 2 / 4 / 8 are the only labelled patients; "
                 "fixed budget 10 epochs x 10,000 draws, last weights; tile mode on the fold's test patients.", models)


def report_uni_cv():
    models = [("UNI site CV (4090)", "tile", [[ev(f"cv47/fm_uni/fold{k}", f"fold{k}", "tile") for k in range(5)]])]
    return write("uni_cv", "UNI site CV (frozen UNI + shared head), 47 patients", "run_cv.py fm_uni on cv47 (15 epochs max, "
                 "patience 4, epochs of 60,000 tiles, 20,000 val tiles), tile mode as for every frozen encoder. Anchor for the "
                 "4090 vs H100 comparison.", models)


def report_confirmation():
    models = [(f"17-patient U-Nets (mean of 5 fold models)", m,
               [[ev(f"cv/imagenet/fold{k}", "conf30", m)] for k in range(5)]) for m in MODES]
    for k in range(5):
        models += [(f"17-patient U-Net fold {k}", m, [[ev(f"cv/imagenet/fold{k}", "conf30", m)]]) for m in MODES]
    return write("confirmation", "Confirmation: the original 17-patient models on the 30 new patients",
                 "The 5 fold U-Nets of the 17-patient CV (runs/cv/imagenet) never saw any of the 30 new patients "
                 "(cv47/conf30.json); each scores all 30. Baseline = the 47-patient CV U-Net on the same 30 patients.", models,
                 notes="Scope (user's choice, 2026-11-04): tile mode only for the confirmation set; no full-mode scores.")


def report_sample():
    """BiomedParse and SAM on cv47/zs_sample.csv; the baseline U-Net's counts on the same tiles from its per_tile.csv."""
    from PIL import Image
    from data import mask_path
    from scipy import ndimage
    s = pd.read_csv(HERE / "cv47" / "zs_sample.csv")          # not released (tile-level tumour fractions)
    data = P.DATASET47
    truth = {r: np.array(Image.open(data / mask_path(r))) > 0 for r in s.image}
    sets = {}
    bp = P.PRED / "biomedparse_v1_zs_sample"
    if len(list(bp.glob("*.npz"))) == 47:
        cells, region = {}, {}
        for f in bp.glob("*.npz"):
            z = np.load(f, allow_pickle=True)
            for img, prob in zip(z["image"], z["prob"]):
                cells[img] = prob >= 128
                region[img] = ndimage.binary_dilation(cells[img], iterations=24)
        sets["BiomedParse v1 zero-shot, cells"], sets["BiomedParse v1 zero-shot, region (24 px dilation)"] = cells, region
    for mode, label in [("box", "SAM ViT-H, box prompts from the annotation"), ("auto", "SAM ViT-H automatic, oracle mask choice")]:
        d = P.PRED / "sam_vit_h" / f"zs_sample_{mode}"
        if (d / "done.json").exists():
            pred = {}
            for f in d.glob("*.npz"):
                z = np.load(f, allow_pickle=True)
                for img, pk in zip(z["image"], z["pred"]):
                    pred[img] = np.unpackbits(pk)[:512 * 512].reshape(512, 512).astype(bool)
            sets[label] = pred
    from metrics import tissue_mask  # noqa: F401  (tissue below via the stored tissue_frac / benign pixels)
    rows = []
    for label, pred in sets.items():
        recs = []
        for _, r in s.iterrows():
            p, t = pred[r.image], truth[r.image]
            recs.append(dict(p=r.patient, group=r.group, image=r.image, tp=int((p & t).sum()), fp=int((p & ~t).sum()),
                             fn=int((~p & t).sum()), tumour_px=int(t.sum()), score=p.mean(), tissue_frac=r.tissue_frac))
        rows.append((label, pd.DataFrame(recs)))
    base = pd.concat([pd.read_csv(ev(f"cv47/imagenet/fold{k}", f"fold{k}", m) / "per_tile.csv").assign(mode=m)
                      for k in range(5) for m in MODES])
    for m in MODES:
        b = s[["image", "patient", "group", "tissue_frac"]].merge(base[base["mode"] == m], on="image").rename(columns={"patient": "p"})
        rows.append((f"U-Net site CV (baseline), {MODES[m]} mode, same tiles", b.assign(score=(b.tp + b.fp) / 512 ** 2)))
    out, csv = [], []
    for label, d in rows:
        g = d.groupby("p")
        tum = d[d.group == "tumour"].groupby("p")[["tp", "fp", "fn", "tumour_px"]].sum()
        free = d[d.group == "tumour-free"].groupby("p")
        pp = pd.DataFrame({"dice": 2 * tum.tp / (2 * tum.tp + tum.fp + tum.fn).clip(lower=1),
                           "sensitivity": tum.tp / tum.tumour_px.clip(lower=1),
                           "precision": (tum.tp / (tum.tp + tum.fp)).where(tum.tp + tum.fp > 0),
                           "false_tumour_free_tiles": free.score.mean()})
        cl = d[(d.tissue_frac >= 0.25) & ((d.tumour_px == 0) | (d.tumour_px >= CLEAN * 512 ** 2))]
        pp["clean_auc"] = cl.groupby("p").apply(lambda x: auc(x.score, x.tumour_px > 0), include_groups=False)
        out.append({"model": label, "patients": len(pp), "Dice on tumour tiles (mean +- SD [CI])": fmt(pp.dice),
                    "median": f"{pp.dice.median():.3f}", "sensitivity": fmt(pp.sensitivity), "precision": fmt(pp.precision),
                    "tumour predicted on tumour-free tiles": fmt(pp.false_tumour_free_tiles), "clean-tile AUC": fmt(pp.clean_auc)})
        csv.append(pp.join(INFO).reset_index().assign(model=label))
    text = (f"# BiomedParse and SAM on the tile sample (47 patients)\n\n_{pd.Timestamp.now():%Y-%m-%d %H:%M}, machine: 4090_\n\n"
            f"Tiles: cv47/zs_sample.csv ({len(s)} tiles: per patient up to 60 with annotated tumour and 60 tumour-free "
            "tissue tiles, seed 0; zs_sample.py). Neither model can score all 449,623 tiles in reasonable time. "
            "Dice / sensitivity / precision over each patient's sampled tumour tiles; false tumour = mean predicted share "
            "of the tumour-free sampled tiles. SAM box mode is given the tumour location (no prediction without a box), "
            "and SAM automatic mode chooses masks with the labels (oracle): both are upper bounds, not comparable to the "
            "U-Net as detectors. The U-Net rows are the baseline's counts on the same tiles.\n\n" + md_table(out))
    (CV / "sample_summary.md").write_text(text, encoding="utf-8")
    if csv:
        pd.concat(csv).to_csv(CV / "sample_by_patient.csv", index=False)
    print(f"-> {CV / 'sample_summary.md'} ({len(out)} rows)")
    return out


def report_reverse():
    """9: our 47-patient fold U-Nets away from home (Karolinska test, SICAPv2 held-out), next to the 17-patient folds."""
    from karolinska_all_report import bootstrap_ci, mean_of_models, paired, per_unit, table
    k_tissue = pd.read_csv(HERE / "karolinska_all_tiles.csv", usecols=["image", "tissue_frac"])
    folds = {"ours, 47-patient CV": "cv47/imagenet", "ours, 17-patient CV": "cv/imagenet"}
    lines = ["# Reverse transfer: our fold U-Nets on public test data", "",
             f"_{pd.Timestamp.now():%Y-%m-%d %H:%M}, machine: 4090 (report47.py reverse)_", "",
             "The five fold U-Nets of the 47-patient CV (runs/cv47/imagenet) and, for comparison, of the 17-patient CV "
             "(runs/cv/imagenet, scored 10-24/25); every unit is scored by all five models, per unit the mean of the five. "
             "Reference: the model trained on that dataset (at home). Mean over units [95% bootstrap CI]; paired "
             "differences on the same units.", ""]
    for mode, title in [("tile", "tile mode"), (FULL, "full mode (context + blending + flips)")]:
        k = f"eval_karolinska_all_split_test_{mode}"
        found, spread = {}, []
        for name, run in folds.items():
            p = [RUNS / run / f"fold{j}" / k / "per_tile.csv" for j in range(5)]
            if all(x.exists() for x in p):
                found[name], d = mean_of_models(p, "slide", k_tissue)
                spread.append(f"- {name}: mean Dice per fold model {min(d):.3f} to {max(d):.3f}")
        home = RUNS / "karolinska_all_unet" / k / "per_tile.csv"
        if home.exists():
            found["Karolinska, all 3,814 slides (home)"] = per_unit([home], "slide", k_tissue)
        lines += [f"## Karolinska test, 1,092 slides, {title} (unit = slide)", "", table(found), "", *spread, "",
                  paired(found, [("ours, 47-patient CV", "ours, 17-patient CV"),
                                 ("Karolinska, all 3,814 slides (home)", "ours, 47-patient CV")]), ""]

    def regions(runs, tag=""):
        """SICAPv2 val + guard (evaluate_sicap.py, tile mode): per overlap region the mean Dice of the models, and the
        Dice over the models' summed pixel counts."""
        dice, pooled = [], np.zeros(3)
        for run in runs:
            p = [RUNS / run / f"eval_sicap_{s}_tile{tag}" / "per_region.csv" for s in ["val", "guard"]]
            if not all(x.exists() for x in p):
                return None
            r = pd.concat([pd.read_csv(x).assign(slide=lambda d, s=s: s + "/" + d.slide) for x, s in zip(p, ["val", "guard"])])
            g = r.groupby("slide")[["tp", "fp", "fn", "tumour_px"]].sum()
            dice.append((2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)).where(g.tumour_px > 0))
            pooled += g[["tp", "fp", "fn"]].sum().values
        return pd.concat(dice, axis=1).mean(axis=1), 2 * pooled[0] / (2 * pooled[0] + pooled[1] + pooled[2])

    sic = {name: regions([f"{run}/fold{j}" for j in range(5)], "_breakdown" if run == "cv/imagenet" else "")
           for name, run in folds.items()}                     # the 17-patient folds' SICAPv2 folders carry "_breakdown"
    sic["SICAPv2 U-Net (home)"] = regions(["sicap_pretrain"])
    sic = {k: v for k, v in sic.items() if v is not None}
    rows = []
    for name, (d, pooled) in sic.items():
        v = d.dropna()
        lo, hi = bootstrap_ci(v)
        rows.append({"model": name, "regions with tumour": len(v), "Dice per region": f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]",
                     "pooled Dice (all pixels)": f"{pooled:.3f}"})
    lines += ["## SICAPv2 held-out (val 1,883 + guard 1,077 tiles), tile mode (unit = overlap region with tumour)", "",
              "Tiles enlarged x1.75 to our scale (evaluate_sicap.py).", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    for a, b in [("ours, 47-patient CV", "ours, 17-patient CV"), ("SICAPv2 U-Net (home)", "ours, 47-patient CV")]:
        if a in sic and b in sic:
            d = (sic[a][0] - sic[b][0]).dropna()
            lo, hi = bootstrap_ci(d)
            lines.append(f"- {a} minus {b}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], higher / lower {(d > 0).sum()} / {(d < 0).sum()} regions")
    (CV / "reverse_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {CV / 'reverse_summary.md'}")


REPORTS = {"baseline": report_baseline, "seed": report_seed, "public": report_public, "fewshot": report_fewshot, "uni_cv": report_uni_cv,
           "confirmation": report_confirmation, "sample": report_sample, "reverse": report_reverse}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for name in (list(REPORTS) if which == ["all"] else which):
        REPORTS[name]()
