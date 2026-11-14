"""
Per-model results for the 47-patient cross-validation, tile mode, written for download:

    python export_model.py uni conch dinov2          # models finished: 5 folds (site CV) or one all-public run
    python export_model.py --pairwise-only

Writes cv47/<key>_summary.md and cv47/<key>_by_patient.csv, copies both into export/<key>/ (next to
export/<key>/predictions/, written by protocol_predict.py through the queue) and refreshes
cv47/pairwise_models.md + export/pairwise_models.md.

Definitions (as in the local reports; the baseline's own per_slide.csv / per_tile.csv are scored the same way):
  per-patient Dice   mean over the patient's slides of the slide Dice (2 tp / (2 tp + fp + fn) of the slide). HEADLINE.
                     Baseline tile mode: 0.720 [0.668, 0.768].
  patient-pooled     Dice of all the patient's tiles together (secondary; baseline 0.755).
  sensitivity, precision, false-tumour rate (tissue-free-of-tumour pixels predicted tumour, glass excluded), clean-tile AUC
                     of the patient's pooled counts, as cv_summary.py / karolinska_all_report.per_unit.
  mean +- SD, median over the 47 patients; CI = 2.5 / 97.5 percentiles of 10,000 bootstrap resamples of the patients (seed 0).
  paired difference  model minus baseline per patient: mean [CI], patients higher / lower, two-sided Wilcoxon signed-rank p.
  Differences below 0.03 between single runs are called ties in the wording (user, 2026-11-05).
Every patient is scored once, by the fold model that did not train on them.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cv_summary import bootstrap_ci
from karolinska_all_report import CLEAN, auc

HERE = P.INDEX
T = P.INDEX
RUNS, CV, EXPORT = P.RUNS, P.RESULTS / "cv47", T / "export"
TIE = 0.03
COUNTS = ["tp", "fp", "fn", "tumour_px", "fp_tissue", "benign_tissue_px"]
BASE = ("baseline U-Net", "imagenet")

# key -> (label, run folders, evaluation folder pattern)
MODELS = {
    "uni": ("UNI + head", "fm_uni"), "dinov2": ("DINOv2-L + head", "fm_dinov2"), "conch": ("CONCH + head", "fm_conch"),
    "virchow2": ("Virchow2 + head", "fm_virchow2"), "unimedclip": ("UniMed-CLIP + head", "fm_unimedclip"), "nnunet": ("nnU-Net-style 2D", "nnunet"),
    "segformer": ("SegFormer-B2", "segformer"), "transunet": ("TransUNet (R50-ViT-B)", "transunet"),
}
ALLPUBLIC = {f"{k}_allpublic": (f"{MODELS[k][0]}, all-public training", f"all_public_{MODELS[k][1]}")
             for k in ("nnunet", "segformer", "transunet")}


def eval_dirs(key):
    if key == "unet_allpublic":                       # trained on the PC; its per_slide.csv / per_tile.csv are copied here
        return [RUNS / "all_public_unet" / "eval_all_test_tile"]
    if key in ALLPUBLIC:
        return [RUNS / ALLPUBLIC[key][1] / "eval_all_test_tile"]
    if key == "baseline":
        return [RUNS / CV.name / BASE[1] / f"fold{k}" / f"eval_fold{k}_test_tile" for k in range(5)]
    return [RUNS / CV.name / MODELS[key][1] / f"fold{k}" / f"eval_fold{k}_test_tile" for k in range(5)]


def finished(key):
    return all((d / "per_slide.csv").exists() and (d / "per_tile.csv").exists() for d in eval_dirs(key))


def patients(frame):
    return frame.slide.str.split(".").str[0].astype(int)


def tissue():
    return pd.read_csv(HERE / "tiles47.csv", usecols=["image", "tissue_frac"])


def per_patient(key, tis):
    """One row per patient: headline Dice (mean of slide Dice), pooled Dice and the pooled-count metrics."""
    dirs = eval_dirs(key)
    sl = pd.concat([pd.read_csv(d / "per_slide.csv") for d in dirs])
    sl["pt"] = patients(sl)
    out = pd.DataFrame({"dice": sl.groupby("pt").dice.mean()})
    g = sl.groupby("pt")[COUNTS].sum()
    out["slides"] = sl.groupby("pt").size()
    out["dice_pooled"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    out["sensitivity"] = g.tp / g.tumour_px
    out["precision"] = g.tp / (g.tp + g.fp)
    out["false_tumour_rate"] = g.fp_tissue / g.benign_tissue_px
    t = pd.concat([pd.read_csv(d / "per_tile.csv", usecols=["slide", "image", "tp", "fp", "tumour_px"]) for d in dirs])
    t = t.merge(tis, on="image")
    t = t[t.tissue_frac >= 0.25].assign(score=lambda d: (d.tp + d.fp) / 512 ** 2, y=lambda d: d.tumour_px > 0)
    t = t[(t.tumour_px == 0) | (t.tumour_px >= CLEAN * 512 ** 2)]              # clean tiles, as protocol.clean_tiles
    out["clean_auc"] = t.groupby(patients(t)).apply(lambda d: auc(d.score, d.y), include_groups=False)
    return out, sl, g


def cell(v, f="{:.3f}"):
    v = v.dropna()
    lo, hi = bootstrap_ci(v)
    return f"{f.format(v.mean())} [{f.format(lo)}, {f.format(hi)}]"


def diff_row(name, d):
    d = d.dropna()
    lo, hi = bootstrap_ci(d)
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = float("nan")
    return {"measure": name, "patients": len(d), "difference [95% CI]": f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
            "higher / lower": f"{(d > 0).sum()} / {(d < 0).sum()}", "Wilcoxon p": f"{p:.3g}"}, d.mean(), lo, hi


def wording(a, b, mean, lo, hi):
    if abs(mean) < TIE:
        return f"tie (difference below {TIE})"
    lead = a if mean > 0 else b
    return f"{lead} ahead" + ("" if (lo > 0 or hi < 0) else ", but the CI includes 0")


def md(df):
    """GitHub markdown table (no tabulate in the queue's environment, which is left untouched)."""
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def training_table(key):
    rows = []
    if key in ALLPUBLIC:
        runs = [("all public", RUNS / ALLPUBLIC[key][1])]
    else:
        runs = [(f"fold {k}", RUNS / CV.name / MODELS[key][1] / f"fold{k}") for k in range(5)]
    for name, r in runs:
        if not (r / "log.csv").exists():
            continue
        log = pd.read_csv(r / "log.csv")
        cfg = json.loads((r / "config.json").read_text()) if (r / "config.json").exists() else {}
        v = cfg.get("versions", {})
        best = log.loc[log.val_dice.idxmax()]
        epoch_tiles = cfg.get("epoch_tiles", 60000)
        rows.append({"run": name, "epochs run": len(log), "best epoch": int(best.epoch), "best val Dice": f"{best.val_dice:.4f}",
                     "train img/s (mean)": f"{epoch_tiles / (log.minutes.mean() * 60):.0f}", "machine": v.get("machine", "?"),
                     "weights evaluated": "last epoch (no early stopping)" if cfg.get("keep") == "last" else "best epoch"})
    return rows, (json.loads((runs[0][1] / "config.json").read_text()) if (runs[0][1] / "config.json").exists() else {})


def status_lines():
    """Virchow2 and UniMed-CLIP: where they stand now."""
    skipped = json.loads((RUNS / "queue_skipped.json").read_text()) if (RUNS / "queue_skipped.json").exists() else {}
    out = []
    v = RUNS / CV.name / "fm_virchow2"
    skip_v = [m for k, m in skipped.items() if "virchow2" in k]
    if finished("virchow2"):
        out.append("- **Virchow2**: finished, see its own summary.")
    elif skip_v:
        out.append(f"- **Virchow2**: not run. {skip_v[0]}")
    elif v.exists():
        done = sum((v / f"fold{k}" / "TRAINED").exists() for k in range(5))
        out.append(f"- **Virchow2**: running, {done} of 5 folds trained.")
    else:
        out.append("- **Virchow2**: not started yet. It runs after SegFormer and UniMed-CLIP in the queue order and has no cutoff gate any more "
                   "(user, 2026-11-08: finish all the training).")
    weights = list((T / "weights").rglob("*unimed*")) + list((T / "weights").rglob("*UniMed*"))
    if finished("unimedclip") if "unimedclip" in MODELS else False:
        out.append("- **UniMed-CLIP**: finished.")
    elif weights:
        out.append(f"- **UniMed-CLIP**: weights found ({weights[0]}); the job can be enabled.")
    else:
        out.append("- **UniMed-CLIP**: not run. There is no UniMed-CLIP repository and no weights in weights/ on this server (the "
                   "server cannot download them); the job is skipped and logged until the weights are placed in weights/.")
    return out


def prediction_status(key):
    pdir = EXPORT / key / "predictions"
    done = sorted(pdir.glob("*/done.json")) if pdir.exists() else []
    n = 1 if key in ALLPUBLIC else 5
    size = sum(f.stat().st_size for f in pdir.rglob("*") if f.is_file()) / 1e9 if pdir.exists() else 0
    return len(done), n, size


def machine_notes(key, site_key):
    """Where the runs of a model trained (config.json machine labels), for the comparability section."""
    def labels(k):
        rows, _ = training_table(k)
        return "; ".join(f"{r['run']}: {r['machine']}" for r in rows)
    out = [f"- Training machines (config.json): {labels(key)}."]
    if site_key:
        out.append(f"- Its site CV ({MODELS[site_key][0]}) trained on: {labels(site_key)}. Runs on the shared GPU 0 ran at about a third of the speed of GPU 1; "
                   "a paused and resumed run continues from its last checkpoint (same settings, same seed).")
    return out


def summary(key, tis, base, per_cache):
    label = (MODELS.get(key) or ALLPUBLIC[key])[0]
    allpub = key in ALLPUBLIC
    site_key = key[:-len("_allpublic")] if allpub else None
    if allpub and not finished(site_key):
        raise SystemExit(f"{key}: its site CV ({site_key}) is not finished, so the architecture-matched comparison cannot be written yet")
    p, sl, g = per_cache(key)
    b, bsl, bg = per_cache("baseline")
    # reference of the wording: the architecture's own site CV for an all-public run, the baseline U-Net otherwise
    r, rsl, rg = per_cache(site_key) if allpub else (b, bsl, bg)
    rname = f"{MODELS[site_key][0]}, site CV (same architecture)" if allpub else BASE[0] + " (4090)"
    rshort = "its own site CV" if allpub else "baseline"
    meta = pd.read_csv(CV / "baseline_imagenet_by_patient.csv").set_index("pt")
    fold = meta.fold.reindex(p.index)
    both = p.index.intersection(r.index)
    tot = lambda s: s[COUNTS].sum()

    def pooled(sums):
        return dict(dice=2 * sums.tp / (2 * sums.tp + sums.fp + sums.fn), sens=sums.tp / sums.tumour_px,
                    prec=sums.tp / (sums.tp + sums.fp), ftr=sums.fp_tissue / sums.benign_tissue_px)

    def main_row(name, x, s):
        d = x.dice
        return {"model": name, "patients": len(x), "**per-patient Dice** (mean of slide Dice) [95% CI]": cell(d),
                "SD": f"{d.std():.3f}", "median": f"{d.median():.3f}",
                "patient-pooled Dice [95% CI] (secondary)": cell(x.dice_pooled),
                "sensitivity": cell(x.sensitivity), "precision": cell(x.precision),
                "false-tumour rate": cell(x.false_tumour_rate), "clean-tile AUC": cell(x.clean_auc)}

    def pix_row(name, s):
        q = pooled(tot(s))
        return {"model": name, "pooled Dice (all pixels)": f"{q['dice']:.3f}", "median slide Dice": f"{s.dice.median():.3f}",
                "sensitivity": f"{q['sens']:.3f}", "precision": f"{q['prec']:.3f}", "false-tumour rate": f"{q['ftr']:.1%}"}

    lines = [f"# {label}: 47-patient 5-fold patient CV, tile mode" if not allpub else
             f"# {label}: one model trained on public data only, tested on all 47 patients, tile mode",
             "", f"Written {time.strftime('%Y-%m-%d %H:%M')} by export_model.py. Metric definitions: see the top of export_model.py "
             "(the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).", ""]
    lines += ["## Machine and comparability",
              "- The encoders and the other models here were trained and evaluated on the **H100 server**; the baseline U-Net "
              "was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings "
              "differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.",
              "- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise."]
    if allpub:
        lines += [f"- **Wording reference**: this run is compared with its own site CV ({MODELS[site_key][0]}, same architecture, recipe, epochs and seed), "
                  "not with the baseline U-Net. The baseline row below is context only.",
                  "- The two differ by design in the training data: the site-CV fold models train on 35-36 of our patients and are tested on the "
                  "held-out 9-10 (five models, every patient once); the all-public model trains on SICAPv2 + Karolinska + Radboud (equal tile draws "
                  "per source, no tile of ours) and is one model tested on all 47 patients."]
    lines += machine_notes(key, site_key)
    if key == "uni":
        lines += ["- UNI fold 0: its first 4 epochs ran on the shared GPU 0 before the GPU 1 reset, then resumed from that checkpoint on GPU 1; "
                  "the config.json machine label of that run is the resumed one."]
    lines += [""]
    main_rows = [main_row(label, p, sl), main_row(rname, r, rsl)] + ([main_row(BASE[0] + " (4090, context only)", b, bsl)] if allpub else [])
    pix_rows = [pix_row(label, sl), pix_row(rname, rsl)] + ([pix_row(BASE[0] + " (4090, context only)", bsl)] if allpub else [])
    lines += ["## Main result (headline: per-patient Dice = mean of the patient's slide Dice)",
              md(pd.DataFrame(main_rows)), "",
              "Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts "
              "(clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).", "",
              "### All pixels pooled (as in cv47/baseline_summary.md)",
              md(pd.DataFrame(pix_rows)), ""]
    # per specimen type and per fold
    rows = []
    typ = meta.type.reindex(p.index)
    for name, mask in [("biopsy", typ == "biopsy"), ("large section", typ == "large")] + [(f"fold {k}", fold == k) for k in range(5)]:
        idx = p.index[mask.values]
        if not len(idx):
            continue
        d = p.dice[idx] - r.dice[idx]
        rows.append({"group": name, "patients": len(idx), "model per-patient Dice": f"{p.dice[idx].mean():.3f}",
                     f"{rshort} per-patient Dice": f"{r.dice[idx].mean():.3f}", "difference": f"{d.mean():+.3f}",
                     "model pooled Dice": f"{p.dice_pooled[idx].mean():.3f}", f"{rshort} pooled Dice": f"{r.dice_pooled[idx].mean():.3f}",
                     "model clean AUC": f"{p.clean_auc[idx].mean():.3f}", f"{rshort} clean AUC": f"{r.clean_auc[idx].mean():.3f}"})
    lines += ["## Per specimen type and per fold (means over the patients of the group)", md(pd.DataFrame(rows)), ""]
    # paired vs the reference
    rows, tie_text = [], ""
    for name, col in [("per-patient Dice (headline)", "dice"), ("patient-pooled Dice", "dice_pooled"), ("sensitivity", "sensitivity"),
                      ("precision", "precision"), ("false-tumour rate (lower is better)", "false_tumour_rate"), ("clean-tile AUC", "clean_auc")]:
        rr, m, lo, hi = diff_row(name, p[col][both] - r[col][both])
        rows.append(rr)
        if col == "dice":
            tie_text = wording(label, rname if allpub else "baseline", m, lo, hi)
    lines += [f"## Paired differences: model minus {rname} (same patients)", md(pd.DataFrame(rows)), "",
              f"Headline wording: **{tie_text}**. higher / lower = number of patients on which the model scored above / below {rshort}.", ""]
    if not allpub:
        lines += pairwise_section(per_cache)
    # training
    tr, cfg = training_table(key)
    lines += ["## Training", md(pd.DataFrame(tr)), "",
              f"Settings (config.json, identical to the baseline except the architecture): arch {cfg.get('arch')}, epochs max {cfg.get('epochs')}, "
              f"patience {cfg.get('patience')}, batch {cfg.get('batch')}, lr {cfg.get('lr')}, weight decay {cfg.get('weight_decay')}, "
              f"EMA {cfg.get('ema')}, epoch = {cfg.get('epoch_tiles')} tiles, seed {cfg.get('seed')}, recipe {cfg.get('recipe')}.",
              "Versions: " + ", ".join(f"{k} {v}" for k, v in (cfg.get("versions") or {}).items() if k not in ("machine", "cuda_visible_devices", "gpu_mem_cap_gb")), ""]
    n, tot_n, size = prediction_status(key)
    lines += ["## Files", f"- predictions (protocol format, tile mode, threshold 0.5): export/{key}/predictions/ - {n} of {tot_n} prediction folders "
              f"finished, {size:.1f} GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)",
              f"- cv47/{key}_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, "
              "clean-tile AUC, baseline values and differences" + (", site-CV values and differences" if allpub else "") + ")", ""]
    lines += ["## Status of the other planned models"] + status_lines() + [""]
    text = "\n".join(lines)
    # by-patient table
    bp = pd.DataFrame({"fold": fold, "type": typ, "slides": p.slides, "tiles": meta.tiles.reindex(p.index),
                       "tumour_pct": meta["tum%"].reindex(p.index),
                       "dice": p.dice, "dice_pooled": p.dice_pooled, "sensitivity": p.sensitivity, "precision": p.precision,
                       "false_tumour_rate": p.false_tumour_rate, "clean_auc": p.clean_auc,
                       "baseline_dice": b.dice.reindex(p.index), "baseline_dice_pooled": b.dice_pooled.reindex(p.index),
                       "baseline_clean_auc": b.clean_auc.reindex(p.index)})
    bp["diff_dice"] = bp.dice - bp.baseline_dice
    bp["diff_dice_pooled"] = bp.dice_pooled - bp.baseline_dice_pooled
    if allpub:
        bp["site_cv_dice"] = r.dice.reindex(p.index)
        bp["site_cv_dice_pooled"] = r.dice_pooled.reindex(p.index)
        bp["site_cv_clean_auc"] = r.clean_auc.reindex(p.index)
        bp["diff_vs_site_cv_dice"] = bp.dice - bp.site_cv_dice
    bp.index.name = "patient"
    return text, bp.round(4)


def pairwise_section(per_cache):
    keys = [k for k in MODELS if finished(k)]
    if len(keys) < 2:
        return []
    rows = []
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            pa, pc = per_cache(a)[0], per_cache(c)[0]
            for name, col in [("per-patient Dice", "dice"), ("patient-pooled Dice", "dice_pooled"), ("clean-tile AUC", "clean_auc")]:
                r, m, lo, hi = diff_row(name, pa[col] - pc[col])
                rows.append({"difference": f"{MODELS[a][0]} minus {MODELS[c][0]}", **r, "wording": wording(MODELS[a][0], MODELS[c][0], m, lo, hi)
                             if col == "dice" else ""})
    return ["## Pairwise differences between the finished models (same 47 patients, tile mode)", md(pd.DataFrame(rows)), "",
            f"Wording applies to the per-patient Dice row: a difference below {TIE} is a tie between single runs.", ""]


def main():
    args = sys.argv[1:]
    tis = tissue()
    cache = {}

    def per_cache(key):
        if key not in cache:
            cache[key] = per_patient(key, tis)
        return cache[key]

    keys = [k for k in args if not k.startswith("--")]
    for key in keys:
        if key not in MODELS and key not in ALLPUBLIC:
            raise SystemExit(f"unknown model {key}; known: {', '.join(list(MODELS) + list(ALLPUBLIC))}")
        if not finished(key):
            raise SystemExit(f"{key}: evaluation results are not complete yet")
        text, bp = summary(key, tis, None, per_cache)
        (CV / f"{key}_summary.md").write_text(text + "\n")
        bp.to_csv(CV / f"{key}_by_patient.csv")
        out = EXPORT / key
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(text + "\n")
        bp.to_csv(out / "by_patient.csv")
        print(f"{key}: written to {out}")
    pw = pairwise_section(per_cache)
    if pw:
        (CV / "pairwise_models.md").write_text("\n".join(pw) + "\n")
        EXPORT.mkdir(parents=True, exist_ok=True)
        (EXPORT / "pairwise_models.md").write_text("\n".join(pw) + "\n")


if __name__ == "__main__":
    main()
