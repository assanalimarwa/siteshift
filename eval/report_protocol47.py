"""
47-patient detection / delineation protocol report (folder 09; user 2026-11-10, item 5 of the final local jobs), from the
parity-verified protocol.py caches (protocol_*_euclid.csv, eval/protocol_cache47.py) of every local prediction folder:

    python report_protocol47.py      # -> cv47/protocol_summary.md, cv47/protocol_by_patient.csv

Unit = patient (pooled pixels of its slides), so the four error bins add up to 1 - patient-pooled Dice (the protocol's Dice;
the headline slide-mean Dice is in the other reports). In the form of the H100's rq6_report.py: the gap to the site U-Net
(same patients, same mode) = public / few-shot minus site error, split into detection (missed regions + false tumour
farther than 50 um from any tumour) and delineation (unmarked interior of found regions + the +-50 um band). Mean over
patients with a 95% bootstrap CI (10,000 patient resamples, seed 0); (n / n) = patients with more / less error than the
site U-Net. Few-shot: each fold model on its own 9-10 test patients, the five folds cover the 47 once.
H100 models (UNI / CONCH / DINOv2-L / Virchow2 site CV, nnU-Net-style and SegFormer-B2 site CV and all-public): the
protocol-format exports transferred from the server (2026-11-12, every file checked against its manifest SHA-256), copied
to this machine and analysed with the same protocol.py.
Architecture-matched gap (added 2026-11-12): a public model minus the site CV of the same architecture (ARCH below).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
from pathlib import Path

import numpy as np
import pandas as pd

import protocol

protocol.INDEX["ours"] = "tiles47.csv"                      # as eval/protocol47.py: our 47 patients
from report47 import INFO, boot, tissue

HERE = P.INDEX
CV = P.RESULTS / "cv47"                      # result tables (splits: P.INDEX / "cv47")
PRED = P.PRED                                 # 4090 runs: <PRED>/<run with / as _>/ours47_tile (evaluate.py --save-pred)
H100_PRED = P.PRED                            # H100 runs: <PRED>/<model>/predictions/<fold or all> (protocol_predict.py)
BINS = ["missed", "fill", "boundary", "false"]
MODES = {"tile": "ours47_tile", "full": "ours47_ctx256_blend_tta"}
NAMES = {"sicap_pretrain": "SICAPv2 U-Net", "sicap_pretrain_seed1": "SICAPv2 U-Net (seed 1)", "sicap_r40": "SICAPv2 U-Net r40",
         "sicap_r80": "SICAPv2 U-Net r80", "karolinska_n20_unet": "Karolinska 20 U-Net", "karolinska_n155_unet": "Karolinska 155 U-Net",
         "karolinska_n419_unet": "Karolinska 419 U-Net", "karolinska_n600_unet": "Karolinska 600 U-Net",
         "karolinska_n1000_unet": "Karolinska 1,000 U-Net", "karolinska_all_unet": "Karolinska 3,814 U-Net",
         "mixed600_unet": "mixed 600 U-Net", "all_public_unet": "all-public (A) U-Net", "radboud_unet": "Radboud U-Net",
         "sicap_fm_uni": "SICAPv2 UNI", "sicap_fm_virchow2": "SICAPv2 Virchow2", "sicap_fm_conch": "SICAPv2 CONCH",
         "sicap_fm_dinov2": "SICAPv2 DINOv2", "sicap_fm_unimedclip": "SICAPv2 UniMed-CLIP", "all_public_uni": "all-public UNI",
         "mixed600_uni": "mixed 600 UNI", "karolinska_n600_uni": "Karolinska 600 UNI"}
FEWSHOT = {"kar": ("U-Net from Karolinska 3,814", "karolinska_all_unet"), "imagenet": ("U-Net from ImageNet", None),
           "uni_scratch": ("UNI, fresh head", None), "uni_public": ("UNI from all-public", "all_public_uni")}
H100 = {"uni": "UNI site CV (H100)", "conch": "CONCH site CV (H100)", "dinov2": "DINOv2-L site CV (H100)",
        "virchow2": "Virchow2 site CV (H100)", "nnunet": "nnU-Net-style site CV (H100)", "segformer": "SegFormer-B2 site CV (H100)",
        "nnunet_allpublic": "nnU-Net-style all-public (H100)", "segformer_allpublic": "SegFormer-B2 all-public (H100)"}
# architecture-matched site model of each public model (U-Nets: the U-Net site CV; UNI heads trained on the 4090: the 4090 UNI site CV)
ARCH = {"nnunet_allpublic": "nnunet", "segformer_allpublic": "segformer", "sicap_fm_uni": "uni_site", "all_public_uni": "uni_site",
        "mixed600_uni": "uni_site", "karolinska_n600_uni": "uni_site", "sicap_fm_virchow2": "virchow2", "sicap_fm_conch": "conch",
        "sicap_fm_dinov2": "dinov2"}
SUBSETS = {"all patients": None, "biopsies": "biopsy", "large sections": "large"}


def models():
    """(key, name, group, {mode: [prediction folders]}) of every local model whose folders are all cached."""
    out = [("unet_site", "U-Net site CV", "site", {m: [PRED / f"cv47_imagenet_fold{k}" / d for k in range(5)] for m, d in MODES.items()}),
           ("uni_site", "UNI site CV (frozen + head)", "site", {"tile": [PRED / f"cv47_fm_uni_fold{k}" / MODES["tile"] for k in range(5)]})]
    out += [(r, n, "public", {m: [PRED / r / d] for m, d in MODES.items()}) for r, n in NAMES.items()]
    for key, name in H100.items():
        runs = ["all"] if key.endswith("allpublic") else [f"fold{k}" for k in range(5)]
        out.append((key, name, "public" if key.endswith("allpublic") else "site", {"tile": [H100_PRED / key / "predictions" / r for r in runs]}))
    for init, (name, _) in FEWSHOT.items():
        for n in [2, 4, 8]:
            out.append((f"fewshot_{init}_n{n}", f"few-shot {name}, {n} patients", "few-shot",
                        {"tile": [PRED / f"fewshot47_{init}_n{n}_fold{k}" / MODES["tile"] for k in range(5)]}))
    keep = []
    for key, name, group, f in out:
        f = {m: fs for m, fs in f.items() if all((x / "done.json").exists() for x in fs)}
        if f:
            keep.append((key, name, group, f))
    return keep


def measures(folders):
    missing = [str(f) for f in folders if not protocol.is_cached(f, False)]
    if missing:
        raise SystemExit(f"not analysed yet (eval/protocol_cache47.py): {missing}")
    s = pd.concat([protocol.analyse(f)[0] for f in folders])
    t = pd.concat([protocol.analyse(f)[1] for f in folders])
    assert not s.slide.duplicated().any()
    m = protocol.measures(s, t, "ours", tissue())
    m["detection"] = m.err_missed + m.err_false
    m["delineation"] = m.err_fill + m.err_boundary
    return m


def ci(v, digits=3):
    v = pd.Series(v).dropna()
    lo, hi = boot(v)
    return f"{v.mean():.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def gap(v):
    v = pd.Series(v).dropna()
    lo, hi = boot(v)
    return f"{v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] ({(v > 0).sum()} / {(v < 0).sum()})"


def md(rows):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows])


def summary_row(name, group, mode, m):
    return {"model": name, "group": group, "mode": mode, "patients": int(m.dice.notna().sum()),
            "Dice (patient-pooled)": ci(m.dice), **{f"err {b}": ci(m[f"err_{b}"]) for b in BINS},
            "detection (missed + false)": ci(m.detection), "delineation (fill + boundary)": ci(m.delineation),
            "share missed / fill / boundary / false": " / ".join(f"{100 * m[f'share_{b}'].mean():.0f}%" for b in BINS),
            "false: isolated": ci(m.err_false_isolated), "false: attached": ci(m.err_false_attached),
            "region recall": ci(m.recall), "false regions per mm^2": f"{m.false_regions_mm2.mean():.2f}",
            "dice_found": ci(m.dice_found), "dice_tolerant": ci(m.dice_tolerant), "clean-tile AUC": ci(m.clean_auc)}


def gap_row(name, mode, m, site, idx=None, label=None):
    idx = m.index.intersection(site.index) if idx is None else idx
    d = {b: m[f"err_{b}"].reindex(idx) - site[f"err_{b}"].reindex(idx) for b in BINS}
    total = sum(d.values())
    det, dl = d["missed"] + d["false"], d["fill"] + d["boundary"]
    share = f"{det.mean() / total.mean():.0%} / {dl.mean() / total.mean():.0%}" if abs(total.mean()) > 1e-9 else "n/a"
    r = {"model": name, "mode": mode}
    if label:
        r["group"] = label
    r.update({"n": len(idx), "Dice gap = added error [CI] (more / less error)": gap(total),
              "detection": gap(det), "- missed": gap(d["missed"]), "- false far from tumour": gap(d["false"]),
              "  - false: isolated": gap(m.err_false_isolated.reindex(idx) - site.err_false_isolated.reindex(idx)),
              "  - false: attached": gap(m.err_false_attached.reindex(idx) - site.err_false_attached.reindex(idx)),
              "delineation": gap(dl), "- fill": gap(d["fill"]), "- boundary": gap(d["boundary"]),
              "detection / delineation share of the gap": share})
    return r


def main():
    ms = models()
    M = {(key, mode): measures(fs) for key, _, _, f in ms for mode, fs in f.items()}
    site = {mode: M[("unet_site", mode)] for mode in MODES}
    typ = INFO.type
    lines = ["# Detection / delineation protocol, 47 patients (parity-verified protocol.py)", "",
             f"_report_protocol47.py, {pd.Timestamp.now():%Y-%m-%d %H:%M}. Rules: protocol.py docstring (fixed 2026-10-29; band, "
             "outline and small-region rule as in the H100 parity check of 2026-11-08: EXACT on 10 of 10 slide x model "
             "comparisons with the reference attribution.py; local protocol.py = the server's, SHA-256 8acab1f1..., plus one line "
             "mapping tiles47.csv). Unit = patient, pooled pixels: the bins add up to 1 - "
             "patient-pooled Dice. Mean over patients [95% bootstrap CI, 10,000 resamples, seed 0]._", "",
             "False tumour (the 'false' bin: predicted tumour farther than 50 um from any annotated tumour) is split as in "
             "protocol.measures (post hoc split of 2026-10-30, same code as the parity-verified bins): **isolated** = pixels of "
             "predicted components (8-connected, stitched slide) with no pixel within 50 um of annotated tumour; **attached** = "
             "the rest of the bin, i.e. the part of components that reach annotated tumour or its 50 um band (over-painting "
             "spill). isolated + attached = err false, per patient.", "",
             "H100 models: transferred from the server (every file matched its manifest SHA-256; truth masks, tile "
             "layout and folds identical to the local exports), analysed on this PC with the same protocol.py. "
             + ("**Missing H100 models: " + ", ".join(k for k in H100 if k not in {m[0] for m in ms}) + "**" if
                any(k not in {m[0] for m in ms} for k in H100) else "All eight H100 models are in."), ""]
    by_patient = []
    for mode in MODES:
        for sub, t in SUBSETS.items():
            sel = lambda m: m if t is None else m.loc[m.index[typ.reindex(m.index) == t]]
            rows = [summary_row(name, group, mode, sel(M[(key, mode)])) for key, name, group, f in ms if mode in f]
            lines += [f"## Results, {mode} mode, {sub}", "", md(rows), ""]
        for sub, t in SUBSETS.items():
            sel = lambda m: m.index if t is None else m.index[typ.reindex(m.index) == t]
            grows = [gap_row(name, mode, M[(key, mode)], site[mode], sel(M[(key, mode)]).intersection(site[mode].index))
                     for key, name, group, f in ms if mode in f and key != "unet_site"]
            lines += [f"## Gap to the site U-Net, {mode} mode, {sub}: detection vs delineation (model minus site U-Net; "
                      "positive = more error)", "", md(grows), ""]
        if mode == "tile":
            for sub, t in SUBSETS.items():
                arows = []
                for key, name, group, f in ms:
                    ref = ARCH.get(key)
                    if ref is None or (ref, "tile") not in M or (key, "tile") not in M:
                        continue
                    m, r = M[(key, "tile")], M[(ref, "tile")]
                    idx = m.index.intersection(r.index)
                    idx = idx if t is None else idx[typ.reindex(idx) == t]
                    g = gap_row(name, "tile", m, r, idx)
                    arows.append({"model": name, "matched site model": dict((k, n) for k, n, _, _ in ms)[ref],
                                  **{c: v for c, v in g.items() if c not in ("model", "mode")}})
                lines += [f"## Architecture-matched gap, tile mode, {sub} (public model minus the site CV of the same "
                          "architecture; U-Nets: see the gap to the site U-Net above)", "", md(arows), ""]
        for key, name, group, f in ms:
            if mode in f:
                by_patient.append(M[(key, mode)].assign(model=key, name=name, group=group, mode=mode).rename_axis("p").reset_index())
    # per specimen type (tile mode; full mode in the CSV)
    srows = []
    for key, name, group, f in ms:
        if "tile" not in f:
            continue
        m = M[(key, "tile")]
        for t in ["biopsy", "large"]:
            idx = m.index[typ.reindex(m.index) == t]
            x = m.loc[idx]
            r = {"model": name, "specimen": t, "n": len(idx), "Dice (patient-pooled)": ci(x.dice),
                 **{f"err {b}": f"{x[f'err_{b}'].mean():.3f}" for b in BINS}}
            if key != "unet_site":
                g = gap_row(name, "tile", m, site["tile"], idx)
                r["gap: detection"], r["gap: delineation"] = g["detection"], g["delineation"]
                r["detection / delineation share"] = g["detection / delineation share of the gap"]
            srows.append(r)
    lines += ["## Per specimen type, tile mode (gap = model minus site U-Net on the same patients)", "", md(srows), ""]
    # few-shot: the two-stage reading (does adaptation remove detection error first, delineation later?)
    crows = []
    for init, (name, start) in FEWSHOT.items():
        for n in [2, 4, 8]:
            m = M.get((f"fewshot_{init}_n{n}", "tile"))
            if m is None:
                continue
            g = gap_row(name, "tile", m, site["tile"])
            crows.append({"start": name, "our patients": n, "n test patients": g["n"],
                          "gap to site U-Net (added error)": g["Dice gap = added error [CI] (more / less error)"],
                          "detection part [CI]": g["detection"], "delineation part [CI]": g["delineation"],
                          "detection / delineation share": g["detection / delineation share of the gap"]})
    lines += ["## Few-shot attribution (the two-stage claim), tile mode", "",
              "Gap of each few-shot model to the site U-Net on the patients it never saw (each fold model on its own 9-10 test "
              "patients; the five folds cover all 47 once), split into detection (missed + false far from tumour) and delineation "
              "(fill + boundary). Mean over patients [95% CI] (patients with more / less error than the site U-Net).", "",
              md(crows), ""]
    frows = []
    for init, (name, start) in FEWSHOT.items():
        steps = ([(f"0 (public start: {NAMES[start]})", M.get((start, "tile")))] if start else []) + \
                [(str(n), M.get((f"fewshot_{init}_n{n}", "tile"))) for n in [2, 4, 8]] + [("site CV (35-36)", site["tile"])]
        prev = None
        for n, m in steps:
            if m is None:
                continue
            r = {"start": name, "our patients": n, "Dice (patient-pooled)": ci(m.dice), "detection error": ci(m.detection),
                 "delineation error": ci(m.delineation)}
            if not n.startswith("site"):
                g = gap_row(name, "tile", m, site["tile"])
                r["gap to site: detection"], r["gap to site: delineation"] = g["detection"], g["delineation"]
                r["detection / delineation share of the gap"] = g["detection / delineation share of the gap"]
            if prev is not None:
                idx = m.index.intersection(prev.index)
                r["change from previous row: detection"] = gap(m.detection[idx] - prev.detection[idx])
                r["change from previous row: delineation"] = gap(m.delineation[idx] - prev.delineation[idx])
            frows.append(r)
            prev = m
    lines += ["## Few-shot adaptation: detection vs delineation as our patients are added (tile mode)", "",
              "Rows per starting point: public model alone (where there is one), then 2 / 4 / 8 of our patients (5 folds, each "
              "fold model on its own test patients), then the site CV. 'change from previous row' = this row minus the "
              "previous one, paired over patients (negative = error removed; (n / n) = patients with more / less error).", "",
              md(frows), ""]
    (CV / "protocol_summary.md").write_text("\n".join(lines), encoding="utf-8")
    pd.concat(by_patient).join(INFO, on="p").to_csv(CV / "protocol_by_patient.csv", index=False)
    print(f"-> {CV / 'protocol_summary.md'} ({len(ms)} models)")


if __name__ == "__main__":
    main()
