"""
Sensitivity of the detection / delineation attribution to the two free parameters of protocol.py (CPU only; the fixed protocol.py is NOT edited).

    python sensitivity.py run [--workers 16] [key ...]     # analyse every (detection threshold, band width) combination for the models, cached
    python sensitivity.py report                           # sensitivity.md / sensitivity_attribution.csv / sensitivity_model_bins.csv

Grid: detection threshold DETECT = 5 / 10 / 20 / 30 % (a region counts as detected when the model marks at least that share of its area) x
band width BAND_UM = 25 / 50 / 100 um. The default (10 %, 50 um) is the one used everywhere else; its numbers are taken from the normal analysis
caches (the same code), the other 11 combinations are computed here by setting protocol.DETECT / protocol.BAND_UM before protocol.analyse is called
(worker processes are forked after the change) and writing the caches under paper_figs/protocol/sensitivity/<key>/. Nothing else differs from protocol47.py.

The pair attribution (public minus site CV, patient-pooled error bins as in rq6_table.md part 2) is recomputed for every combination, for all
patients, biopsies and large sections. Every cell of the sign table says whether a conclusion keeps its sign / its dominant side:
  gap sign, detection component sign, delineation component sign, and which of detection / delineation is larger.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import protocol  # noqa: E402

protocol.INDEX["ours"] = "tiles47.csv"
import export_model as em  # noqa: E402

OUT = P.OUT
SENS = OUT / "sensitivity"
DETS, BANDS = (5, 10, 20, 30), (25, 50, 100)
DEFAULT = (10, 50)
KEYS = ["uni", "conch", "dinov2", "unet", "nnunet", "nnunet_allpublic", "segformer", "segformer_allpublic"]
PAIRS = [("nnU-Net-style 2D", "nnunet", "nnunet_allpublic"), ("SegFormer-B2", "segformer", "segformer_allpublic")]
EXPECT = lambda k: 1 if k.endswith("_allpublic") else 5
BINS = ("missed", "false", "fill", "boundary")
_orig_cache = protocol.cache


def folders(key):
    d = em.EXPORT / key / "predictions"
    return sorted(f.parent for f in d.glob("*/done.json")) if d.exists() else []


def csv_path(key, det, band):
    return SENS / key / f"by_patient_d{det:02d}_b{band:03d}.csv" if (det, band) != DEFAULT else em.CV / f"protocol_{key}_by_patient.csv"


def sens_cache(det, band):
    def cache(folder, per_tile):
        f = Path(folder)
        d = SENS / f.parent.parent.name / "caches"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{f.name}_slides_d{det:02d}_b{band:03d}.csv", d / f"{f.name}_tiles_d{det:02d}_b{band:03d}.csv"
    return cache


def run_combo(key, det, band, workers, tis):
    out = csv_path(key, det, band)
    fs = folders(key)
    if out.exists() or len(fs) < EXPECT(key):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    protocol.DETECT, protocol.BAND_UM = det / 100, band
    protocol.cache = sens_cache(det, band)
    t0 = time.time()
    sl, ti = [], []
    for f in fs:
        s, t = protocol.analyse(f, False, workers)
        sl.append(s)
        ti.append(t)
    m = protocol.measures(pd.concat(sl), pd.concat(ti), "ours", tis)
    m.index.name = "patient"
    m.round(5).to_csv(out)
    protocol.cache = _orig_cache
    print(f"{key} detect {det}% band {band} um: {time.time() - t0:.0f} s", flush=True)


def run(args):
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 16
    keys = [a for a in args if a in KEYS] or KEYS
    tis = em.tissue()
    combos = [(d, b) for b in BANDS for d in DETS if (d, b) != DEFAULT]
    for key in keys:
        for det, band in combos:
            run_combo(key, det, band, workers, tis)


def report():
    from cv_summary import bootstrap_ci
    meta = pd.read_csv(em.CV / "baseline_imagenet_by_patient.csv").set_index("pt")
    rows, mrows = [], []
    for name, site, pub in PAIRS:
        for det in DETS:
            for band in BANDS:
                fs, fp = csv_path(site, det, band), csv_path(pub, det, band)
                if not (fs.exists() and fp.exists()):
                    continue
                s, p = pd.read_csv(fs, index_col=0), pd.read_csv(fp, index_col=0)
                both = s.index.intersection(p.index)
                typ = meta.type.reindex(both)
                for g, idx in [("all patients", both), ("biopsies", both[typ == "biopsy"]), ("large sections", both[typ == "large"])]:
                    d = {b: (p[f"err_{b}"][idx] - s[f"err_{b}"][idx]).dropna() for b in BINS}
                    gap, det_, dl = sum(d.values()), d["missed"] + d["false"], d["fill"] + d["boundary"]
                    lo_g, hi_g = bootstrap_ci(gap)
                    lo_d, hi_d = bootstrap_ci(det_)
                    lo_l, hi_l = bootstrap_ci(dl)
                    rows.append(dict(architecture=name, group=g, detect_pct=det, band_um=band, n=len(idx),
                                     gap=gap.mean(), gap_lo=lo_g, gap_hi=hi_g, detection=det_.mean(), det_lo=lo_d, det_hi=hi_d,
                                     delineation=dl.mean(), del_lo=lo_l, del_hi=hi_l,
                                     missed=d["missed"].mean(), false=d["false"].mean(), fill=d["fill"].mean(), boundary=d["boundary"].mean(),
                                     det_share=det_.mean() / gap.mean() if abs(gap.mean()) > 1e-9 else np.nan,
                                     recall_site=s.recall[idx].mean(), recall_public=p.recall[idx].mean(),
                                     dice_found_site=s.dice_found[idx].mean(), dice_found_public=p.dice_found[idx].mean()))
    for key in KEYS:
        for det in DETS:
            for band in BANDS:
                f = csv_path(key, det, band)
                if f.exists():
                    m = pd.read_csv(f, index_col=0)
                    mrows.append(dict(model=key, detect_pct=det, band_um=band,
                                      **{f"err_{b}": m[f"err_{b}"].mean() for b in BINS}, recall=m.recall.mean(), dice_found=m.dice_found.mean()))
    A, M = pd.DataFrame(rows), pd.DataFrame(mrows)
    A.round(5).to_csv(OUT / "sensitivity_attribution.csv", index=False)
    M.round(5).to_csv(OUT / "sensitivity_model_bins.csv", index=False)
    if A.empty:
        print("nothing to report yet")
        return

    def c(r, k, lo, hi):
        return f"{r[k]:+.3f} [{r[lo]:+.3f}, {r[hi]:+.3f}]"

    L = ["# Sensitivity of the detection / delineation attribution (RQ6 pairs)", "",
         "Free parameters: detection threshold (share of a region's area the model must mark) 5 / 10 / 20 / 30 %, band width 25 / 50 / 100 um; the default is 10 % and 50 um "
         "(bold). Cells: public minus site CV, patient-pooled error added by public training (positive = public training adds error), mean [95% bootstrap CI over patients]. "
         "protocol.py itself is unchanged; the parameters were set before the analysis (sensitivity.py).", ""]
    for name, _, _ in PAIRS:
        for g in ("all patients", "biopsies", "large sections"):
            sub = A[(A.architecture == name) & (A.group == g)]
            if sub.empty:
                continue
            L += [f"## {name}, {g} (n = {int(sub.n.iloc[0])})", "", "| detect | band (um) | Dice gap | detection | delineation | detection share | missed | false far | fill | boundary | region recall site / public | dice_found site / public |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.sort_values(["band_um", "detect_pct"]).iterrows():
                dflt = (r.detect_pct, r.band_um) == DEFAULT
                b = (lambda x: f"**{x}**") if dflt else (lambda x: x)
                L.append("| " + " | ".join(map(b, [f"{int(r.detect_pct)}%", f"{int(r.band_um)}", c(r, 'gap', 'gap_lo', 'gap_hi'), c(r, 'detection', 'det_lo', 'det_hi'),
                                                 c(r, 'delineation', 'del_lo', 'del_hi'), f"{r.det_share:.0%}", f"{r.missed:+.3f}", f"{r.false:+.3f}",
                                                 f"{r.fill:+.3f}", f"{r.boundary:+.3f}", f"{r.recall_site:.3f} / {r.recall_public:.3f}",
                                                 f"{r.dice_found_site:.3f} / {r.dice_found_public:.3f}"])) + " |")
            L.append("")
    # sign table
    L += ["## Does any conclusion change?", "",
          "Each row compares the 12 combinations (or the finished ones) with the default. 'sign' = sign of the mean; 'CI excludes 0' = the 95% CI does not contain zero.", "",
          "| architecture | group | combinations | gap > 0 in all | gap CI excludes 0 in all | detection > 0 in all | delineation > 0 in all | delineation > detection in all / none | range of detection share | combinations where a sign or the dominant side differs from the default |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    any_flip = False
    for name, _, _ in PAIRS:
        for g in ("all patients", "biopsies", "large sections"):
            sub = A[(A.architecture == name) & (A.group == g)]
            if sub.empty:
                continue
            d0 = sub[(sub.detect_pct == DEFAULT[0]) & (sub.band_um == DEFAULT[1])]
            flips = []
            if len(d0):
                d0 = d0.iloc[0]
                for _, r in sub.iterrows():
                    diff = []
                    if np.sign(r.gap) != np.sign(d0.gap): diff.append("gap sign")
                    if np.sign(r.detection) != np.sign(d0.detection): diff.append("detection sign")
                    if np.sign(r.delineation) != np.sign(d0.delineation): diff.append("delineation sign")
                    if (r.delineation > r.detection) != (d0.delineation > d0.detection): diff.append("dominant side")
                    if (r.det_lo > 0) != (d0.det_lo > 0): diff.append("detection CI vs 0")
                    if (r.del_lo > 0) != (d0.del_lo > 0): diff.append("delineation CI vs 0")
                    if diff:
                        flips.append(f"{int(r.detect_pct)}% / {int(r.band_um)} um: " + ", ".join(diff))
            any_flip |= bool(flips)
            dom = (sub.delineation > sub.detection)
            L.append(f"| {name} | {g} | {len(sub)} | {'yes' if (sub.gap > 0).all() else 'no'} | {'yes' if ((sub.gap_lo > 0) | (sub.gap_hi < 0)).all() else 'no'} | "
                     f"{'yes' if (sub.detection > 0).all() else 'no'} | {'yes' if (sub.delineation > 0).all() else 'no'} | "
                     f"{'all' if dom.all() else ('none' if (~dom).all() else 'mixed')} | {sub.det_share.min():.0%} to {sub.det_share.max():.0%} | {'; '.join(flips) or 'none'} |")
    L += ["", "No sign or dominant-side change in any combination." if not any_flip else "At least one conclusion differs from the default in some combination (last column).", ""]
    # model-level bins
    if not M.empty:
        L += ["## Error bins per model across the grid (mean over patients of err_*; 1 - patient-pooled Dice split into the four bins)", "",
              "| model | detect | band (um) | missed | false far | fill | boundary | region recall | dice_found |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in M.sort_values(["model", "band_um", "detect_pct"]).iterrows():
            L.append(f"| {r.model} | {int(r.detect_pct)}% | {int(r.band_um)} | {r.err_missed:.3f} | {r.err_false:.3f} | {r.err_fill:.3f} | {r.err_boundary:.3f} | {r.recall:.3f} | {r.dice_found:.3f} |")
    (OUT / "sensitivity.md").write_text("\n".join(L) + "\n")
    print("written", OUT / "sensitivity.md")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[1:]) if a and a[0] == "run" else report()
