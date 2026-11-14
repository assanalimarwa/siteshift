"""
Threshold analysis on the 47 patients (user 2026-11-10, item 3 of the final local jobs), tile mode, for the site U-Net CV
(5 fold models, each on its own test patients), the all-public U-Net (A) and the Karolinska 3,814 U-Net. The probabilities
come from evaluate.py --save-prob (run_threshold47.py), uint8 per pixel, under $SITESHIFT_PRED/prob47.

    python threshold47.py sweep       # per-slide tp / fp at all 256 uint8 levels -> cv47/threshold/hist_<model>.npz
    python threshold47.py binarize    # protocol-format predictions at tau* and at the site-calibrated thresholds
    python threshold47.py protocol    # protocol.py (parity-verified, Euclidean band) on 0.5, tau* and calibrated
    python threshold47.py report      # -> cv47/threshold_summary.md, threshold_by_patient.csv, threshold_curve.csv

Definitions (fixed 2026-11-10, before any result):
  grid         tau = 0.05, 0.10, ..., 0.95; a pixel is tumour when round(255 p) > 255 tau. At 0.5 this differs from
               evaluate.py's p > 0.5 only for p in (0.498, 0.5]; the report gives the largest per-slide Dice difference
  Dice         per patient = mean of the patient's slide Dice (the headline definition, report47.py)
  tau*         per model, the grid value with the highest mean per-patient Dice over the 47 patients (ties: closest to 0.5)
  E_thr        Dice(tau*) - Dice(0.5), mean over patients. CI: 10,000 patient resamples (seed 0) with tau* chosen again in
               every resample, so the CI includes the choice of tau*. tau* is still scored on the patients it was chosen
               on (in-sample, an upper bound on what a fixed threshold gains); the site-calibrated variant avoids that
  calibrated   per few-shot fold k (data/fewshot/fold<k>_n2.json): tau_k = the grid value with the highest mean Dice on
               the fold's two adaptation patients (patients.train) only, applied to the fold's test patients
               (patients.test, never used for tau_k); the five test sets cover the 47 patients once. For the site U-Net the
               adaptation patients' predictions are the out-of-fold ones (the CV fold model that did not see them)
  attribution  protocol.py error bins per patient (pooled pixels; bins / (2 tp + fp + fn) add up to 1 - pooled Dice) at
               0.5 (the predictions evaluate.py --save-pred saved with the 47-patient evaluations), at tau* and at the calibrated
               thresholds (both binarised from the uint8 probabilities here)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import protocol

protocol.INDEX["ours"] = "tiles47.csv"                      # as eval/protocol47.py: our 47 patients
from report47 import B, INFO, SEED, boot, tissue

HERE = P.INDEX
CV = P.RESULTS / "cv47"                      # result tables (splits: P.INDEX / "cv47")
OUT = CV / "threshold"
PROB = P.PRED / "prob47"
PRED = P.PRED
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
J05 = int(np.argmin(abs(GRID - 0.5)))
BINS = ["missed", "fill", "boundary", "false"]
# key: (name in the report, [(run, tag of its tile-mode evaluation folder eval_<tag>_test_tile)])
MODELS = {"site_unet": ("U-Net site CV (5 folds, out-of-fold)", [(f"cv47/imagenet/fold{k}", f"fold{k}") for k in range(5)]),
          "all_public_unet": ("U-Net all public (A)", [("all_public_unet", "all47")]),
          "karolinska_all_unet": ("U-Net Karolinska 3,814", [("karolinska_all_unet", "all47")])}
KINDS = {"05": "0.5", "tau": "tau*", "cal": "site-calibrated"}


def prob_dirs(key):
    return [PROB / r.replace("/", "_") / "ours47_tile" for r, _ in MODELS[key][1]]


def pred_dirs(key, kind):
    if kind == "05":
        return [PRED / r.replace("/", "_") / "ours47_tile" for r, _ in MODELS[key][1]]
    return [PRED / "threshold47" / f"{key}_{kind}_tile"]


def eval_dirs(key):
    return [P.RUNS / r / f"eval_{t}_test_tile" for r, t in MODELS[key][1]]


def level(tau):
    """Pixels with uint8 probability above this value are tumour at threshold tau."""
    return np.floor(255 * np.asarray(tau) + 1e-9).astype(int)


# ---------------------------------------------------------------- sweep
def slide_hist(path):
    z = np.load(path)
    q, n = z["prob"], len(z["prob"])
    px = q.shape[1] * q.shape[2]
    truth = np.unpackbits(z["truth"], axis=1)[:, :px]
    h = np.zeros(512, np.int64)
    for i in range(0, n, 128):
        h += np.bincount(q[i:i + 128].reshape(-1).astype(np.int16) * 2 + truth[i:i + 128].reshape(-1), minlength=512)
    return str(z["slide"]), h[1::2], h[0::2]                  # per uint8 level: tumour pixels, other pixels


def sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        dst = OUT / f"hist_{key}.npz"
        if dst.exists():
            continue
        for d in prob_dirs(key):
            assert (d / "done.json").exists(), f"{d} has no finished probabilities"
        files = [f for d in prob_dirs(key) for f in sorted(d.glob("*.npz"))]
        t0 = time.time()
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(slide_hist, files))
        slides = np.array([r[0] for r in res])
        assert len(set(slides)) == len(slides) == 194, f"{key}: {len(slides)} slides"
        np.savez(dst, slide=slides, h1=np.stack([r[1] for r in res]), h0=np.stack([r[2] for r in res]))
        print(f"{key}: {len(files)} slides, {time.time() - t0:.0f} s -> {dst}", flush=True)


def slide_dice(key):
    z = np.load(OUT / f"hist_{key}.npz")
    above1 = np.cumsum(z["h1"][:, ::-1], 1)[:, ::-1]           # [:, j] = pixels at level >= j
    above0 = np.cumsum(z["h0"][:, ::-1], 1)[:, ::-1]
    c = level(GRID) + 1
    tp, fp = above1[:, c], above0[:, c]
    fn = z["h1"].sum(1)[:, None] - tp
    return pd.DataFrame(2 * tp / np.maximum(2 * tp + fp + fn, 1), index=pd.Index(z["slide"], name="slide"), columns=GRID)


def patient_dice(key):
    d = slide_dice(key)
    return d.groupby(d.index.str.split(".").str[0].astype(int)).mean()


def pick(m):
    """Index of the best grid value of a mean-Dice curve (ties: the one closest to 0.5)."""
    m = np.asarray(m)
    best = np.flatnonzero(m >= m.max() - 1e-12)
    return int(best[np.argmin(abs(GRID[best] - 0.5))])


def calibration(D):
    folds, seen = {}, []
    for k in range(5):
        s = json.loads((HERE / "fewshot" / f"fold{k}_n2.json").read_text())["patients"]
        folds[k] = dict(adapt=s["train"], test=s["test"], tau=float(GRID[pick(D.loc[s["train"]].mean(0))]))
        assert not set(s["train"]) & set(s["test"])
        seen += s["test"]
    assert sorted(seen) == sorted(D.index), "the few-shot test sets do not cover the 47 patients once"
    return folds


def thresholds(key):
    D = patient_dice(key)
    cal = calibration(D)
    return float(GRID[pick(D.mean(0))]), cal, {p: f["tau"] for f in cal.values() for p in f["test"]}


# ---------------------------------------------------------------- binarize + protocol
def _binarize(path, dst, taus):
    z = np.load(path)
    s = str(z["slide"])
    q = z["prob"].reshape(len(z["prob"]), -1)
    pred = q > level(taus[int(s.split(".")[0])])
    np.savez_compressed(dst / path.name, slide=s, size=z["size"], image=z["image"], pred=np.packbits(pred, axis=1),
                        truth=z["truth"], tissue=z["tissue"])


def binarize():
    for key in MODELS:
        tau, _, cal = thresholds(key)
        for kind, taus in [("tau", {p: tau for p in cal}), ("cal", cal)]:
            dst = pred_dirs(key, kind)[0]
            if (dst / "done.json").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            files = [f for d in prob_dirs(key) for f in sorted(d.glob("*.npz"))]
            t0 = time.time()
            with ProcessPoolExecutor(4) as ex:
                list(ex.map(_binarize, files, [dst] * len(files), [taus] * len(files)))
            (dst / "done.json").write_text(json.dumps(dict(
                by="threshold47.py binarize", kind=KINDS[kind], tau_by_patient={str(p): t for p, t in sorted(taus.items())},
                source=[str(d) for d in prob_dirs(key)], index=str(HERE / "tiles47.csv"), size=512, context=0, tta=False,
                slides=len(files)), indent=2))
            print(f"{key} {kind}: {len(files)} slides, {time.time() - t0:.0f} s -> {dst}", flush=True)
    (OUT / "binarize.DONE").write_text(time.strftime("%Y-%m-%d %H:%M"))


def run_protocol():
    for key in MODELS:
        for kind in KINDS:
            for f in pred_dirs(key, kind):
                if not protocol.is_cached(f, False):
                    t0 = time.time()
                    slides, _ = protocol.analyse(f)
                    print(f"{f}: {len(slides)} slides, {time.time() - t0:.0f} s", flush=True)
    (OUT / "protocol.DONE").write_text(time.strftime("%Y-%m-%d %H:%M"))


def attribution(folders):
    s = pd.concat([protocol.analyse(f)[0] for f in folders])
    t = pd.concat([protocol.analyse(f)[1] for f in folders])
    assert not s.slide.duplicated().any() and len(s) == 194, len(s)
    return protocol.measures(s, t, "ours", tissue())


# ---------------------------------------------------------------- report
def paired(d):
    d = pd.Series(d).dropna()
    lo, hi = boot(d)
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]", f"{(d > 0).sum()}/{(d < 0).sum()}", f"{p:.3g}"


def mean_ci(v):
    v = pd.Series(v).dropna()
    lo, hi = boot(v)
    return f"{v.mean():.4f} [{lo:.4f}, {hi:.4f}]"


def e_thr_bootstrap(D):
    """Selection-aware bootstrap of E_thr: tau* is chosen again in every patient resample."""
    X = D.values
    idx = np.random.default_rng(SEED).integers(0, len(X), (B, len(X)))
    M = X[idx].mean(1)
    js = np.array([pick(m) for m in M])
    E = M[np.arange(B), js] - M[:, J05]
    return np.percentile(E, [2.5, 97.5]), float((js == J05).mean()), pd.Series(GRID[js]).value_counts().sort_index()


def eval_check(key):
    """Largest per-slide difference between the sweep's Dice at 0.5 (uint8) and evaluate.py's per_slide.csv."""
    ev = pd.concat([pd.read_csv(e / "per_slide.csv") for e in eval_dirs(key)]).set_index("slide").dice
    d = slide_dice(key)[GRID[J05]]
    return float((d - ev.reindex(d.index)).abs().max())


def report():
    lines = ["# Threshold analysis, 47 patients, tile mode", "",
             f"_threshold47.py report, {pd.Timestamp.now():%Y-%m-%d %H:%M}. Definitions: threshold47.py docstring "
             "(fixed before any result). Per-patient Dice = mean of the patient's slide Dice; 95% CIs: 10,000 patient "
             "resamples, seed 0; paired differences with patients better/worse and Wilcoxon signed-rank p._", "",
             "tau grid 0.05-0.95 in steps of 0.05; tau* = the grid value with the best mean per-patient Dice over all 47 "
             "patients (in-sample). E_thr = Dice(tau*) - Dice(0.5). The selection-aware CI chooses tau* again in every "
             "resample; 'tau* = 0.5' gives the share of resamples in which 0.5 itself was best. The site-calibrated "
             "threshold is chosen per few-shot fold on its two adaptation patients only and applied to that fold's "
             "unseen test patients (out-of-sample).", ""]
    rows, cal_rows, curve, attr_rows, attr_diff, spec_rows, by_patient = [], [], {}, [], [], [], []
    for key, (name, _) in MODELS.items():
        D = patient_dice(key)
        tau, cal, cal_p = thresholds(key)
        j = int(np.flatnonzero(GRID == tau)[0])
        curve[name] = D.mean(0)
        (lo, hi), share05, dist = e_thr_bootstrap(D)
        e = D[GRID[j]] - D[GRID[J05]]
        de, bw, p = paired(e)
        rows.append({"model": name, "tau*": f"{tau:.2f}", "Dice at 0.5": mean_ci(D[GRID[J05]]),
                     "Dice at tau*": mean_ci(D[GRID[j]]), "E_thr (fixed-tau* CI)": de,
                     "E_thr selection-aware 95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "better/worse": bw, "Wilcoxon p": p,
                     "resamples with tau* = 0.5": f"{100 * share05:.0f}%",
                     "tau* over resamples": ", ".join(f"{t:.2f}: {100 * n / B:.0f}%" for t, n in dist.items() if n / B >= 0.01),
                     "max slide |Dice uint8 - evaluate.py| at 0.5": f"{eval_check(key):.1e}"})
        dc = pd.Series({pt: D.loc[pt, cal_p[pt]] for pt in D.index})
        d1, bw1, p1 = paired(dc - D[GRID[J05]])
        d2, bw2, p2 = paired(dc - D[GRID[j]])
        cal_rows.append({"model": name, "tau_k folds 0-4": ", ".join(f"{cal[k]['tau']:.2f}" for k in range(5)),
                         "adaptation patients": "; ".join(",".join(map(str, cal[k]["adapt"])) for k in range(5)),
                         "Dice at 0.5": mean_ci(D[GRID[J05]]), "Dice calibrated": mean_ci(dc),
                         "calibrated - 0.5": d1, "better/worse": bw1, "Wilcoxon p": p1,
                         "calibrated - tau* (in-sample)": d2, "better/worse vs tau*": bw2})
        info = D.join(INFO)
        for typ in ["biopsy", "large"]:
            x = info[info.type == typ]
            dd, bwt, pt_ = paired(x[GRID[j]] - x[GRID[J05]])
            dct = paired(dc.loc[x.index] - x[GRID[J05]])
            spec_rows.append({"model": name, "specimen": typ, "patients": len(x), "Dice at 0.5": mean_ci(x[GRID[J05]]),
                              "E_thr (tau*)": dd, "better/worse": bwt, "calibrated - 0.5": dct[0], "better/worse (cal)": dct[1]})
        A = {kind: attribution(pred_dirs(key, kind)) for kind in KINDS}
        for kind, a in A.items():
            attr_rows.append({"model": name, "threshold": KINDS[kind] + (f" ({tau:.2f})" if kind == "tau" else ""),
                              "patient-pooled Dice": f"{a.dice.mean():.4f}",
                              **{f"err {b}": f"{a[f'err_{b}'].mean():.4f}" for b in BINS},
                              **{f"share {b}": f"{100 * a[f'share_{b}'].mean():.0f}%" for b in BINS},
                              "false isolated / attached": f"{a.err_false_isolated.mean():.4f} / {a.err_false_attached.mean():.4f}",
                              "recall (regions)": f"{a.recall.mean():.3f}", "clean-tile AUC": f"{a.clean_auc.mean():.3f}"})
        for kind in ["tau", "cal"]:
            r = {"model": name, "change": f"{KINDS[kind]} - 0.5"}
            for b in BINS:
                r[f"err {b}"], r[f"{b} better/worse"], _ = paired(A["05"][f"err_{b}"] - A[kind][f"err_{b}"])
            attr_diff.append(r)
        bp = pd.DataFrame({"p": D.index, "model": key, "dice_05": D[GRID[J05]].values, "tau_star": tau,
                           "dice_tau": D[GRID[j]].values, "tau_cal": [cal_p[x] for x in D.index], "dice_cal": dc.values})
        for kind, a in A.items():
            for b in BINS:
                bp[f"err_{b}_{kind}"] = a[f"err_{b}"].reindex(D.index).values
        by_patient.append(bp.join(INFO, on="p"))

    def table(rs):
        cols = list(rs[0])
        return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rs]

    curve = pd.DataFrame(curve).rename_axis("tau")
    lines += ["## Results: E_thr (one threshold per model)", "", *table(rows), "",
              "## Site-calibrated threshold (chosen on 2 adaptation patients per fold, applied to the fold's unseen patients)", "",
              *table(cal_rows), "", "## By specimen type", "", *table(spec_rows), "",
              "## Attribution at 0.5, tau* and the site-calibrated thresholds (protocol.py, parity-verified Euclidean band)", "",
              "err = bin / (2 tp + fp + fn) per patient (pooled pixels), mean over patients; the four err bins add up to "
              "1 - patient-pooled Dice. share = bin / (fp + fn).", "", *table(attr_rows), "",
              "Change in each error bin (0.5 minus the other threshold: positive = that error shrinks), paired over patients:", "",
              *table(attr_diff), "",
              "## Mean per-patient Dice over the grid", "",
              "| tau | " + " | ".join(curve.columns) + " |", "|" + "---|" * (len(curve.columns) + 1)]
    lines += [f"| {t:.2f} | " + " | ".join(f"{v:.4f}" for v in curve.loc[t]) + " |" for t in curve.index]
    (CV / "threshold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.concat(by_patient).to_csv(CV / "threshold_by_patient.csv", index=False)
    curve.to_csv(CV / "threshold_curve.csv")
    print(f"-> {CV / 'threshold_summary.md'}")


if __name__ == "__main__":
    {"sweep": sweep, "binarize": binarize, "protocol": run_protocol, "report": report}[sys.argv[1]]()
