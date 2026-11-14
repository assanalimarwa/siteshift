"""
Leaderboard baselines on the withheld fold (fold 2: patients 6, 10, 23, 25, 28, 30, 34, 36, 38, 46; 5 biopsy, 5 large-section),
from the released per-patient results, without new predictions:

    python leaderboard_fold2.py      # -> results/leaderboard_fold2.md, results/leaderboard_fold2.csv

Every number is a mean over fold-2 patients only, tile mode, threshold 0.5:
  Dice           per-patient Dice = mean of the patient's slide Dice (the headline), 95% bootstrap CI over the patients
                 (10,000 resamples, seed 0)
  vs site U-Net  paired difference to the site-CV U-Net on the same patients [95% CI] (better / worse patients)
  AUC            clean-tile AUC, mean over patients
  detection / delineation
                 protocol.py error bins (patient-pooled pixels): missed + false tumour far from tumour / unmarked interior +
                 boundary band, mean over patients (they add up to 1 - patient-pooled Dice)
Tracks: A = public-only models on the 10 patients; B = the same on the 5 large-section patients; C = few-shot models of
fold 2 (adapted on fold 2's own draws data/fewshot/fold2_n<n>.json, whose patients are outside fold 2), scored on the
10 patients. Site-CV fold-2 models are a reference (trained on the other 37 patients), not a track.
Interval widths are measured and printed under each table: with 10 patients a single model's 95% Dice interval is
about +-0.14 (Track B, 5 patients: about +-0.24); paired differences to the site U-Net are narrower (about +-0.07).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json

import numpy as np
import pandas as pd

from cv_summary import bootstrap_ci
from export_model import md
from report47 import PUBLIC_LABELS

CV = P.RESULTS / "cv47"
FOLD = 2
FEWSHOT = {"kar": "U-Net from Karolinska 3,814", "imagenet": "U-Net from ImageNet", "uni_scratch": "UNI fresh head",
           "uni_public": "UNI from all-public UNI"}
H100 = ["uni", "conch", "dinov2", "virchow2", "nnunet", "segformer", "nnunet_allpublic", "segformer_allpublic"]
TRAINING = {"site": "site CV fold 2 (the other 37 patients: train + val)", "public": "public data only", "few-shot": "start model + n fold-2 adaptation patients"}


def dice_sources():
    """key -> per-patient DataFrame (index p; dice, dice_pooled, clean_auc)."""
    pub = pd.read_csv(CV / "public_by_patient.csv")
    pub = pub[pub["mode"] == "tile"]
    out = {"unet_site": pub[pub.model == "U-Net site CV (baseline)"].set_index("p")}
    out["uni_site"] = pd.read_csv(CV / "uni_cv_by_patient.csv").query("mode == 'tile'").set_index("p")
    for key, label in PUBLIC_LABELS.items():
        out[key] = pub[pub.model == label].set_index("p")
    for key in H100:
        d = pd.read_csv(CV / f"{key}_by_patient.csv").rename(columns={"patient": "p"}).set_index("p")
        out[key] = d[["dice", "dice_pooled", "clean_auc"]]
    few = pd.read_csv(CV / "fewshot_by_patient.csv")
    for init, label in FEWSHOT.items():
        for n in (2, 4, 8):
            out[f"fewshot_{init}_n{n}"] = few[few.model == f"{label} + {n} patients"].set_index("p")
    return out


def table(keys, names, groups, src, prot, patients, ref):
    rows = []
    for key in keys:
        d = src[key].reindex(patients)
        assert d.dice.notna().all(), (key, "patients missing")
        lo, hi = bootstrap_ci(d.dice.values)
        diff = (d.dice - ref.reindex(patients).dice).values
        dlo, dhi = bootstrap_ci(diff)
        pp = prot[prot.model == key].set_index("p").reindex(patients)
        rows.append({"model": names[key], "training data": TRAINING[groups[key]], "patients": len(patients),
                     "Dice [95% CI]": f"{d.dice.mean():.3f} [{lo:.3f}, {hi:.3f}]",
                     "vs site U-Net [95% CI] (better / worse)": "reference" if key == "unet_site" else
                     f"{diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}] ({(diff > 0).sum()} / {(diff < 0).sum()})",
                     "patient-pooled Dice": f"{d.dice_pooled.mean():.3f}", "clean-tile AUC": f"{np.nanmean(d.clean_auc):.3f}",
                     "detection error": f"{pp.detection.mean():.3f}", "delineation error": f"{pp.delineation.mean():.3f}",
                     "_dice": d.dice.mean(), "_key": key})
    t = pd.DataFrame(rows).sort_values("_dice", ascending=False)
    t.insert(0, "rank", range(1, len(t) + 1))
    return t


def main():
    fold = json.loads((P.INDEX / "cv47" / f"fold{FOLD}.json").read_text())
    patients = sorted(fold["patients"]["test"])
    info = pd.read_csv(CV / "dataset_by_patient.csv").set_index("p")
    assert patients == [6, 10, 23, 25, 28, 30, 34, 36, 38, 46], patients
    large = [p for p in patients if info.type[p] == "large"]
    prot = pd.read_csv(CV / "protocol_by_patient.csv").query("mode == 'tile'")
    names = prot.drop_duplicates("model").set_index("model").name.to_dict()
    groups = prot.drop_duplicates("model").set_index("model").group.to_dict()
    src = dice_sources()
    ref = src["unet_site"]
    public = [k for k in names if groups[k] == "public"]
    few = [k for k in names if groups[k] == "few-shot"]
    site = [k for k in names if groups[k] == "site"]
    parts = [("Track A: cross-site transfer (public data only), fold 2, all 10 patients", public, patients),
             ("Track B: large sections, fold 2, the 5 large-section patients", public, large),
             ("Track C: few-shot adaptation on fold 2's own draws (2 / 4 / 8 patients outside fold 2), scored on fold 2", few, patients),
             ("Reference (not a track): site-CV fold-2 models, trained on the other 37 patients", site, patients)]
    lines = ["# Leaderboard baselines on the withheld fold (fold 2)", "",
             f"_leaderboard_fold2.py. Fold 2 = patients {', '.join(map(str, patients))} ({len(patients) - len(large)} biopsy, "
             f"{len(large)} large-section: {', '.join(map(str, large))}). Definitions in the script's docstring; tile mode, threshold 0.5; "
             "computed from the released per-patient results (results/cv47), the same runs as the 47-patient tables._", "",
             "**Small test set.** {SMALL} Differences below 0.03 are ties under the benchmark rules, and only differences whose "
             "paired interval excludes 0 count. Use these rows as baselines to beat, not as a ranking of the baselines.", ""]
    csv, half = [], {}
    for title, keys, pts in parts:
        t = table(keys, names, groups, src, prot, pts, ref)
        widths = [float(x.split("[")[1].split(",")[1].rstrip("]")) - float(x.split("[")[1].split(",")[0]) for x in t["Dice [95% CI]"]]
        pw = [float(x.split("[")[1].split(",")[1].split("]")[0]) - float(x.split("[")[1].split(",")[0])
              for x in t["vs site U-Net [95% CI] (better / worse)"] if x != "reference"]
        half[title.split(":")[0]] = (np.median(widths) / 2, np.median(pw) / 2)
        lines += [f"## {title}", "", md(t.drop(columns=["_dice", "_key"])), "",
                  f"Width of the 95% Dice interval: median {np.median(widths):.2f} (range {min(widths):.2f}-{max(widths):.2f}); "
                  f"of the paired difference to the site U-Net: median {np.median(pw):.2f}.", ""]
        csv.append(t.drop(columns=["_dice"]).rename(columns={"_key": "key"}).assign(table=title.split(":")[0]))
    a, b = half["Track A"], half["Track B"]
    small = (f"With the 10 fold-2 patients the 95% interval of one model's Dice is about +-{a[0]:.2f} (median over the Track A "
             f"rows; Track B, 5 patients: about +-{b[0]:.2f}); the paired difference to the site U-Net on the same patients is "
             f"narrower, about +-{a[1]:.2f}.")
    lines = [x.replace("{SMALL}", small) for x in lines]
    lines += ["Track D (ranking without site labels) is proposed without a baseline.", ""]
    (P.RESULTS / "leaderboard_fold2.md").write_text("\n".join(lines), encoding="utf-8")
    pd.concat(csv).to_csv(P.RESULTS / "leaderboard_fold2.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
