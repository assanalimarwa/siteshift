"""
RQ6 table: per architecture, what training on our own patients (site CV) gains over training on public data only (all-public run),
from the tile-mode results on the same 47 patients.

    python rq6_report.py            # writes cv47/rq6_table.md and export/rq6_table.md

Architectures: U-Net (baseline site CV from runs/cv47/imagenet; all-public U-Net from the PC, copied to runs/all_public_unet/eval_all_test_tile/),
nnU-Net-style and SegFormer (TransUNet when it has run). A row appears when both runs of the pair are complete.
gap = site CV minus all-public, per patient (positive = our own data helps); headline = per-patient Dice (mean of slide Dice).
Part 2 attributes the gap to detection (missed regions + false tumour far from the tumour) and delineation (unmarked interior +
boundary band), from protocol.py (protocol47.py); there the Dice is the patient-pooled one, whose error bins add up to 1 - Dice.
"""
import pandas as pd

import export_model as em
from cv_summary import bootstrap_ci

ARCH = [("U-Net", "baseline", "unet_allpublic"), ("nnU-Net-style 2D", "nnunet", "nnunet_allpublic"),
        ("SegFormer-B2", "segformer", "segformer_allpublic"), ("TransUNet", "transunet", "transunet_allpublic")]
BINS = [("missed", "detection"), ("false", "detection"), ("fill", "delineation"), ("boundary", "delineation")]


def finished(key):
    return key == "baseline" or em.finished(key)


def gap_cell(d):
    d = d.dropna()
    lo, hi = bootstrap_ci(d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] ({(d > 0).sum()} / {(d < 0).sum()})"


def dice_table(meta, per):
    rows, pending = [], []
    for name, site, pub in ARCH:
        if not (finished(site) and finished(pub)):
            pending.append(name)
            continue
        s, p = per(site), per(pub)
        both = s.index.intersection(p.index)
        d = (s.dice[both] - p.dice[both]).dropna()
        try:
            from scipy.stats import wilcoxon
            wp = f"{wilcoxon(d).pvalue:.2g}"
        except ValueError:
            wp = "n/a"
        typ = meta.type.reindex(both)
        wording = "tie (below 0.03)" if abs(d.mean()) < em.TIE else ("site CV ahead" if d.mean() > 0 else "all-public ahead")
        rows.append({"architecture": name, "patients": len(both),
                     "site CV Dice [95% CI]": em.cell(s.dice[both]), "all-public Dice [95% CI]": em.cell(p.dice[both]),
                     "gap site - public [95% CI] (site better / worse)": gap_cell(d), "Wilcoxon p": wp, "wording": wording,
                     "gap biopsies (n)": gap_cell(d[typ[d.index] == "biopsy"]) + f" n={(typ[d.index] == 'biopsy').sum()}",
                     "gap large sections (n)": gap_cell(d[typ[d.index] == "large"]) + f" n={(typ[d.index] == 'large').sum()}",
                     "patient-pooled gap": gap_cell((s.dice_pooled[both] - p.dice_pooled[both]))})
    return rows, pending


def attribution_table(meta):
    rows, pending = [], []
    for name, site, pub in ARCH:
        fs, fp = em.CV / f"protocol_{site}_by_patient.csv", em.CV / f"protocol_{pub}_by_patient.csv"
        if not (fs.exists() and fp.exists()):
            pending.append(name)
            continue
        s, p = pd.read_csv(fs, index_col=0), pd.read_csv(fp, index_col=0)
        both = s.index.intersection(p.index)
        typ = meta.type.reindex(both)
        for gname, idx in [("all patients", both), ("biopsies", both[typ == "biopsy"]), ("large sections", both[typ == "large"])]:
            d = {b: (p[f"err_{b}"][idx] - s[f"err_{b}"][idx]).dropna() for b, _ in BINS}
            gap = sum(d.values())
            det, dl = d["missed"] + d["false"], d["fill"] + d["boundary"]
            share = f"{det.mean() / gap.mean():.0%} / {dl.mean() / gap.mean():.0%}" if abs(gap.mean()) > 1e-9 else "n/a"
            ccell = lambda col: f"{s[col][idx].mean():.3f} / {p[col][idx].mean():.3f}"
            rows.append({"architecture": name, "group": gname, "n": len(idx),
                         "Dice gap (patient-pooled), error added by public training": gap_cell(gap),
                         "detection (missed + false tumour)": gap_cell(det), "- missed regions": gap_cell(d["missed"]),
                         "- false tumour far from tumour": gap_cell(d["false"]),
                         "delineation (fill + boundary)": gap_cell(dl), "- unmarked interior": gap_cell(d["fill"]),
                         "- boundary band": gap_cell(d["boundary"]),
                         "detection / delineation share of the gap": share,
                         "clean-tile AUC site / public": ccell("clean_auc"), "region recall site / public": ccell("recall"),
                         "dice_found site / public": ccell("dice_found")})
    return rows, pending


def main():
    meta = pd.read_csv(em.CV / "baseline_imagenet_by_patient.csv").set_index("pt")
    tis = em.tissue()
    cache = {}

    def per(key):
        if key not in cache:
            cache[key] = em.per_patient(key, tis)[0]
        return cache[key]

    rows, pend = dice_table(meta, per)
    lines = ["# RQ6: site data vs public data only, per architecture (tile mode, the same 47 patients)", "",
             "Written by rq6_report.py. Headline: per-patient Dice = mean of the patient's slide Dice. gap = site CV minus all-public, per patient "
             "(positive = our own patients help); in brackets: 95% bootstrap CI over patients, and the number of patients where site CV is "
             "better / worse. A gap below 0.03 between single runs is a tie.", "",
             "Training data differ by design: the site-CV fold models train on 35-36 of our patients and are tested on the held-out 9-10; the all-public "
             "model trains on SICAPv2 + Karolinska + Radboud (equal tile draws, none of ours) and is tested on all 47. Each pair uses the same architecture, "
             "recipe, epochs and seed. Machines: see the summary of each model (the U-Net baseline ran on the 4090, the others on the H100 server, "
             "folds on GPU 1 or on the shared GPU 0; fold-0 machine difference about +0.016).", ""]
    lines += ["## 1. Dice", md_or_none(rows)]
    if pend:
        lines += ["", "Pending (one run of the pair is not finished or its files are not on the server yet): " + ", ".join(pend) +
                  ". The U-Net all-public run was made on the PC: copy runs/all_public_unet/eval_all_test_tile/per_slide.csv and per_tile.csv here."]
    arows, apend = attribution_table(meta)
    lines += ["", "## 2. Where the gap comes from (detection vs delineation, protocol.py)", md_or_none(arows),
              "", "Dice here is the patient-pooled one, so the error bins add up to 1 - Dice: gap = sum of the four bins. The cells are public minus "
              "site CV (positive = public training adds error), mean [95% CI] (patients with more / less error under public training). Detection: "
              "missed regions (annotated tumour regions the model did not find) and false tumour farther than 50 um from any tumour. Delineation: "
              "unmarked tumour inside found regions and errors within 50 um of the outline."]
    if apend:
        lines += ["", "Pending (protocol analysis not run yet or predictions missing): " + ", ".join(apend)]
    text = "\n".join(lines) + "\n"
    (em.CV / "rq6_table.md").write_text(text)
    em.EXPORT.mkdir(parents=True, exist_ok=True)
    (em.EXPORT / "rq6_table.md").write_text(text)
    print("rq6_table.md written;", "pairs:", len(rows), "attribution rows:", len(arows))


def md_or_none(rows):
    return em.md(pd.DataFrame(rows)) if rows else "(no complete pair yet)"


if __name__ == "__main__":
    main()
