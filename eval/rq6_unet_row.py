"""
The U-Net rows of the RQ6 table (results/cv47/rq6_table.md, results/rq6_table.md). rq6_report.py builds the table from the
runs on the server, where the all-public U-Net was not; its pair (site CV U-Net, all-public U-Net) was trained and evaluated on
the 4090 PC. This script adds the U-Net rows from the released per-patient results, with rq6_report.py's own cell formats:

    python rq6_unet_row.py

  Dice          results/cv47/public_by_patient.csv (eval/report47.py public): "U-Net site CV (baseline)" and
                "all public (A) U-Net", tile mode; per-patient Dice = mean of the patient's slide Dice, patient-pooled Dice
  attribution   results/cv47/protocol_by_patient.csv (eval/report_protocol47.py): unet_site and all_public_unet, tile mode
The site U-Net's per-patient Dice here equals the server's copy (results/cv47/<model>_by_patient.csv, baseline_dice).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import pandas as pd
from scipy.stats import wilcoxon

import export_model as em
from rq6_report import BINS, gap_cell

CV = P.RESULTS / "cv47"
NAME = "U-Net (ResNet-34)"
OLD_NOTE = "The U-Net all-public run was trained and tested on a separate machine and is not part of this table (see release_report.md)."
NEW_NOTE = ("U-Net row: both runs of the pair were trained and evaluated on the 4090 PC; added by rq6_unet_row.py from "
            "results/cv47/public_by_patient.csv and protocol_by_patient.csv with the same cell formats.")


def md_row(d):
    return "| " + " | ".join(str(v) for v in d.values()) + " |"


def rows():
    meta = pd.read_csv(CV / "baseline_imagenet_by_patient.csv").set_index("pt")
    pb = pd.read_csv(CV / "public_by_patient.csv")
    tile = pb[pb["mode"] == "tile"]
    s = tile[tile.model == "U-Net site CV (baseline)"].set_index("p")
    p = tile[tile.model == "all public (A) U-Net"].set_index("p")
    both = s.index.intersection(p.index)
    d = (s.dice[both] - p.dice[both]).dropna()
    typ = meta.type.reindex(both)
    wording = "tie (below 0.03)" if abs(d.mean()) < em.TIE else ("site CV ahead" if d.mean() > 0 else "all-public ahead")
    dice = {"architecture": NAME, "patients": len(both),
            "site CV Dice [95% CI]": em.cell(s.dice[both]), "all-public Dice [95% CI]": em.cell(p.dice[both]),
            "gap site - public [95% CI] (site better / worse)": gap_cell(d), "Wilcoxon p": f"{wilcoxon(d).pvalue:.2g}", "wording": wording,
            "gap biopsies (n)": gap_cell(d[typ[d.index] == "biopsy"]) + f" n={(typ[d.index] == 'biopsy').sum()}",
            "gap large sections (n)": gap_cell(d[typ[d.index] == "large"]) + f" n={(typ[d.index] == 'large').sum()}",
            "patient-pooled gap": gap_cell((s.dice_pooled[both] - p.dice_pooled[both]))}
    pr = pd.read_csv(CV / "protocol_by_patient.csv")
    pr = pr[pr["mode"] == "tile"]
    ps, pp = pr[pr.model == "unet_site"].set_index("p"), pr[pr.model == "all_public_unet"].set_index("p")
    both = ps.index.intersection(pp.index)
    typ = meta.type.reindex(both)
    attr = []
    for gname, idx in [("all patients", both), ("biopsies", both[typ == "biopsy"]), ("large sections", both[typ == "large"])]:
        dd = {b: (pp[f"err_{b}"][idx] - ps[f"err_{b}"][idx]).dropna() for b, _ in BINS}
        gap = sum(dd.values())
        det, dl = dd["missed"] + dd["false"], dd["fill"] + dd["boundary"]
        share = f"{det.mean() / gap.mean():.0%} / {dl.mean() / gap.mean():.0%}" if abs(gap.mean()) > 1e-9 else "n/a"
        ccell = lambda col: f"{ps[col][idx].mean():.3f} / {pp[col][idx].mean():.3f}"
        attr.append({"architecture": NAME, "group": gname, "n": len(idx),
                     "Dice gap (patient-pooled), error added by public training": gap_cell(gap),
                     "detection (missed + false tumour)": gap_cell(det), "- missed regions": gap_cell(dd["missed"]),
                     "- false tumour far from tumour": gap_cell(dd["false"]),
                     "delineation (fill + boundary)": gap_cell(dl), "- unmarked interior": gap_cell(dd["fill"]),
                     "- boundary band": gap_cell(dd["boundary"]),
                     "detection / delineation share of the gap": share,
                     "clean-tile AUC site / public": ccell("clean_auc"), "region recall site / public": ccell("recall"),
                     "dice_found site / public": ccell("dice_found")})
    return dice, attr


def insert(text, dice, attr):
    """The U-Net rows go first in both tables (after each table's separator line)."""
    if f"| {NAME} |" in text:
        raise SystemExit("the U-Net rows are already in the table")
    lines = text.split("\n")
    seps = [i for i, x in enumerate(lines) if x.startswith("|---")]
    assert len(seps) == 2, seps
    for header, new in [(lines[seps[0] - 1], [dice]), (lines[seps[1] - 1], attr)]:
        assert [c.strip() for c in header.strip("|").split("|")] == list(new[0].keys()), header
    out = lines[:seps[0] + 1] + [md_row(dice)] + lines[seps[0] + 1:seps[1] + 1] + [md_row(r) for r in attr] + lines[seps[1] + 1:]
    text = "\n".join(out)
    return text.replace(OLD_NOTE, NEW_NOTE) if OLD_NOTE in text else text + "\n" + NEW_NOTE + "\n"


def main():
    dice, attr = rows()
    for path in [CV / "rq6_table.md", P.RESULTS / "rq6_table.md"]:
        path.write_text(insert(path.read_text(encoding="utf-8"), dice, attr), encoding="utf-8")
        print(f"-> {path}")
    print(md_row(dice))
    for r in attr:
        print(md_row(r))


if __name__ == "__main__":
    main()
