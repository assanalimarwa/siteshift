"""
Write-ups for the per-dataset analysis (run after ds_props.py; re-run after ds_distance.py to add the distances):

    python ds_writeup.py  ->  paper_figs/datasets/<ds>_report.md, comparison_table.tex, datasets_summary.md
                              (+ fig_stain.pdf, fig_label_geometry.pdf via ds_report)

Metadata come from the dataset papers and pages (sources listed per dataset; collected 2026-11-04). The two paragraphs
per dataset were written by hand from the measured numbers; re-check them if the numbers change (e.g. a changed scale).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import numpy as np
import pandas as pd

import ds_report
from ds_lib import DATA, OUT

META = {
    "ours": dict(site="withheld for review (double-blind)", scanner="Roche Ventana (.bif files; model to be filled in)",
                 um="0.5 (exported at 2x down from 0.25)", specimens="needle biopsies (25 patients) and large sections (22)",
                 counts="47 patients, 194 slides (89 biopsy, 105 large-section), 449,623 tiles of 512 px",
                 labels="whole-tumour region outlines (tumour vs everything else); no Gleason pattern",
                 splits="none official; our fixed patient-level 5-fold CV (cv47)", license="private (not distributed)",
                 fmt="PNG tiles (lossless) exported from the .bif slides with QuPath",
                 issues="some slides out of focus in parts (blur below); blue tissue-marking dye at some edges",
                 sources="our export (tiles47.csv, cv47/dataset_by_patient.csv)"),
    "sicapv2": dict(site="Hospital Clinico Universitario de Valencia / Universitat Politecnica de Valencia, Spain",
                    scanner="Ventana iScan Coreo (40x)", um="not stated; slides down-sampled to 10x (~1 um/px); 0.875 used here (1.75x, from nucleus size)",
                    specimens="needle biopsies", counts="95 patients, 155 biopsies (the abstract and Table 1 say 182 WSIs), 18,783 patches of 512 px with 50% overlap in the Kaggle copy used",
                    labels="pixel-level Gleason annotation (non-cancer, G3, G4, cribriform G4, G5) by expert pathologists, consensus for uncertain cases; tumour = G3-G5",
                    splits="patient-level 4-fold CV + test fold (not recoverable from the Kaggle copy: no slide ids)",
                    license="CC BY 4.0 (Mendeley)", fmt="JPEG patches; masks greyscale JPEG in the Kaggle copy (grey levels undocumented)",
                    issues="155 vs 182 slide count conflict; 50% patch overlap (leakage risk without patient ids); unannotated patches of cancer biopsies discarded; Kaggle Val/ overlaps Train/",
                    sources="Silva-Rodriguez et al. 2020 (arXiv 2105.10490); data.mendeley.com/datasets/9xxm58dvs3/2"),
    "karolinska": dict(site="Karolinska Institutet, Stockholm, Sweden (STHLM3 study, 2013-2015)",
                       scanner="Hamamatsu C9600-12 (0.452 um/px, 3,263 slides) and Aperio ScanScope AT2 (0.503 um/px, 2,193 slides)",
                       um="0.452 / 0.503 (per slide, TIFF tags)", specimens="needle biopsies",
                       counts="1,085 patients (no patient ids released), 5,456 slides (5,452 with tumour mask content used)",
                       labels="coarse region masks: background, benign, cancer (one experienced pathologist plus pen-mark annotations); slide-level ISUP / Gleason",
                       splits="only the development set is public (tuning and validation sets not released)",
                       license="CC BY-NC-SA 4.0 (PANDA, Kaggle)", fmt="tiled TIFF, JPEG 512-px tiles, 3 levels; masks LZW",
                       issues="slide-level split only (no patient ids: possible patient leakage); noisy labels; masks drawn at low resolution (staircase edges); two scanners and two pixel sizes",
                       sources="Bulten et al. 2022 Nature Medicine; PANDA challenge design document (Zenodo 3715938); TIFF tags"),
    "radboud": dict(site="Radboud University Medical Center, Nijmegen, Netherlands (2012-2017)", scanner="3DHistech Pannoramic Flash II 250 (20x)",
                    um="0.486 (top level of 0.24 removed)", specimens="needle biopsies (one biopsy per file)",
                    counts="1,028 patients (no patient ids released), 5,160 biopsies (5,060 with masks)",
                    labels="gland-level masks: background, stroma, benign epithelium, Gleason 3 / 4 / 5 (semi-automatic, from student-read reports); tumour = Gleason 3-5",
                    splits="only the development set is public", license="CC BY-NC-SA 4.0 (PANDA, Kaggle)",
                    fmt="tiled TIFF, JPEG 512-px tiles, 3 levels; masks LZW",
                    issues="label noise (student labels vs expert consensus kappa 0.853); stroma label is a blocky halo over glass; ink sometimes labelled as cancer; 100 slides without mask",
                    sources="Bulten et al. 2022 Nature Medicine; PANDA design document; our audit cv/radboud_labels.md"),
}

WRITE = {
    "ours": ("Our dataset is the only one with large sections next to biopsies and the only one where every slide holds "
             "tumour (194/194; median tumour share 37% of tissue per slide). Its stain is the most purple: the Macenko "
             "haematoxylin and eosin directions are only 14 degrees apart (19-25 in the public sets) and tissue is the least red "
             "(a* 18 vs 23-37), so H and E separate worst. It also has the most out-of-focus fields (11% of tissue fields with "
             "Laplacian variance < 30, vs <= 1.4% in PANDA), concentrated in a few slides.",
             "The labels are region-level: one region per 256-um window, almost no gaps that a 40-um closing would fill (1.4%), "
             "outline density 10 um per 1,000 um^2 and the largest regions after Karolinska (median 0.22 mm^2). Tumour "
             "outlines rarely cover glass (0.2%). By label geometry our convention is closest to PANDA-Karolinska and far from "
             "the gland-level SICAPv2 and PANDA-Radboud masks."),
    "sicapv2": ("SICAPv2 comes as overlapping 512-px patches without slide identifiers; 238 overlap regions were rebuilt from "
                "the patch overlaps and serve as units. Its patches are ~1.75x smaller in pixels than ours (pixel size not "
                "stated), so every field was enlarged to 0.5 um/px, which lowers its blur measure (Laplacian variance 81) by "
                "construction: that number is not comparable. Colour is lighter and bluer than ours (L* 66, b* -13).",
                "Tumour is outlined close to the glands: a 40-um closing would add 10% to the tumour area (ours 1.4%), "
                "outlines are 3x denser than ours (33 um per 1,000 um^2) and regions are small (median 0.042 mm^2). "
                "63% of regions contain tumour, with a median share of 32% of tissue. The Gleason information of the Kaggle "
                "copy is a set of undocumented grey levels."),
    "karolinska": ("PANDA-Karolinska is the most saturated and sharpest set: tissue is far redder than in any other dataset "
                   "(a* 37), eosin optical density is the highest (0.65) and the Laplacian variance is the highest (922, no "
                   "blurry fields). It mixes two scanners: 60% of slides at 0.452 um/px (Hamamatsu) and 40% at 0.503 (Aperio). "
                   "65% of slides contain tumour; Gleason scores range from 3+3 (1,814 slides) to 5+5 (16).",
                   "Its cancer masks are coarse regions: the largest regions of all datasets (median 1.1 mm^2), one region per "
                   "window and few gaps (2.4%), like ours, but drawn at low resolution with staircase edges and spilling onto "
                   "glass (5.3% of the tumour label lies on glass, ours 0.2%). Per slide, tumour covers a median 39% of the "
                   "labelled tissue."),
    "radboud": ("PANDA-Radboud is the lightest set (L* 68) and the one with the most pen ink (6.4% of tissue fields have >= 3% "
                "ink, ours 0.7%; the ink is drawn next to tissue on many slides). One scanner at 0.486 um/px. 80% of biopsies "
                "contain tumour, and Gleason pattern 4 makes up most of the cancer pixels (58%; pattern 3 37%, 5 5%).",
                "Its masks are gland-level: a 40-um closing would almost double the tumour area (+89%), there are six tumour "
                "regions per 256-um window and outline density is 12x ours (127 um per 1,000 um^2); regions are tiny (median "
                "0.006 mm^2). The tumour share of tissue is accordingly the lowest (15%): the masks leave the stroma between "
                "glands out."),
}

SUMMARY = [
    "**Annotation convention splits the datasets in two.** Ours and PANDA-Karolinska outline tumour regions (a 40-um closing "
    "adds 1.4% / 2.4%, ~1 region per 256-um window, median region 0.22 / 1.1 mm^2); SICAPv2 and PANDA-Radboud outline "
    "glands (closing adds 10% / 89%, outlines 3-12x denser, median regions 0.042 / 0.006 mm^2).",
    "**Region-level is not the same as precise.** Karolinska's masks are drawn at low resolution and put 5.3% of the tumour "
    "label on glass (ours 0.2%, SICAPv2 0.7%, Radboud 0.2%).",
    "**Tumour content differs with the convention and the case mix.** Every one of our slides holds tumour (median 37% of "
    "tissue); 63-80% of the public units do, and the gland-level Radboud masks give the lowest share (15%).",
    "**Our stain is the most purple and least separable.** Haematoxylin and eosin directions 14 deg apart (public 20-25), "
    "tissue least red (a* 18 vs 23-37); Karolinska is the reddest and most saturated. In UNI feature space every dataset forms "
    "its own cluster (5-fold accuracy 0.999 for telling the datasets apart). Frechet distance to ours is 4.2x our own "
    "noise floor for Radboud, 4.8x for SICAPv2 and 5.2x for Karolinska, but our biopsies and our large sections are "
    "already 4.1x apart: the specimen-type gap inside our site is as large as the gap to the nearest public set.",
    "**Image quality differs in kind.** Ours has the most out-of-focus fields (11%), Radboud the most pen ink (6% of fields), "
    "Karolinska the sharpest images and two scanners at two pixel sizes; SICAPv2 is distributed without a stated pixel size, "
    "so its scale is an estimate (1.75x ours, from nucleus size).",
]

ALL_PUBLIC = ("\"All public combined\" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + 1,800 "
              "Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source."
              )


def tex_table(rows):
    def g(d, k, pct=False):
        v = rows[d].get(k, "n/a")
        return v.split(" [")[0] if isinstance(v, str) else "n/a"
    lines = ["% generated by paper_figs/src/ds_writeup.py", "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{llllrrrrrr}", "\\toprule",
             "Dataset & Site & Scanner & \\textmu m/px & Tumour share & Region area & Gap fill & Outline & H--E angle & Ink \\\\",
             " & & & & (median) & (mm$^2$) & 40\\,\\textmu m & (\\textmu m/10$^3$\\,\\textmu m$^2$) & (deg) & fields \\\\", "\\midrule"]
    short = {"ours": ("Ours", "ours", "Ventana", "0.50"), "sicapv2": ("SICAPv2", "Valencia, ES", "Ventana iScan", "$\\approx$0.9$^a$"),
             "karolinska": ("PANDA-Karolinska", "Stockholm, SE", "Hamamatsu / Aperio", "0.45 / 0.50"),
             "radboud": ("PANDA-Radboud", "Nijmegen, NL", "3DHistech", "0.49")}
    for d in ds_report.ORDER:
        if d not in rows:
            continue
        n, site, sc, um = short[d]
        vals = [g(d, "tumour share of tissue per unit").replace("%", "\\%"), g(d, "region area, mm^2 (median [CI])"),
                g(d, "gap fill at 40 um").replace("%", "\\%"), g(d, "outline, um per 1,000 um^2"), g(d, "H-E angle, deg (Macenko)"),
                g(d, "fields with pen ink (>= 3% of the field)").replace("%", "\\%")]
        lines.append(" & ".join([n, site, sc, um] + vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Per-dataset analysis at 0.5\\,\\textmu m/px (medians over slides; SICAPv2: overlap regions). Gap fill: tumour area a closing with a 40-\\textmu m disk would add (gland-level labels "
              "score high). Outline: tumour outline length per tumour area. H--E angle: angle between the Macenko "
              "haematoxylin and eosin vectors. Ink: tissue fields with $\\geq$3\\% pen-ink pixels. $^a$Not stated; "
              "1.75$\\times$ our scale from nucleus size.}",
              "\\label{tab:dataset_analysis}", "\\end{table*}"]
    (OUT / "comparison_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def distance_lines(d):
    p = DATA / "distance.csv"
    if not p.exists():
        return ["Domain distance: pending (ds_distance.py)."]
    dist = pd.read_csv(p).set_index("comparison")
    fl = dist.loc["ours (half) vs ours (half)"]
    out = [f"UNI class tokens, 1,000 vs 1,000 tiles, 200 subsamples without replacement (mean [2.5-97.5 percentile]). "
           f"Noise floor (ours vs ours, two random halves): Frechet distance {fl.FD:.1f} [{fl.FD_lo:.1f}, {fl.FD_hi:.1f}], "
           f"MMD^2 {fl.MMD2:.4f} [{fl.MMD2_lo:.4f}, {fl.MMD2_hi:.4f}] (unbiased, so ~0: no ratio for MMD^2)."]
    if d == "ours":
        b = dist.loc["ours biopsies vs ours large sections"]
        out.append(f"- ours biopsies vs ours large sections: FD {b.FD:.1f} [{b.FD_lo:.1f}, {b.FD_hi:.1f}] = "
                   f"{b.FD / fl.FD:.2f} x the floor; MMD^2 {b.MMD2:.4f} [{b.MMD2_lo:.4f}, {b.MMD2_hi:.4f}]")
        return out
    name = ds_report.LABEL[d]
    for c in [f"{name} vs ours", f"{name} vs ours biopsies"]:
        r = dist.loc[c]
        ratio = (f"{r.FD_ratio:.2f} [{r.FD_ratio_lo:.2f}, {r.FD_ratio_hi:.2f}] x the floor" if pd.notna(r.get("FD_ratio"))
                 else f"{r.FD / fl.FD:.2f} x the floor")
        out.append(f"- {c}: FD {r.FD:.1f} [{r.FD_lo:.1f}, {r.FD_hi:.1f}] = {ratio}; MMD^2 {r.MMD2:.4f} "
                   f"[{r.MMD2_lo:.4f}, {r.MMD2_hi:.4f}], {'above' if r.MMD2_lo > fl.MMD2_hi else 'not above'} the floor")
    return out

def main():
    rows, tables = ds_report.main()
    for d, r in rows.items():
        m = META[d]
        md = [f"# {ds_report.LABEL[d]}: dataset analysis", "",
              f"_paper_figs/src/ds_props.py + ds_report.py + ds_writeup.py, {pd.Timestamp.now():%Y-%m-%d %H:%M}, at 0.5 um/px_", "",
              WRITE[d][0], "", WRITE[d][1], "", f"_Note: {ALL_PUBLIC}_", "", "## Metadata", ""]
        md += [f"- **{k}**: {v}" for k, v in [("Data source / country", m["site"]), ("Scanner", m["scanner"]), ("um/px", m["um"]),
                                                ("Specimens", m["specimens"]), ("Counts", m["counts"]), ("Labels", m["labels"]),
                                                ("Official splits", m["splits"]), ("License", m["license"]), ("Format", m["fmt"]),
                                                ("Known issues", m["issues"]), ("Sources", m["sources"])]]
        md += ["", "## Measurements (median over units [95% bootstrap CI]; rates: mean over units)", "",
               "| measure | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r.items()]
        md += ["", "## Domain distance to ours (UNI features)", ""] + distance_lines(d)
        md += ["", "## Not reported", "",
               "- Fold and air-bubble rates: the simple detectors (dark tissue with mean optical density > 1; round empty "
               "areas with a dark rim) failed the check by eye (data/check_fold.png): the fold detector flags densely "
               "cellular and darkly stained tissue (17-59% of fields in every dataset), the bubble detector found none.", ""]
        # stroma share inside tumour outlines (item 4, ds_stroma.py): dropped for good by the user 2026-11-10 (no check labels)
        (OUT / f"{d}_report.md").write_text("\n".join(md), encoding="utf-8")
    tex_table(rows)
    (OUT / "datasets_summary.md").write_text(
        "# What differs between the datasets\n\n" + "\n".join(f"- {b}" for b in SUMMARY) +
        "\n\nNumbers: <dataset>_report.md, comparison_table.tex; figures fig_stain.pdf, fig_label_geometry.pdf, "
        "fig_<dataset>_samples.pdf, fig_domain_distance.pdf.\n", encoding="utf-8")
    print("-> reports, comparison_table.tex, datasets_summary.md")


if __name__ == "__main__":
    main()
