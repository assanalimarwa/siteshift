# Ours: dataset analysis

_paper_figs/src/ds_props.py + ds_report.py + ds_writeup.py, 2026-11-04 23:50, at 0.5 um/px_

Our dataset is the only one with large sections next to biopsies and the only one where every slide holds tumour (194/194; median tumour share 37% of tissue per slide). Its stain is the most purple: the Macenko haematoxylin and eosin directions are only 14 degrees apart (19-25 in the public sets) and tissue is the least red (a* 18 vs 23-37), so H and E separate worst. It also has the most out-of-focus fields (11% of tissue fields with Laplacian variance < 30, vs <= 1.4% in PANDA), concentrated in a few slides.

The labels are region-level: one region per 256-um window, almost no gaps that a 40-um closing would fill (1.4%), outline density 10 um per 1,000 um^2 and the largest regions after Karolinska (median 0.22 mm^2). Tumour outlines rarely cover glass (0.2%). By label geometry our convention is closest to PANDA-Karolinska and far from the gland-level SICAPv2 and PANDA-Radboud masks.

_Note: "All public combined" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + 1,800 Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source._

## Metadata

- **Data source / country**: withheld for review (double-blind)
- **Scanner**: Roche Ventana (.bif files; model to be filled in)
- **um/px**: 0.5 (exported at 2x down from 0.25)
- **Specimens**: needle biopsies (25 patients) and large sections (22)
- **Counts**: 47 patients, 194 slides (89 biopsy, 105 large-section), 449,623 tiles of 512 px
- **Labels**: whole-tumour region outlines (tumour vs everything else); no Gleason pattern
- **Official splits**: none official; our fixed patient-level 5-fold CV (cv47)
- **License**: private (not distributed)
- **Format**: PNG tiles (lossless) exported from the .bif slides with QuPath
- **Known issues**: some slides out of focus in parts (blur below); blue tissue-marking dye at some edges
- **Sources**: our export (tiles47.csv, cv47/dataset_by_patient.csv)

## Measurements (median over units [95% bootstrap CI]; rates: mean over units)

| measure | value |
|---|---|
| units (all) | 194 |
| units sampled | 100 |
| fields (tissue / tumour) | 2,000 / 1,952 |
| units with tumour | 194/194 (100%) |
| tumour share of tissue per unit | 36.9 [30.4, 40.9]% |
| tumour share of tissue per patient | 36.4 [25.0, 37.7]% (n = 47) |
| H-E angle, deg (Macenko) | 14.1 [13.5, 14.9] |
| mean H optical density | 0.490 [0.445, 0.530] |
| mean E optical density | 0.517 [0.469, 0.576] |
| L* (tissue) | 60.4 [57.6, 62.0] |
| a* (tissue) | 17.9 [16.9, 19.0] |
| b* (tissue) | -8.2 [-9.0, -7.4] |
| blur: Laplacian variance | 539 [425, 616] |
| blurry fields (variance < 30) | 11.3 [7.1, 16.0]% |
| fields with pen ink (>= 3% of the field) | 0.7 [0.3, 1.1]% |
| tissue fraction of tissue fields | 0.93 [0.91, 0.95] |
| tumour regions >= 2,500 um^2 (uncut) | 758 (0% of 759 cut by a border) |
| region area, mm^2 (median [CI]) | 0.2220 [0.1746, 0.2787] |
| region roughness P^2/(4 pi A) | 2.54 [2.40, 2.82] |
| gap fill at 40 um | 1.4 [1.1, 2.6]% |
| glass inside tumour | 0.2 [0.1, 0.3]% |
| lumen inside tumour | 2.5 [2.1, 3.2]% |
| tumour regions per 256-um window | 1.05 [1.05, 1.05] |
| outline, um per 1,000 um^2 | 10.2 [7.3, 12.0] |
| Gleason | not annotated (tumour vs non-tumour only) |

## Domain distance to ours (UNI features)

UNI class tokens, 1,000 vs 1,000 tiles, 200 subsamples without replacement (mean [2.5-97.5 percentile]). Noise floor (ours vs ours, two random halves): Frechet distance 180.6 [177.1, 184.2], MMD^2 -0.0000 [-0.0002, 0.0002] (unbiased, so ~0: no ratio for MMD^2).
- ours biopsies vs ours large sections: FD 745.2 [730.4, 760.8] = 4.13 x the floor; MMD^2 0.0632 [0.0609, 0.0653]

## Not reported

- Fold and air-bubble rates: the simple detectors (dark tissue with mean optical density > 1; round empty areas with a dark rim) failed the check by eye (data/check_fold.png): the fold detector flags densely cellular and darkly stained tissue (17-59% of fields in every dataset), the bubble detector found none.
