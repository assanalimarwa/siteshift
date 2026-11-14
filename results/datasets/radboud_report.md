# PANDA-Radboud: dataset analysis

_paper_figs/src/ds_props.py + ds_report.py + ds_writeup.py, 2026-11-04 23:50, at 0.5 um/px_

PANDA-Radboud is the lightest set (L* 68) and the one with the most pen ink (6.4% of tissue fields have >= 3% ink, ours 0.7%; the ink is drawn next to tissue on many slides). One scanner at 0.486 um/px. 80% of biopsies contain tumour, and Gleason pattern 4 makes up most of the cancer pixels (58%; pattern 3 37%, 5 5%).

Its masks are gland-level: a 40-um closing would almost double the tumour area (+89%), there are six tumour regions per 256-um window and outline density is 12x ours (127 um per 1,000 um^2); regions are tiny (median 0.006 mm^2). The tumour share of tissue is accordingly the lowest (15%): the masks leave the stroma between glands out.

_Note: "All public combined" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + 1,800 Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source._

## Metadata

- **Data source / country**: Radboud University Medical Center, Nijmegen, Netherlands (2012-2017)
- **Scanner**: 3DHistech Pannoramic Flash II 250 (20x)
- **um/px**: 0.486 (top level of 0.24 removed)
- **Specimens**: needle biopsies (one biopsy per file)
- **Counts**: 1,028 patients (no patient ids released), 5,160 biopsies (5,060 with masks)
- **Labels**: gland-level masks: background, stroma, benign epithelium, Gleason 3 / 4 / 5 (semi-automatic, from student-read reports); tumour = Gleason 3-5
- **Official splits**: only the development set is public
- **License**: CC BY-NC-SA 4.0 (PANDA, Kaggle)
- **Format**: tiled TIFF, JPEG 512-px tiles, 3 levels; masks LZW
- **Known issues**: label noise (student labels vs expert consensus kappa 0.853); stroma label is a blocky halo over glass; ink sometimes labelled as cancer; 100 slides without mask
- **Sources**: Bulten et al. 2022 Nature Medicine; PANDA design document; our audit cv/radboud_labels.md

## Measurements (median over units [95% bootstrap CI]; rates: mean over units)

| measure | value |
|---|---|
| units (all) | 5060 |
| units sampled | 100 |
| fields (tissue / tumour) | 1,945 / 1,265 |
| units with tumour | 4,029/5,060 (80%) |
| tumour share of tissue per unit | 15.2 [14.8, 15.7]% |
| H-E angle, deg (Macenko) | 19.9 [19.0, 23.2] |
| mean H optical density | 0.462 [0.441, 0.484] |
| mean E optical density | 0.517 [0.475, 0.574] |
| L* (tissue) | 68.4 [67.6, 69.6] |
| a* (tissue) | 26.3 [25.3, 27.3] |
| b* (tissue) | -3.9 [-4.5, -3.5] |
| blur: Laplacian variance | 219 [206, 237] |
| blurry fields (variance < 30) | 1.4 [0.7, 2.2]% |
| fields with pen ink (>= 3% of the field) | 6.4 [3.4, 9.7]% |
| tissue fraction of tissue fields | 0.73 [0.70, 0.78] |
| tumour regions >= 2,500 um^2 (uncut) | 4,670 (0% of 4,670 cut by a border) |
| region area, mm^2 (median [CI]) | 0.0061 [0.0059, 0.0063] |
| region roughness P^2/(4 pi A) | 3.97 [3.88, 4.06] |
| gap fill at 40 um | 89.4 [79.6, 97.2]% |
| glass inside tumour | 0.2 [0.1, 0.2]% |
| lumen inside tumour | 2.1 [1.7, 2.5]% |
| tumour regions per 256-um window | 6.17 [5.92, 7.00] |
| outline, um per 1,000 um^2 | 127.4 [123.0, 136.6] |
| slides by Gleason score (all slides) | 0+0 967, 4+3 925, 3+3 852, 3+4 675, 4+4 660, 4+5 641, 5+4 221, 5+5 111, 3+5 67, 5+3 41 |
| cancer pixels by Gleason pattern (sampled slides) | pattern 3 37%, 4 58%, 5 5% |

## Domain distance to ours (UNI features)

UNI class tokens, 1,000 vs 1,000 tiles, 200 subsamples without replacement (mean [2.5-97.5 percentile]). Noise floor (ours vs ours, two random halves): Frechet distance 180.6 [177.1, 184.2], MMD^2 -0.0000 [-0.0002, 0.0002] (unbiased, so ~0: no ratio for MMD^2).
- PANDA-Radboud vs ours: FD 755.5 [737.5, 773.1] = 4.18 [4.06, 4.32] x the floor; MMD^2 0.0736 [0.0714, 0.0758], above the floor
- PANDA-Radboud vs ours biopsies: FD 794.7 [778.9, 812.1] = 4.40 [4.27, 4.53] x the floor; MMD^2 0.0837 [0.0816, 0.0863], above the floor

## Not reported

- Fold and air-bubble rates: the simple detectors (dark tissue with mean optical density > 1; round empty areas with a dark rim) failed the check by eye (data/check_fold.png): the fold detector flags densely cellular and darkly stained tissue (17-59% of fields in every dataset), the bubble detector found none.
