# PANDA-Karolinska: dataset analysis

_paper_figs/src/ds_props.py + ds_report.py + ds_writeup.py, 2026-11-04 23:50, at 0.5 um/px_

PANDA-Karolinska is the most saturated and sharpest set: tissue is far redder than in any other dataset (a* 37), eosin optical density is the highest (0.65) and the Laplacian variance is the highest (922, no blurry fields). It mixes two scanners: 60% of slides at 0.452 um/px (Hamamatsu) and 40% at 0.503 (Aperio). 65% of slides contain tumour; Gleason scores range from 3+3 (1,814 slides) to 5+5 (16).

Its cancer masks are coarse regions: the largest regions of all datasets (median 1.1 mm^2), one region per window and few gaps (2.4%), like ours, but drawn at low resolution with staircase edges and spilling onto glass (5.3% of the tumour label lies on glass, ours 0.2%). Per slide, tumour covers a median 39% of the labelled tissue.

_Note: "All public combined" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + 1,800 Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source._

## Metadata

- **Data source / country**: Karolinska Institutet, Stockholm, Sweden (STHLM3 study, 2013-2015)
- **Scanner**: Hamamatsu C9600-12 (0.452 um/px, 3,263 slides) and Aperio ScanScope AT2 (0.503 um/px, 2,193 slides)
- **um/px**: 0.452 / 0.503 (per slide, TIFF tags)
- **Specimens**: needle biopsies
- **Counts**: 1,085 patients (no patient ids released), 5,456 slides (5,452 with tumour mask content used)
- **Labels**: coarse region masks: background, benign, cancer (one experienced pathologist plus pen-mark annotations); slide-level ISUP / Gleason
- **Official splits**: only the development set is public (tuning and validation sets not released)
- **License**: CC BY-NC-SA 4.0 (PANDA, Kaggle)
- **Format**: tiled TIFF, JPEG 512-px tiles, 3 levels; masks LZW
- **Known issues**: slide-level split only (no patient ids: possible patient leakage); noisy labels; masks drawn at low resolution (staircase edges); two scanners and two pixel sizes
- **Sources**: Bulten et al. 2022 Nature Medicine; PANDA challenge design document (Zenodo 3715938); TIFF tags

## Measurements (median over units [95% bootstrap CI]; rates: mean over units)

| measure | value |
|---|---|
| units (all) | 5452 |
| units sampled | 100 |
| fields (tissue / tumour) | 1,975 / 1,350 |
| units with tumour | 3,528/5,452 (65%) |
| tumour share of tissue per unit | 38.7 [37.5, 40.4]% |
| H-E angle, deg (Macenko) | 24.6 [24.1, 25.2] |
| mean H optical density | 0.525 [0.502, 0.545] |
| mean E optical density | 0.647 [0.600, 0.695] |
| L* (tissue) | 57.8 [56.3, 59.0] |
| a* (tissue) | 36.5 [34.6, 38.2] |
| b* (tissue) | -17.3 [-17.9, -16.6] |
| blur: Laplacian variance | 922 [795, 1027] |
| blurry fields (variance < 30) | 0.0 [0.0, 0.0]% |
| fields with pen ink (>= 3% of the field) | 0.0 [0.0, 0.0]% |
| tissue fraction of tissue fields | 0.81 [0.79, 0.84] |
| tumour regions >= 2,500 um^2 (uncut) | 98 (0% of 98 cut by a border) |
| region area, mm^2 (median [CI]) | 1.1195 [0.9836, 1.5358] |
| region roughness P^2/(4 pi A) | 2.38 [2.20, 2.89] |
| gap fill at 40 um | 2.4 [2.1, 2.7]% |
| glass inside tumour | 5.3 [3.7, 6.9]% |
| lumen inside tumour | 3.2 [2.6, 3.8]% |
| tumour regions per 256-um window | 1.00 [1.00, 1.00] |
| outline, um per 1,000 um^2 | 10.1 [9.7, 10.8] |
| slides by Gleason score (all slides) | 0+0 1,925, 3+3 1,814, 3+4 667, 4+4 466, 4+3 318, 4+5 208, 5+4 27, 5+5 16, 3+5 13, 5+3 2 |

## Domain distance to ours (UNI features)

UNI class tokens, 1,000 vs 1,000 tiles, 200 subsamples without replacement (mean [2.5-97.5 percentile]). Noise floor (ours vs ours, two random halves): Frechet distance 180.6 [177.1, 184.2], MMD^2 -0.0000 [-0.0002, 0.0002] (unbiased, so ~0: no ratio for MMD^2).
- PANDA-Karolinska vs ours: FD 939.9 [917.3, 959.1] = 5.20 [5.06, 5.36] x the floor; MMD^2 0.1016 [0.0983, 0.1046], above the floor
- PANDA-Karolinska vs ours biopsies: FD 940.5 [921.6, 962.3] = 5.21 [5.08, 5.34] x the floor; MMD^2 0.1062 [0.1036, 0.1090], above the floor

## Not reported

- Fold and air-bubble rates: the simple detectors (dark tissue with mean optical density > 1; round empty areas with a dark rim) failed the check by eye (data/check_fold.png): the fold detector flags densely cellular and darkly stained tissue (17-59% of fields in every dataset), the bubble detector found none.
