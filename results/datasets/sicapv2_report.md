# SICAPv2: dataset analysis

_paper_figs/src/ds_props.py + ds_report.py + ds_writeup.py, 2026-11-04 23:50, at 0.5 um/px_

SICAPv2 comes as overlapping 512-px patches without slide identifiers; 238 overlap regions were rebuilt from the patch overlaps and serve as units. Its patches are ~1.75x smaller in pixels than ours (pixel size not stated), so every field was enlarged to 0.5 um/px, which lowers its blur measure (Laplacian variance 81) by construction: that number is not comparable. Colour is lighter and bluer than ours (L* 66, b* -13).

Tumour is outlined close to the glands: a 40-um closing would add 10% to the tumour area (ours 1.4%), outlines are 3x denser than ours (33 um per 1,000 um^2) and regions are small (median 0.042 mm^2). 63% of regions contain tumour, with a median share of 32% of tissue. The Gleason information of the Kaggle copy is a set of undocumented grey levels.

_Note: "All public combined" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 Karolinska + 1,800 Radboud slides + 202 SICAPv2 regions = 5,816 units, sampled evenly per source._

## Metadata

- **Data source / country**: Hospital Clinico Universitario de Valencia / Universitat Politecnica de Valencia, Spain
- **Scanner**: Ventana iScan Coreo (40x)
- **um/px**: not stated; slides down-sampled to 10x (~1 um/px); 0.875 used here (1.75x, from nucleus size)
- **Specimens**: needle biopsies
- **Counts**: 95 patients, 155 biopsies (the abstract and Table 1 say 182 WSIs), 18,783 patches of 512 px with 50% overlap in the Kaggle copy used
- **Labels**: pixel-level Gleason annotation (non-cancer, G3, G4, cribriform G4, G5) by expert pathologists, consensus for uncertain cases; tumour = G3-G5
- **Official splits**: patient-level 4-fold CV + test fold (not recoverable from the Kaggle copy: no slide ids)
- **License**: CC BY 4.0 (Mendeley)
- **Format**: JPEG patches; masks greyscale JPEG in the Kaggle copy (grey levels undocumented)
- **Known issues**: 155 vs 182 slide count conflict; 50% patch overlap (leakage risk without patient ids); unannotated patches of cancer biopsies discarded; Kaggle Val/ overlaps Train/
- **Sources**: Silva-Rodriguez et al. 2020 (arXiv 2105.10490); data.mendeley.com/datasets/9xxm58dvs3/2

## Measurements (median over units [95% bootstrap CI]; rates: mean over units)

| measure | value |
|---|---|
| units (all) | 238 |
| units sampled | 100 |
| fields (tissue / tumour) | 1,696 / 1,068 |
| units with tumour | 150/238 (63%) |
| tumour share of tissue per unit | 32.1 [27.3, 35.7]% |
| H-E angle, deg (Macenko) | 20.2 [19.8, 20.7] |
| mean H optical density | 0.529 [0.471, 0.545] |
| mean E optical density | 0.421 [0.394, 0.447] |
| L* (tissue) | 66.3 [63.7, 68.7] |
| a* (tissue) | 22.7 [21.7, 23.8] |
| b* (tissue) | -13.2 [-13.8, -12.7] |
| blur: Laplacian variance | 81 [75, 90] |
| blurry fields (variance < 30) | 7.0 [5.3, 8.9]% |
| fields with pen ink (>= 3% of the field) | 0.1 [0.0, 0.2]% |
| tissue fraction of tissue fields | 0.70 [0.66, 0.75] |
| tumour regions >= 2,500 um^2 (uncut) | 370 (0% of 371 cut by a border) |
| region area, mm^2 (median [CI]) | 0.0422 [0.0355, 0.0546] |
| region roughness P^2/(4 pi A) | 3.01 [2.75, 3.31] |
| gap fill at 40 um | 10.3 [7.0, 13.0]% |
| glass inside tumour | 0.7 [0.4, 1.4]% |
| lumen inside tumour | 4.1 [2.7, 5.2]% |
| tumour regions per 256-um window | 1.32 [1.20, 1.40] |
| outline, um per 1,000 um^2 | 33.0 [28.3, 37.2] |
| cancer pixels by mask grey level (Kaggle copy; mapping to Gleason patterns not documented) | 50 17%, 65 9%, 75 1%, 85 5%, 100 38%, 150 9%, 165 8%, 175 1%, 185 1%, 200 10% |

## Domain distance to ours (UNI features)

UNI class tokens, 1,000 vs 1,000 tiles, 200 subsamples without replacement (mean [2.5-97.5 percentile]). Noise floor (ours vs ours, two random halves): Frechet distance 180.6 [177.1, 184.2], MMD^2 -0.0000 [-0.0002, 0.0002] (unbiased, so ~0: no ratio for MMD^2).
- SICAPv2 vs ours: FD 868.9 [851.1, 888.5] = 4.81 [4.67, 4.95] x the floor; MMD^2 0.0900 [0.0876, 0.0925], above the floor
- SICAPv2 vs ours biopsies: FD 796.3 [780.8, 813.0] = 4.41 [4.28, 4.53] x the floor; MMD^2 0.0820 [0.0802, 0.0840], above the floor

## Not reported

- Fold and air-bubble rates: the simple detectors (dark tissue with mean optical density > 1; round empty areas with a dark rim) failed the check by eye (data/check_fold.png): the fold detector flags densely cellular and darkly stained tissue (17-59% of fields in every dataset), the bubble detector found none.
