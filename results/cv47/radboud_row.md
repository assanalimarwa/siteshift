# Radboud U-Net, main-table row (47 patients)

_From cv47/public_summary.md (report47.py; headline per-patient Dice = mean of the patient's slide Dice; 95% CI = 10,000 patient resamples, seed 0; sensitivity, precision, false-tumour rate and clean-tile AUC = mean +- SD [CI] over patients). Copied 2026-11-12._

| model | mode | patients | Dice per patient (mean +- SD [95% CI]) | median | patient-pooled Dice | pooled Dice (all pixels) | sensitivity | precision | false-tumour rate | clean-tile AUC | vs U-Net baseline (Dice diff [CI]) | better/worse | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Radboud U-Net | tile | 47 | 0.438 +- 0.174 [0.389, 0.486] | 0.456 | 0.470 +- 0.167 [0.422, 0.516] | 0.466 | 0.407 +- 0.164 [0.361, 0.453] | 0.639 +- 0.241 [0.569, 0.706] | 0.106 +- 0.096 [0.081, 0.135] | 0.884 +- 0.077 [0.862, 0.905] | -0.282 [-0.323, -0.238] | 1/46 | 7.62e-12 |
| Radboud U-Net | full | 47 | 0.443 +- 0.172 [0.395, 0.491] | 0.458 | 0.474 +- 0.164 [0.428, 0.520] | 0.469 | 0.398 +- 0.165 [0.352, 0.444] | 0.681 +- 0.230 [0.614, 0.744] | 0.091 +- 0.092 [0.067, 0.119] | 0.894 +- 0.080 [0.871, 0.916] | -0.302 [-0.347, -0.256] | 1/46 | 2.4e-12 |

Per specimen type:

| model | mode | biopsy n | biopsy Dice mean [CI] | biopsy median | biopsy false-tumour | large n | large Dice mean [CI] | large median | large false-tumour |
|---|---|---|---|---|---|---|---|---|---|
| Radboud U-Net | tile | 25 | 0.520 [0.462, 0.578] | 0.488 | 0.147 | 22 | 0.345 [0.282, 0.409] | 0.357 | 0.060 |
| Radboud U-Net | full | 25 | 0.524 [0.465, 0.582] | 0.489 | 0.130 | 22 | 0.351 [0.291, 0.413] | 0.361 | 0.046 |
