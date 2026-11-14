# UNI site CV (frozen UNI + shared head), 47 patients

_2026-11-08 11:51, machine: 4090 (report47.py uni_cv)_

run_cv.py fm_uni on cv47 (15 epochs max, patience 4, epochs of 60,000 tiles, 20,000 val tiles), tile mode as for every frozen encoder. Anchor for the 4090 vs H100 comparison.

Per-patient Dice = mean of the patient's slide Dice; CI = 95% bootstrap of the mean over patients.

## Results

| model | mode | patients | Dice per patient (mean +- SD [95% CI]) | median | patient-pooled Dice | pooled Dice (all pixels) | sensitivity | precision | false-tumour rate | clean-tile AUC | vs U-Net baseline (Dice diff [CI]) | better/worse | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UNI site CV (4090) | tile | 47 | 0.791 +- 0.143 [0.749, 0.829] | 0.843 | 0.824 +- 0.119 [0.789, 0.856] | 0.882 | 0.829 +- 0.151 [0.784, 0.870] | 0.838 +- 0.121 [0.802, 0.871] | 0.078 +- 0.087 [0.055, 0.104] | 0.964 +- 0.042 [0.951, 0.975] | +0.070 [+0.052, +0.093] | 44/3 | 9.09e-12 |

## By specimen type (per-patient Dice)

| model | mode | biopsy n | biopsy Dice mean [CI] | biopsy median | biopsy false-tumour | large n | large Dice mean [CI] | large median | large false-tumour |
|---|---|---|---|---|---|---|---|---|---|
| UNI site CV (4090) | tile | 25 | 0.803 [0.750, 0.848] | 0.843 | 0.120 | 22 | 0.776 [0.709, 0.839] | 0.849 | 0.029 |

## By fold (mean per-patient Dice of each fold's test patients)

| model | mode | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| UNI site CV (4090) | tile | 0.840 | 0.769 | 0.793 | 0.732 | 0.825 |

