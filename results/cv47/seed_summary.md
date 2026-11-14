# Seed check: baseline U-Net fold 0, seed 1 vs seed 0

_2026-11-04 16:37, machine: 4090 (report47.py seed)_

Fold 0 of cv47 (9 test patients); same data, folds and settings, only --seed differs (it also changes the 20,000 val tiles drawn).

Per-patient Dice = mean of the patient's slide Dice; CI = 95% bootstrap of the mean over patients.

## Results

| model | mode | patients | Dice per patient (mean +- SD [95% CI]) | median | patient-pooled Dice | pooled Dice (all pixels) | sensitivity | precision | false-tumour rate | clean-tile AUC | vs U-Net baseline (Dice diff [CI]) | better/worse | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net seed 0 (baseline) fold 0 | tile | 9 | 0.762 +- 0.110 [0.693, 0.828] | 0.788 | 0.781 +- 0.120 [0.706, 0.854] | 0.858 | 0.837 +- 0.133 [0.750, 0.908] | 0.761 +- 0.173 [0.651, 0.863] | 0.104 +- 0.098 [0.049, 0.168] | 0.943 +- 0.050 [0.910, 0.969] | +0.000 [+0.000, +0.000] | 0/0 | 1 |
| U-Net seed 1 fold 0 | tile | 9 | 0.771 +- 0.104 [0.708, 0.832] | 0.780 | 0.791 +- 0.113 [0.722, 0.860] | 0.866 | 0.846 +- 0.137 [0.755, 0.920] | 0.774 +- 0.166 [0.666, 0.869] | 0.103 +- 0.103 [0.046, 0.173] | 0.946 +- 0.048 [0.913, 0.970] | +0.010 [+0.001, +0.019] | 6/3 | 0.129 |
| U-Net seed 0 (baseline) fold 0 | full | 9 | 0.780 +- 0.116 [0.709, 0.849] | 0.795 | 0.794 +- 0.125 [0.716, 0.870] | 0.885 | 0.890 +- 0.132 [0.801, 0.958] | 0.745 +- 0.177 [0.632, 0.849] | 0.120 +- 0.113 [0.056, 0.195] | 0.958 +- 0.048 [0.925, 0.982] | +0.000 [+0.000, +0.000] | 0/0 | 1 |
| U-Net seed 1 fold 0 | full | 9 | 0.796 +- 0.106 [0.732, 0.860] | 0.807 | 0.811 +- 0.115 [0.739, 0.879] | 0.892 | 0.883 +- 0.141 [0.788, 0.954] | 0.776 +- 0.156 [0.674, 0.866] | 0.102 +- 0.101 [0.047, 0.171] | 0.961 +- 0.047 [0.929, 0.983] | +0.016 [+0.006, +0.029] | 8/1 | 0.0195 |

## By specimen type (per-patient Dice)

| model | mode | biopsy n | biopsy Dice mean [CI] | biopsy median | biopsy false-tumour | large n | large Dice mean [CI] | large median | large false-tumour |
|---|---|---|---|---|---|---|---|---|---|
| U-Net seed 0 (baseline) fold 0 | tile | 5 | 0.715 [0.640, 0.791] | 0.683 | 0.167 | 4 | 0.819 [0.714, 0.879] | 0.865 | 0.025 |
| U-Net seed 1 fold 0 | tile | 5 | 0.729 [0.662, 0.805] | 0.683 | 0.166 | 4 | 0.824 [0.721, 0.880] | 0.871 | 0.025 |
| U-Net seed 0 (baseline) fold 0 | full | 5 | 0.722 [0.649, 0.795] | 0.678 | 0.190 | 4 | 0.853 [0.752, 0.905] | 0.902 | 0.031 |
| U-Net seed 1 fold 0 | full | 5 | 0.745 [0.677, 0.823] | 0.694 | 0.162 | 4 | 0.860 [0.767, 0.913] | 0.900 | 0.028 |

## By fold (mean per-patient Dice of each fold's test patients)

| model | mode | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| U-Net seed 0 (baseline) fold 0 | tile | 0.762 |  |  |  |  |
| U-Net seed 1 fold 0 | tile | 0.771 |  |  |  |  |
| U-Net seed 0 (baseline) fold 0 | full | 0.780 |  |  |  |  |
| U-Net seed 1 fold 0 | full | 0.796 |  |  |  |  |


Reference given by the user: seed 0 0.7616 (tile) / 0.7800 (full); H100 run 0.7774 / 0.8065 (per-patient mean Dice, fold 0, 9 patients). The H100 per-patient table is not on this PC.

The seed-1 run crashed at 13:04 on 2026-11-04 (C: full) and resumed from its epoch-12 checkpoint; its log.csv lists epoch 13 twice, the second row is the real one. Model and scores are unaffected.
