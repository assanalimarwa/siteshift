# Label-convention experiment, 47-patient folds, tile mode

_labels47.py report, 2026-11-12 13:41. Conventions and rules: labels47.py docstring (fixed before any result). Same recipe and folds as the baseline U-Net (cv47/imagenet); training (train + val) on transformed labels, test against the ORIGINAL labels, compared on the same patients with the matching baseline folds. Per-patient Dice = mean of the patient's slide Dice; 95% CIs: 10,000 patient resamples, seed 0; paired differences with patients better/worse and Wilcoxon p._

## Transformed labels against the originals (all 47 patients, per-patient mean of slide Dice)

| convention | label Dice vs original (mean [CI]) | median | min |
|---|---|---|---|
| outlines dilated by 50 um | 0.8791 [0.8580, 0.8993] | 0.902 | 0.732 |
| outlines eroded by 50 um | 0.8300 [0.7958, 0.8628] | 0.871 | 0.591 |

## Results (test patients of the finished folds, original labels)

| convention | model | folds | patients | label Dice of the training convention | Dice per patient (mean [CI]) | median | vs baseline (diff [CI]) | better/worse | Wilcoxon p | clean-tile AUC | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| outlines dilated by 50 um | baseline U-Net (original labels) | 0,1,2,3,4 | 47 | 1 | 0.7201 [0.6683, 0.7682] | 0.775 |  |  |  | 0.9323 [0.9108, 0.9508] | 0.7826 [0.7277, 0.8310] | 0.7819 [0.7285, 0.8312] | 0.0966 [0.0714, 0.1249] |
| outlines dilated by 50 um | U-Net, outlines dilated by 50 um | 0,1,2,3,4 | 47 | 0.8791 [0.8580, 0.8993] | 0.6960 [0.6483, 0.7416] | 0.703 | -0.0241 [-0.0391, -0.0078] | 15/32 | 0.00102 | 0.9402 [0.9215, 0.9568] | 0.8241 [0.7728, 0.8698] | 0.7006 [0.6480, 0.7515] | 0.1336 [0.1005, 0.1706] |
| outlines eroded by 50 um | baseline U-Net (original labels) | 0,1,2,3,4 | 47 | 1 | 0.7201 [0.6683, 0.7682] | 0.775 |  |  |  | 0.9323 [0.9108, 0.9508] | 0.7826 [0.7277, 0.8310] | 0.7819 [0.7285, 0.8312] | 0.0966 [0.0714, 0.1249] |
| outlines eroded by 50 um | U-Net, outlines eroded by 50 um | 0,1,2,3,4 | 47 | 0.8300 [0.7958, 0.8628] | 0.6653 [0.6113, 0.7156] | 0.706 | -0.0548 [-0.0724, -0.0381] | 4/43 | 8.58e-09 | 0.9182 [0.8925, 0.9398] | 0.6594 [0.6048, 0.7070] | 0.8158 [0.7632, 0.8644] | 0.0610 [0.0421, 0.0833] |

## Attribution (protocol.py, parity-verified Euclidean band; err = bin / (2 tp + fp + fn) per patient, mean over patients; the change row is paired over patients, convention minus baseline)

| convention | model | patient-pooled Dice | err missed | err fill | err boundary | err false | share missed | share fill | share boundary | share false | recall (regions) | dice_found | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| outlines dilated by 50 um | baseline U-Net (original labels) | 0.7554 | 0.0206 | 0.0670 | 0.0615 | 0.0955 | 4% | 30% | 31% | 35% | 0.879 | 0.824 |  |
| outlines dilated by 50 um | U-Net, outlines dilated by 50 um | 0.7312 | 0.0176 | 0.0525 | 0.0798 | 0.1189 | 4% | 23% | 34% | 39% | 0.909 | 0.806 |  |
| outlines dilated by 50 um | change (convention - baseline) |  | -0.0029 [-0.0114, +0.0040] | -0.0145 [-0.0203, -0.0096] | +0.0183 [+0.0125, +0.0242] | +0.0234 [+0.0155, +0.0316] |  |  |  |  |  |  | +0.0079 [+0.0034, +0.0133] |
| outlines eroded by 50 um | baseline U-Net (original labels) | 0.7554 | 0.0206 | 0.0670 | 0.0615 | 0.0955 | 4% | 30% | 31% | 35% | 0.879 | 0.824 |  |
| outlines eroded by 50 um | U-Net, outlines eroded by 50 um | 0.7029 | 0.0342 | 0.0917 | 0.0938 | 0.0775 | 6% | 34% | 36% | 23% | 0.792 | 0.759 |  |
| outlines eroded by 50 um | change (convention - baseline) |  | +0.0136 [+0.0033, +0.0313] | +0.0247 [+0.0162, +0.0333] | +0.0323 [+0.0211, +0.0428] | -0.0181 [-0.0304, -0.0069] |  |  |  |  |  |  | -0.0141 [-0.0209, -0.0082] |
