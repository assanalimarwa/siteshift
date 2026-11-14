# Reverse transfer: our fold U-Nets on public test data

_2026-11-09 15:26, machine: 4090 (report47.py reverse)_

The five fold U-Nets of the 47-patient CV (runs/cv47/imagenet) and, for comparison, of the 17-patient CV (runs/cv/imagenet, scored 10-24/25); every unit is scored by all five models, per unit the mean of the five. Reference: the model trained on that dataset (at home). Mean over units [95% bootstrap CI]; paired differences on the same units.

## Karolinska test, 1,092 slides, tile mode (unit = slide)

| trained on                          |   units | Dice                 | sensitivity          | false-tumour rate    | tile AUC             | clean tile AUC       |
|:------------------------------------|--------:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ours, 47-patient CV                 |    1092 | 0.423 [0.405, 0.441] | 0.317 [0.301, 0.333] | 0.025 [0.022, 0.028] | 0.712 [0.703, 0.722] | 0.737 [0.727, 0.746] |
| ours, 17-patient CV                 |    1092 | 0.416 [0.398, 0.434] | 0.310 [0.294, 0.326] | 0.025 [0.022, 0.029] | 0.720 [0.710, 0.729] | 0.745 [0.736, 0.755] |
| Karolinska, all 3,814 slides (home) |    1092 | 0.713 [0.698, 0.728] | 0.633 [0.616, 0.650] | 0.056 [0.049, 0.063] | 0.809 [0.800, 0.818] | 0.836 [0.827, 0.845] |

- ours, 47-patient CV: mean Dice per fold model 0.377 to 0.490
- ours, 17-patient CV: mean Dice per fold model 0.346 to 0.477

| difference                                                    | metric       |   units | mean [95% CI]           | higher / lower   |
|:--------------------------------------------------------------|:-------------|--------:|:------------------------|:-----------------|
| ours, 47-patient CV minus ours, 17-patient CV                 | dice         |     706 | +0.007 [+0.001, +0.012] | 405 / 298        |
| ours, 47-patient CV minus ours, 17-patient CV                 | sensitivity  |     706 | +0.007 [+0.002, +0.012] | 405 / 298        |
| ours, 47-patient CV minus ours, 17-patient CV                 | false_tumour |    1092 | -0.000 [-0.002, +0.001] | 460 / 610        |
| ours, 47-patient CV minus ours, 17-patient CV                 | tile_auc     |     662 | -0.008 [-0.010, -0.005] | 267 / 394        |
| ours, 47-patient CV minus ours, 17-patient CV                 | clean_auc    |     661 | -0.009 [-0.012, -0.006] | 271 / 387        |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | dice         |     706 | +0.290 [+0.278, +0.302] | 697 / 8          |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | sensitivity  |     706 | +0.316 [+0.305, +0.327] | 701 / 4          |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | false_tumour |    1092 | +0.031 [+0.025, +0.036] | 665 / 413        |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | tile_auc     |     662 | +0.097 [+0.091, +0.102] | 614 / 48         |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | clean_auc    |     661 | +0.100 [+0.094, +0.106] | 613 / 47         |

## Karolinska test, 1,092 slides, full mode (context + blending + flips) (unit = slide)

| trained on                          |   units | Dice                 | sensitivity          | false-tumour rate    | tile AUC             | clean tile AUC       |
|:------------------------------------|--------:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ours, 47-patient CV                 |    1092 | 0.470 [0.450, 0.490] | 0.367 [0.349, 0.386] | 0.023 [0.020, 0.027] | 0.761 [0.750, 0.771] | 0.785 [0.775, 0.796] |
| ours, 17-patient CV                 |    1092 | 0.421 [0.401, 0.440] | 0.317 [0.300, 0.335] | 0.015 [0.013, 0.018] | 0.746 [0.735, 0.756] | 0.769 [0.759, 0.780] |
| Karolinska, all 3,814 slides (home) |    1092 | 0.790 [0.777, 0.804] | 0.753 [0.738, 0.769] | 0.077 [0.068, 0.086] | 0.897 [0.889, 0.904] | 0.919 [0.912, 0.926] |

- ours, 47-patient CV: mean Dice per fold model 0.402 to 0.546
- ours, 17-patient CV: mean Dice per fold model 0.330 to 0.494

| difference                                                    | metric       |   units | mean [95% CI]           | higher / lower   |
|:--------------------------------------------------------------|:-------------|--------:|:------------------------|:-----------------|
| ours, 47-patient CV minus ours, 17-patient CV                 | dice         |     706 | +0.049 [+0.043, +0.056] | 524 / 175        |
| ours, 47-patient CV minus ours, 17-patient CV                 | sensitivity  |     706 | +0.050 [+0.043, +0.056] | 528 / 171        |
| ours, 47-patient CV minus ours, 17-patient CV                 | false_tumour |    1092 | +0.008 [+0.006, +0.010] | 684 / 322        |
| ours, 47-patient CV minus ours, 17-patient CV                 | tile_auc     |     662 | +0.015 [+0.012, +0.019] | 430 / 225        |
| ours, 47-patient CV minus ours, 17-patient CV                 | clean_auc    |     661 | +0.016 [+0.012, +0.020] | 415 / 236        |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | dice         |     706 | +0.320 [+0.306, +0.335] | 693 / 12         |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | sensitivity  |     706 | +0.386 [+0.374, +0.399] | 702 / 3          |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | false_tumour |    1092 | +0.054 [+0.046, +0.062] | 783 / 289        |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | tile_auc     |     662 | +0.136 [+0.129, +0.144] | 633 / 28         |
| Karolinska, all 3,814 slides (home) minus ours, 47-patient CV | clean_auc    |     661 | +0.134 [+0.126, +0.143] | 619 / 33         |

## SICAPv2 held-out (val 1,883 + guard 1,077 tiles), tile mode (unit = overlap region with tumour)

Tiles enlarged x1.75 to our scale (evaluate_sicap.py).

| model                |   regions with tumour | Dice per region      |   pooled Dice (all pixels) |
|:---------------------|----------------------:|:---------------------|---------------------------:|
| ours, 47-patient CV  |                    19 | 0.623 [0.509, 0.724] |                      0.688 |
| ours, 17-patient CV  |                    19 | 0.658 [0.557, 0.747] |                      0.696 |
| SICAPv2 U-Net (home) |                    19 | 0.803 [0.745, 0.855] |                      0.809 |

- ours, 47-patient CV minus ours, 17-patient CV: -0.035 [-0.076, +0.001], higher / lower 7 / 12 regions
- SICAPv2 U-Net (home) minus ours, 47-patient CV: +0.180 [+0.114, +0.261], higher / lower 18 / 1 regions
