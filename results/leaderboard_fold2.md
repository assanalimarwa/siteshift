# Leaderboard baselines on the withheld fold (fold 2)

_leaderboard_fold2.py. Fold 2 = patients 6, 10, 23, 25, 28, 30, 34, 36, 38, 46 (5 biopsy, 5 large-section: 10, 25, 30, 34, 36). Definitions in the script's docstring; tile mode, threshold 0.5; computed from the released per-patient results (results/cv47), the same runs as the 47-patient tables._

**Small test set.** With the 10 fold-2 patients the 95% interval of one model's Dice is about +-0.14 (median over the Track A rows; Track B, 5 patients: about +-0.24); the paired difference to the site U-Net on the same patients is narrower, about +-0.07. Differences below 0.03 are ties under the benchmark rules, and only differences whose paired interval excludes 0 count. Use these rows as baselines to beat, not as a ranking of the baselines.

## Track A: cross-site transfer (public data only), fold 2, all 10 patients

| rank | model | training data | patients | Dice [95% CI] | vs site U-Net [95% CI] (better / worse) | patient-pooled Dice | clean-tile AUC | detection error | delineation error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SICAPv2 U-Net r80 | public data only | 10 | 0.610 [0.442, 0.759] | -0.109 [-0.182, -0.045] (1 / 9) | 0.644 | 0.897 | 0.227 | 0.128 |
| 2 | nnU-Net-style all-public (H100) | public data only | 10 | 0.589 [0.441, 0.723] | -0.130 [-0.181, -0.082] (0 / 10) | 0.618 | 0.916 | 0.207 | 0.175 |
| 3 | SICAPv2 U-Net (seed 1) | public data only | 10 | 0.586 [0.416, 0.739] | -0.133 [-0.213, -0.068] (1 / 9) | 0.615 | 0.886 | 0.225 | 0.160 |
| 4 | SICAPv2 U-Net | public data only | 10 | 0.582 [0.412, 0.736] | -0.137 [-0.213, -0.076] (0 / 10) | 0.612 | 0.886 | 0.235 | 0.153 |
| 5 | SICAPv2 U-Net r40 | public data only | 10 | 0.580 [0.407, 0.737] | -0.139 [-0.215, -0.072] (1 / 9) | 0.611 | 0.882 | 0.278 | 0.111 |
| 6 | all-public UNI | public data only | 10 | 0.578 [0.443, 0.700] | -0.141 [-0.190, -0.094] (0 / 10) | 0.609 | 0.952 | 0.212 | 0.179 |
| 7 | mixed 600 UNI | public data only | 10 | 0.571 [0.450, 0.688] | -0.149 [-0.200, -0.097] (0 / 10) | 0.589 | 0.951 | 0.137 | 0.273 |
| 8 | all-public (A) U-Net | public data only | 10 | 0.568 [0.425, 0.698] | -0.151 [-0.205, -0.100] (0 / 10) | 0.604 | 0.928 | 0.209 | 0.187 |
| 9 | SICAPv2 UniMed-CLIP | public data only | 10 | 0.560 [0.402, 0.704] | -0.159 [-0.233, -0.089] (1 / 9) | 0.593 | 0.915 | 0.227 | 0.181 |
| 10 | SegFormer-B2 all-public (H100) | public data only | 10 | 0.558 [0.408, 0.691] | -0.161 [-0.214, -0.113] (0 / 10) | 0.594 | 0.892 | 0.218 | 0.188 |
| 11 | Karolinska 600 UNI | public data only | 10 | 0.554 [0.411, 0.693] | -0.166 [-0.205, -0.122] (0 / 10) | 0.586 | 0.955 | 0.344 | 0.070 |
| 12 | SICAPv2 Virchow2 | public data only | 10 | 0.554 [0.377, 0.718] | -0.166 [-0.268, -0.073] (2 / 8) | 0.560 | 0.889 | 0.225 | 0.215 |
| 13 | mixed 600 U-Net | public data only | 10 | 0.548 [0.405, 0.682] | -0.171 [-0.225, -0.118] (0 / 10) | 0.571 | 0.912 | 0.206 | 0.223 |
| 14 | Karolinska 3,814 U-Net | public data only | 10 | 0.520 [0.376, 0.647] | -0.199 [-0.242, -0.158] (0 / 10) | 0.549 | 0.888 | 0.339 | 0.112 |
| 15 | Karolinska 1,000 U-Net | public data only | 10 | 0.506 [0.369, 0.625] | -0.213 [-0.295, -0.152] (0 / 10) | 0.541 | 0.868 | 0.312 | 0.147 |
| 16 | Karolinska 155 U-Net | public data only | 10 | 0.497 [0.364, 0.611] | -0.222 [-0.328, -0.135] (0 / 10) | 0.530 | 0.848 | 0.286 | 0.184 |
| 17 | Karolinska 419 U-Net | public data only | 10 | 0.496 [0.364, 0.609] | -0.223 [-0.313, -0.160] (0 / 10) | 0.523 | 0.852 | 0.305 | 0.172 |
| 18 | Radboud U-Net | public data only | 10 | 0.472 [0.355, 0.589] | -0.247 [-0.304, -0.185] (0 / 10) | 0.494 | 0.907 | 0.196 | 0.310 |
| 19 | Karolinska 600 U-Net | public data only | 10 | 0.465 [0.328, 0.590] | -0.254 [-0.364, -0.178] (0 / 10) | 0.491 | 0.838 | 0.372 | 0.137 |
| 20 | SICAPv2 UNI | public data only | 10 | 0.424 [0.267, 0.579] | -0.295 [-0.416, -0.171] (1 / 9) | 0.425 | 0.852 | 0.358 | 0.217 |
| 21 | SICAPv2 DINOv2 | public data only | 10 | 0.424 [0.260, 0.586] | -0.296 [-0.400, -0.195] (0 / 10) | 0.429 | 0.832 | 0.288 | 0.283 |
| 22 | SICAPv2 CONCH | public data only | 10 | 0.364 [0.204, 0.519] | -0.356 [-0.475, -0.239] (0 / 10) | 0.377 | 0.809 | 0.385 | 0.238 |
| 23 | Karolinska 20 U-Net | public data only | 10 | 0.360 [0.226, 0.503] | -0.359 [-0.506, -0.231] (0 / 10) | 0.412 | 0.760 | 0.314 | 0.275 |

Width of the 95% Dice interval: median 0.28 (range 0.23-0.34); of the paired difference to the site U-Net: median 0.14.

## Track B: large sections, fold 2, the 5 large-section patients

| rank | model | training data | patients | Dice [95% CI] | vs site U-Net [95% CI] (better / worse) | patient-pooled Dice | clean-tile AUC | detection error | delineation error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | nnU-Net-style all-public (H100) | public data only | 5 | 0.553 [0.294, 0.794] | -0.127 [-0.199, -0.055] (0 / 5) | 0.588 | 0.885 | 0.273 | 0.138 |
| 2 | Karolinska 600 UNI | public data only | 5 | 0.551 [0.303, 0.800] | -0.129 [-0.192, -0.066] (0 / 5) | 0.599 | 0.964 | 0.354 | 0.046 |
| 3 | all-public UNI | public data only | 5 | 0.508 [0.294, 0.731] | -0.173 [-0.210, -0.138] (0 / 5) | 0.552 | 0.949 | 0.272 | 0.175 |
| 4 | SegFormer-B2 all-public (H100) | public data only | 5 | 0.504 [0.246, 0.740] | -0.176 [-0.240, -0.113] (0 / 5) | 0.551 | 0.836 | 0.299 | 0.150 |
| 5 | all-public (A) U-Net | public data only | 5 | 0.497 [0.267, 0.709] | -0.183 [-0.221, -0.146] (0 / 5) | 0.546 | 0.894 | 0.297 | 0.158 |
| 6 | mixed 600 UNI | public data only | 5 | 0.496 [0.314, 0.680] | -0.184 [-0.234, -0.132] (0 / 5) | 0.511 | 0.937 | 0.174 | 0.315 |
| 7 | SICAPv2 U-Net r80 | public data only | 5 | 0.491 [0.232, 0.749] | -0.190 [-0.263, -0.114] (0 / 5) | 0.530 | 0.835 | 0.330 | 0.139 |
| 8 | mixed 600 U-Net | public data only | 5 | 0.474 [0.246, 0.702] | -0.206 [-0.261, -0.155] (0 / 5) | 0.498 | 0.868 | 0.276 | 0.225 |
| 9 | SICAPv2 U-Net | public data only | 5 | 0.470 [0.199, 0.740] | -0.211 [-0.307, -0.126] (0 / 5) | 0.499 | 0.818 | 0.338 | 0.163 |
| 10 | SICAPv2 U-Net r40 | public data only | 5 | 0.466 [0.200, 0.731] | -0.215 [-0.307, -0.132] (0 / 5) | 0.497 | 0.827 | 0.391 | 0.112 |
| 11 | SICAPv2 U-Net (seed 1) | public data only | 5 | 0.465 [0.198, 0.733] | -0.215 [-0.313, -0.128] (0 / 5) | 0.497 | 0.815 | 0.327 | 0.176 |
| 12 | SICAPv2 UniMed-CLIP | public data only | 5 | 0.465 [0.206, 0.724] | -0.215 [-0.301, -0.141] (0 / 5) | 0.491 | 0.881 | 0.377 | 0.133 |
| 13 | SICAPv2 Virchow2 | public data only | 5 | 0.455 [0.168, 0.742] | -0.225 [-0.354, -0.104] (0 / 5) | 0.445 | 0.850 | 0.363 | 0.191 |
| 14 | Karolinska 3,814 U-Net | public data only | 5 | 0.454 [0.207, 0.681] | -0.227 [-0.292, -0.153] (0 / 5) | 0.488 | 0.811 | 0.398 | 0.113 |
| 15 | Karolinska 1,000 U-Net | public data only | 5 | 0.419 [0.184, 0.634] | -0.261 [-0.402, -0.149] (0 / 5) | 0.466 | 0.770 | 0.354 | 0.180 |
| 16 | Karolinska 419 U-Net | public data only | 5 | 0.403 [0.187, 0.591] | -0.277 [-0.430, -0.165] (0 / 5) | 0.431 | 0.747 | 0.345 | 0.224 |
| 17 | Radboud U-Net | public data only | 5 | 0.380 [0.213, 0.560] | -0.300 [-0.357, -0.234] (0 / 5) | 0.401 | 0.875 | 0.259 | 0.340 |
| 18 | Karolinska 155 U-Net | public data only | 5 | 0.377 [0.192, 0.525] | -0.303 [-0.463, -0.172] (0 / 5) | 0.424 | 0.743 | 0.333 | 0.243 |
| 19 | Karolinska 600 U-Net | public data only | 5 | 0.372 [0.154, 0.595] | -0.308 [-0.510, -0.167] (0 / 5) | 0.403 | 0.728 | 0.430 | 0.167 |
| 20 | SICAPv2 DINOv2 | public data only | 5 | 0.339 [0.090, 0.587] | -0.342 [-0.456, -0.249] (0 / 5) | 0.326 | 0.755 | 0.421 | 0.253 |
| 21 | SICAPv2 UNI | public data only | 5 | 0.302 [0.093, 0.580] | -0.378 [-0.526, -0.230] (0 / 5) | 0.267 | 0.774 | 0.599 | 0.134 |
| 22 | SICAPv2 CONCH | public data only | 5 | 0.270 [0.055, 0.536] | -0.410 [-0.544, -0.277] (0 / 5) | 0.244 | 0.722 | 0.594 | 0.161 |
| 23 | Karolinska 20 U-Net | public data only | 5 | 0.213 [0.100, 0.320] | -0.468 [-0.667, -0.279] (0 / 5) | 0.241 | 0.668 | 0.391 | 0.368 |

Width of the 95% Dice interval: median 0.48 (range 0.22-0.57); of the paired difference to the site U-Net: median 0.17.

## Track C: few-shot adaptation on fold 2's own draws (2 / 4 / 8 patients outside fold 2), scored on fold 2

| rank | model | training data | patients | Dice [95% CI] | vs site U-Net [95% CI] (better / worse) | patient-pooled Dice | clean-tile AUC | detection error | delineation error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | few-shot UNI from all-public, 2 patients | start model + n fold-2 adaptation patients | 10 | 0.753 [0.631, 0.856] | +0.034 [+0.003, +0.068] (8 / 2) | 0.775 | 0.971 | 0.070 | 0.155 |
| 2 | few-shot UNI, fresh head, 2 patients | start model + n fold-2 adaptation patients | 10 | 0.737 [0.625, 0.833] | +0.018 [-0.012, +0.050] (6 / 4) | 0.761 | 0.962 | 0.059 | 0.179 |
| 3 | few-shot UNI, fresh head, 8 patients | start model + n fold-2 adaptation patients | 10 | 0.737 [0.603, 0.848] | +0.018 [-0.009, +0.042] (7 / 3) | 0.752 | 0.952 | 0.079 | 0.169 |
| 4 | few-shot U-Net from Karolinska 3,814, 8 patients | start model + n fold-2 adaptation patients | 10 | 0.699 [0.568, 0.818] | -0.020 [-0.037, -0.004] (3 / 7) | 0.723 | 0.938 | 0.121 | 0.156 |
| 5 | few-shot UNI from all-public, 8 patients | start model + n fold-2 adaptation patients | 10 | 0.698 [0.531, 0.836] | -0.021 [-0.077, +0.027] (5 / 5) | 0.715 | 0.946 | 0.147 | 0.138 |
| 6 | few-shot U-Net from Karolinska 3,814, 2 patients | start model + n fold-2 adaptation patients | 10 | 0.630 [0.517, 0.739] | -0.090 [-0.152, -0.034] (1 / 9) | 0.665 | 0.899 | 0.101 | 0.234 |
| 7 | few-shot UNI, fresh head, 4 patients | start model + n fold-2 adaptation patients | 10 | 0.617 [0.425, 0.785] | -0.103 [-0.197, -0.016] (5 / 5) | 0.618 | 0.891 | 0.234 | 0.148 |
| 8 | few-shot U-Net from ImageNet, 8 patients | start model + n fold-2 adaptation patients | 10 | 0.608 [0.463, 0.750] | -0.111 [-0.183, -0.059] (0 / 10) | 0.624 | 0.915 | 0.185 | 0.191 |
| 9 | few-shot UNI from all-public, 4 patients | start model + n fold-2 adaptation patients | 10 | 0.590 [0.392, 0.770] | -0.129 [-0.236, -0.034] (4 / 6) | 0.596 | 0.885 | 0.229 | 0.175 |
| 10 | few-shot U-Net from Karolinska 3,814, 4 patients | start model + n fold-2 adaptation patients | 10 | 0.546 [0.373, 0.703] | -0.173 [-0.304, -0.057] (3 / 7) | 0.544 | 0.818 | 0.228 | 0.228 |
| 11 | few-shot U-Net from ImageNet, 4 patients | start model + n fold-2 adaptation patients | 10 | 0.479 [0.313, 0.642] | -0.240 [-0.368, -0.122] (1 / 9) | 0.474 | 0.793 | 0.341 | 0.184 |
| 12 | few-shot U-Net from ImageNet, 2 patients | start model + n fold-2 adaptation patients | 10 | 0.445 [0.305, 0.587] | -0.274 [-0.406, -0.158] (0 / 10) | 0.471 | 0.823 | 0.272 | 0.257 |

Width of the 95% Dice interval: median 0.28 (range 0.21-0.38); of the paired difference to the site U-Net: median 0.12.

## Reference (not a track): site-CV fold-2 models, trained on the other 37 patients

| rank | model | training data | patients | Dice [95% CI] | vs site U-Net [95% CI] (better / worse) | patient-pooled Dice | clean-tile AUC | detection error | delineation error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Virchow2 site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.799 [0.723, 0.873] | +0.079 [+0.040, +0.135] (10 / 0) | 0.819 | 0.976 | 0.071 | 0.110 |
| 2 | UNI site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.798 [0.709, 0.878] | +0.079 [+0.048, +0.114] (10 / 0) | 0.820 | 0.973 | 0.061 | 0.119 |
| 3 | UNI site CV (frozen + head) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.793 [0.700, 0.876] | +0.074 [+0.046, +0.105] (10 / 0) | 0.817 | 0.969 | 0.063 | 0.120 |
| 4 | CONCH site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.777 [0.691, 0.861] | +0.058 [+0.025, +0.106] (10 / 0) | 0.806 | 0.968 | 0.079 | 0.114 |
| 5 | SegFormer-B2 site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.740 [0.620, 0.843] | +0.021 [+0.010, +0.035] (9 / 1) | 0.761 | 0.947 | 0.139 | 0.101 |
| 6 | DINOv2-L site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.737 [0.617, 0.841] | +0.018 [-0.009, +0.044] (7 / 3) | 0.761 | 0.972 | 0.097 | 0.142 |
| 7 | U-Net site CV | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.719 [0.598, 0.827] | reference | 0.743 | 0.957 | 0.142 | 0.115 |
| 8 | nnU-Net-style site CV (H100) | site CV fold 2 (the other 37 patients: train + val) | 10 | 0.706 [0.584, 0.816] | -0.014 [-0.036, +0.002] (4 / 6) | 0.740 | 0.946 | 0.154 | 0.106 |

Width of the 95% Dice interval: median 0.20 (range 0.15-0.23); of the paired difference to the site U-Net: median 0.06.

Track D (ranking without site labels) is proposed without a baseline.
