# Threshold analysis, 47 patients, tile mode

_threshold47.py report, 2026-11-11 02:17. Definitions: threshold47.py docstring (fixed before any result). Per-patient Dice = mean of the patient's slide Dice; 95% CIs: 10,000 patient resamples, seed 0; paired differences with patients better/worse and Wilcoxon signed-rank p._

tau grid 0.05-0.95 in steps of 0.05; tau* = the grid value with the best mean per-patient Dice over all 47 patients (in-sample). E_thr = Dice(tau*) - Dice(0.5). The selection-aware CI chooses tau* again in every resample; 'tau* = 0.5' gives the share of resamples in which 0.5 itself was best. The site-calibrated threshold is chosen per few-shot fold on its two adaptation patients only and applied to that fold's unseen test patients (out-of-sample).

## Results: E_thr (one threshold per model)

| model | tau* | Dice at 0.5 | Dice at tau* | E_thr (fixed-tau* CI) | E_thr selection-aware 95% CI | better/worse | Wilcoxon p | resamples with tau* = 0.5 | tau* over resamples | max slide |Dice uint8 - evaluate.py| at 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net site CV (5 folds, out-of-fold) | 0.55 | 0.7201 [0.6683, 0.7682] | 0.7205 [0.6689, 0.7684] | +0.0003 [-0.0017, +0.0024] | [+0.0000, +0.0054] | 22/25 | 0.626 | 10% | 0.30: 2%, 0.35: 2%, 0.40: 6%, 0.45: 16%, 0.50: 10%, 0.55: 26%, 0.60: 27%, 0.65: 9%, 0.70: 1% | 5.6e-05 |
| U-Net all public (A) | 0.35 | 0.5688 [0.5152, 0.6189] | 0.5703 [0.5159, 0.6207] | +0.0016 [-0.0031, +0.0065] | [+0.0000, +0.0101] | 23/24 | 0.996 | 7% | 0.20: 4%, 0.25: 20%, 0.30: 20%, 0.35: 14%, 0.40: 14%, 0.45: 12%, 0.50: 7%, 0.55: 4%, 0.60: 2%, 0.65: 1% | 5.2e-05 |
| U-Net Karolinska 3,814 | 0.90 | 0.5509 [0.4970, 0.6009] | 0.5761 [0.5249, 0.6230] | +0.0253 [+0.0030, +0.0460] | [+0.0115, +0.0464] | 34/13 | 0.00908 | 0% | 0.75: 1%, 0.80: 14%, 0.85: 29%, 0.90: 54%, 0.95: 2% | 1.1e-04 |

## Site-calibrated threshold (chosen on 2 adaptation patients per fold, applied to the fold's unseen patients)

| model | tau_k folds 0-4 | adaptation patients | Dice at 0.5 | Dice calibrated | calibrated - 0.5 | better/worse | Wilcoxon p | calibrated - tau* (in-sample) | better/worse vs tau* |
|---|---|---|---|---|---|---|---|---|---|
| U-Net site CV (5 folds, out-of-fold) | 0.20, 0.65, 0.70, 0.70, 0.25 | 8,46; 9,2; 35,9; 36,42; 15,45 | 0.7201 [0.6683, 0.7682] | 0.7096 [0.6562, 0.7592] | -0.0105 [-0.0200, -0.0009] | 15/32 | 0.0107 | -0.0108 [-0.0196, -0.0023] | 14/33 |
| U-Net all public (A) | 0.10, 0.50, 0.90, 0.40, 0.50 | 8,46; 9,2; 35,9; 36,42; 15,45 | 0.5688 [0.5152, 0.6189] | 0.5573 [0.5044, 0.6078] | -0.0115 [-0.0207, -0.0029] | 10/19 | 0.0282 | -0.0131 [-0.0223, -0.0042] | 17/30 |
| U-Net Karolinska 3,814 | 0.50, 0.90, 0.95, 0.40, 0.95 | 8,46; 9,2; 35,9; 36,42; 15,45 | 0.5509 [0.4970, 0.6009] | 0.5661 [0.5122, 0.6154] | +0.0152 [-0.0107, +0.0393] | 22/16 | 0.0911 | -0.0100 [-0.0260, +0.0057] | 20/18 |

## By specimen type

| model | specimen | patients | Dice at 0.5 | E_thr (tau*) | better/worse | calibrated - 0.5 | better/worse (cal) |
|---|---|---|---|---|---|---|---|
| U-Net site CV (5 folds, out-of-fold) | biopsy | 25 | 0.7218 [0.6527, 0.7827] | +0.0003 [-0.0022, +0.0026] | 13/12 | -0.0131 [-0.0257, -0.0017] | 8/17 |
| U-Net site CV (5 folds, out-of-fold) | large | 22 | 0.7182 [0.6373, 0.7921] | +0.0004 [-0.0028, +0.0040] | 9/13 | -0.0075 [-0.0220, +0.0092] | 7/15 |
| U-Net all public (A) | biopsy | 25 | 0.6293 [0.5722, 0.6858] | +0.0026 [-0.0034, +0.0094] | 14/11 | -0.0092 [-0.0199, +0.0006] | 6/9 |
| U-Net all public (A) | large | 22 | 0.5000 [0.4160, 0.5815] | +0.0003 [-0.0067, +0.0077] | 9/13 | -0.0141 [-0.0300, -0.0002] | 4/10 |
| U-Net Karolinska 3,814 | biopsy | 25 | 0.6037 [0.5563, 0.6496] | +0.0469 [+0.0231, +0.0664] | 22/3 | +0.0346 [+0.0078, +0.0607] | 13/7 |
| U-Net Karolinska 3,814 | large | 22 | 0.4908 [0.3942, 0.5801] | +0.0008 [-0.0364, +0.0370] | 12/10 | -0.0067 [-0.0509, +0.0328] | 9/9 |

## Attribution at 0.5, tau* and the site-calibrated thresholds (protocol.py, parity-verified Euclidean band)

err = bin / (2 tp + fp + fn) per patient (pooled pixels), mean over patients; the four err bins add up to 1 - patient-pooled Dice. share = bin / (fp + fn).

| model | threshold | patient-pooled Dice | err missed | err fill | err boundary | err false | share missed | share fill | share boundary | share false | false isolated / attached | recall (regions) | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net site CV (5 folds, out-of-fold) | 0.5 | 0.7554 | 0.0206 | 0.0670 | 0.0615 | 0.0955 | 4% | 30% | 31% | 35% | 0.0696 / 0.0260 | 0.879 | 0.932 |
| U-Net site CV (5 folds, out-of-fold) | tau* (0.55) | 0.7558 | 0.0240 | 0.0691 | 0.0616 | 0.0895 | 5% | 30% | 31% | 33% | 0.0649 / 0.0246 | 0.872 | 0.931 |
| U-Net site CV (5 folds, out-of-fold) | site-calibrated | 0.7470 | 0.0286 | 0.0722 | 0.0619 | 0.0903 | 5% | 30% | 30% | 34% | 0.0610 / 0.0293 | 0.868 | 0.929 |
| U-Net all public (A) | 0.5 | 0.6046 | 0.0224 | 0.1400 | 0.0789 | 0.1542 | 5% | 35% | 23% | 36% | 0.1206 / 0.0336 | 0.912 | 0.926 |
| U-Net all public (A) | tau* (0.35) | 0.6052 | 0.0142 | 0.1277 | 0.0766 | 0.1763 | 3% | 33% | 23% | 41% | 0.1380 / 0.0383 | 0.938 | 0.926 |
| U-Net all public (A) | site-calibrated | 0.5900 | 0.0215 | 0.1467 | 0.0802 | 0.1615 | 5% | 35% | 23% | 37% | 0.1233 / 0.0382 | 0.905 | 0.923 |
| U-Net Karolinska 3,814 | 0.5 | 0.5795 | 0.0054 | 0.0452 | 0.0709 | 0.2990 | 1% | 14% | 21% | 65% | 0.1795 / 0.1195 | 0.944 | 0.907 |
| U-Net Karolinska 3,814 | tau* (0.90) | 0.6107 | 0.0183 | 0.1083 | 0.0829 | 0.1797 | 4% | 28% | 26% | 42% | 0.1167 / 0.0630 | 0.831 | 0.891 |
| U-Net Karolinska 3,814 | site-calibrated | 0.5973 | 0.0201 | 0.0868 | 0.0792 | 0.2165 | 4% | 23% | 25% | 48% | 0.1323 / 0.0843 | 0.859 | 0.896 |

Change in each error bin (0.5 minus the other threshold: positive = that error shrinks), paired over patients:

| model | change | err missed | missed better/worse | err fill | fill better/worse | err boundary | boundary better/worse | err false | false better/worse |
|---|---|---|---|---|---|---|---|---|---|
| U-Net site CV (5 folds, out-of-fold) | tau* - 0.5 | -0.0034 [-0.0092, -0.0001] | 0/46 | -0.0021 [-0.0048, +0.0019] | 2/45 | -0.0001 [-0.0011, +0.0016] | 5/42 | +0.0060 [+0.0041, +0.0084] | 47/0 |
| U-Net site CV (5 folds, out-of-fold) | site-calibrated - 0.5 | -0.0080 [-0.0200, +0.0001] | 18/29 | -0.0052 [-0.0148, +0.0039] | 21/26 | -0.0004 [-0.0034, +0.0032] | 20/27 | +0.0053 [-0.0075, +0.0192] | 29/18 |
| U-Net all public (A) | tau* - 0.5 | +0.0082 [+0.0018, +0.0172] | 46/0 | +0.0123 [+0.0038, +0.0189] | 44/3 | +0.0023 [+0.0008, +0.0036] | 40/7 | -0.0221 [-0.0274, -0.0173] | 0/47 |
| U-Net all public (A) | site-calibrated - 0.5 | +0.0008 [-0.0034, +0.0061] | 36/10 | -0.0067 [-0.0241, +0.0089] | 36/11 | -0.0014 [-0.0058, +0.0024] | 35/12 | -0.0073 [-0.0293, +0.0154] | 10/37 |
| U-Net Karolinska 3,814 | tau* - 0.5 | -0.0129 [-0.0208, -0.0063] | 0/47 | -0.0632 [-0.0829, -0.0448] | 1/46 | -0.0121 [-0.0148, -0.0097] | 1/46 | +0.1193 [+0.1009, +0.1391] | 47/0 |
| U-Net Karolinska 3,814 | site-calibrated - 0.5 | -0.0147 [-0.0280, -0.0045] | 17/28 | -0.0416 [-0.0625, -0.0233] | 20/27 | -0.0084 [-0.0118, -0.0052] | 19/28 | +0.0825 [+0.0551, +0.1118] | 28/19 |

## Mean per-patient Dice over the grid

| tau | U-Net site CV (5 folds, out-of-fold) | U-Net all public (A) | U-Net Karolinska 3,814 |
|---|---|---|---|
| 0.05 | 0.6581 | 0.5175 | 0.3961 |
| 0.10 | 0.6845 | 0.5525 | 0.4665 |
| 0.15 | 0.6998 | 0.5626 | 0.4928 |
| 0.20 | 0.7078 | 0.5677 | 0.5141 |
| 0.25 | 0.7118 | 0.5696 | 0.5224 |
| 0.30 | 0.7151 | 0.5703 | 0.5285 |
| 0.35 | 0.7171 | 0.5703 | 0.5346 |
| 0.40 | 0.7187 | 0.5701 | 0.5406 |
| 0.45 | 0.7196 | 0.5696 | 0.5458 |
| 0.50 | 0.7201 | 0.5688 | 0.5509 |
| 0.55 | 0.7205 | 0.5677 | 0.5556 |
| 0.60 | 0.7201 | 0.5663 | 0.5598 |
| 0.65 | 0.7184 | 0.5646 | 0.5633 |
| 0.70 | 0.7151 | 0.5623 | 0.5668 |
| 0.75 | 0.7095 | 0.5588 | 0.5702 |
| 0.80 | 0.7000 | 0.5533 | 0.5734 |
| 0.85 | 0.6859 | 0.5447 | 0.5757 |
| 0.90 | 0.6537 | 0.5266 | 0.5761 |
| 0.95 | 0.5606 | 0.4786 | 0.5633 |
