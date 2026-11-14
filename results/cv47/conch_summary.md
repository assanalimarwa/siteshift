# CONCH + head: 47-patient 5-fold patient CV, tile mode

Written 2026-11-05 16:10 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server (GPU 1)**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| CONCH + head | 47 | 0.781 [0.741, 0.818] | 0.138 | 0.823 | 0.814 [0.780, 0.846] | 0.823 [0.783, 0.859] | 0.823 [0.783, 0.860] | 0.078 [0.058, 0.102] | 0.961 [0.947, 0.972] |
| baseline U-Net (4090) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| CONCH + head | 0.873 | 0.856 | 0.834 | 0.916 | 3.2% |
| baseline U-Net (4090) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | baseline per-patient Dice | difference | model pooled Dice | baseline pooled Dice | model clean AUC | baseline clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.798 | 0.722 | +0.076 | 0.831 | 0.760 | 0.970 | 0.940 |
| large section | 22 | 0.761 | 0.718 | +0.043 | 0.794 | 0.751 | 0.950 | 0.924 |
| fold 0 | 9 | 0.828 | 0.762 | +0.066 | 0.850 | 0.781 | 0.972 | 0.943 |
| fold 1 | 9 | 0.783 | 0.719 | +0.063 | 0.825 | 0.765 | 0.956 | 0.918 |
| fold 2 | 10 | 0.777 | 0.719 | +0.058 | 0.806 | 0.743 | 0.968 | 0.957 |
| fold 3 | 10 | 0.717 | 0.636 | +0.081 | 0.736 | 0.666 | 0.933 | 0.888 |
| fold 4 | 9 | 0.807 | 0.773 | +0.033 | 0.862 | 0.833 | 0.976 | 0.958 |

## Paired differences: model minus baseline U-Net (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | +0.061 [+0.042, +0.083] | 42 / 5 | 1.41e-10 |
| patient-pooled Dice | 47 | +0.059 [+0.039, +0.082] | 42 / 5 | 3.44e-09 |
| sensitivity | 47 | +0.040 [+0.021, +0.063] | 36 / 11 | 4.63e-05 |
| precision | 47 | +0.041 [+0.019, +0.066] | 33 / 14 | 0.000271 |
| false-tumour rate (lower is better) | 47 | -0.019 [-0.032, -0.007] | 17 / 30 | 0.00695 |
| clean-tile AUC | 47 | +0.028 [+0.020, +0.038] | 43 / 4 | 3.6e-12 |

Headline wording: **CONCH + head ahead**. higher / lower = number of patients on which the model scored above / below the baseline.

## Pairwise differences between the finished models (same 47 patients, tile mode)
| difference | measure | patients | difference [95% CI] | higher / lower | Wilcoxon p | wording |
|---|---|---|---|---|---|---|
| UNI + head minus DINOv2-L + head | per-patient Dice | 47 | +0.062 [+0.046, +0.079] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus DINOv2-L + head | patient-pooled Dice | 47 | +0.060 [+0.042, +0.080] | 46 / 1 | 1.08e-11 |  |
| UNI + head minus DINOv2-L + head | clean-tile AUC | 47 | +0.008 [+0.001, +0.017] | 26 / 21 | 0.0542 |  |
| UNI + head minus CONCH + head | per-patient Dice | 47 | +0.012 [+0.004, +0.020] | 37 / 10 | 0.000448 | tie (difference below 0.03) |
| UNI + head minus CONCH + head | patient-pooled Dice | 47 | +0.011 [+0.004, +0.019] | 38 / 9 | 0.000118 |  |
| UNI + head minus CONCH + head | clean-tile AUC | 47 | +0.007 [+0.002, +0.013] | 35 / 12 | 0.000125 |  |
| DINOv2-L + head minus CONCH + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 4 / 43 | 1.41e-10 | CONCH + head ahead |
| DINOv2-L + head minus CONCH + head | patient-pooled Dice | 47 | -0.048 [-0.067, -0.032] | 4 / 43 | 1.79e-11 |  |
| DINOv2-L + head minus CONCH + head | clean-tile AUC | 47 | -0.001 [-0.010, +0.007] | 32 / 15 | 0.0798 |  |

Wording applies to the per-patient Dice row: a difference below 0.03 is a tie between single runs.

## Training
| run | epochs run | best epoch | best val Dice | train img/s (mean) | machine |
|---|---|---|---|---|---|
| fold 0 | 6 | 2 | 0.8930 | 347 | H100 (GPU 1) |
| fold 1 | 9 | 5 | 0.9113 | 348 | H100 (GPU 1) |
| fold 2 | 6 | 2 | 0.8936 | 347 | H100 (GPU 1) |
| fold 3 | 9 | 5 | 0.9174 | 348 | H100 (GPU 1) |
| fold 4 | 6 | 2 | 0.8866 | 347 | H100 (GPU 1) |

Settings (config.json, identical to the baseline except the architecture): arch fm_conch, epochs max 15, patience 4, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/conch/predictions/ - 5 of 5 prediction folders finished, 1.9 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/conch_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It starts after the nnU-Net site CV and all-public run, and only if the estimated end is before 2026-11-09 08:00 (the queue decides when it gets there; otherwise it is logged as not started).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

