# DINOv2-L + head: 47-patient 5-fold patient CV, tile mode

Written 2026-11-05 15:33 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server (GPU 1)**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2-L + head | 47 | 0.731 [0.680, 0.778] | 0.177 | 0.797 | 0.766 [0.718, 0.809] | 0.780 [0.724, 0.829] | 0.790 [0.738, 0.836] | 0.090 [0.066, 0.117] | 0.960 [0.945, 0.973] |
| baseline U-Net (4090) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| DINOv2-L + head | 0.841 | 0.827 | 0.785 | 0.904 | 3.5% |
| baseline U-Net (4090) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | baseline per-patient Dice | difference | model pooled Dice | baseline pooled Dice | model clean AUC | baseline clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.749 | 0.722 | +0.028 | 0.784 | 0.760 | 0.963 | 0.940 |
| large section | 22 | 0.710 | 0.718 | -0.008 | 0.745 | 0.751 | 0.956 | 0.924 |
| fold 0 | 9 | 0.778 | 0.762 | +0.017 | 0.807 | 0.781 | 0.975 | 0.943 |
| fold 1 | 9 | 0.716 | 0.719 | -0.003 | 0.759 | 0.765 | 0.933 | 0.918 |
| fold 2 | 10 | 0.737 | 0.719 | +0.018 | 0.761 | 0.743 | 0.972 | 0.957 |
| fold 3 | 10 | 0.649 | 0.636 | +0.013 | 0.668 | 0.666 | 0.942 | 0.888 |
| fold 4 | 9 | 0.783 | 0.773 | +0.010 | 0.844 | 0.833 | 0.977 | 0.958 |

## Paired differences: model minus baseline U-Net (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | +0.011 [-0.002, +0.024] | 27 / 20 | 0.119 |
| patient-pooled Dice | 47 | +0.010 [-0.004, +0.024] | 30 / 17 | 0.0676 |
| sensitivity | 47 | -0.003 [-0.026, +0.016] | 24 / 23 | 0.718 |
| precision | 47 | +0.008 [-0.011, +0.026] | 28 / 19 | 0.109 |
| false-tumour rate (lower is better) | 47 | -0.007 [-0.020, +0.007] | 20 / 27 | 0.147 |
| clean-tile AUC | 47 | +0.028 [+0.017, +0.040] | 40 / 7 | 1.68e-08 |

Headline wording: **tie (difference below 0.03)**. higher / lower = number of patients on which the model scored above / below the baseline.

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
| fold 0 | 10 | 6 | 0.8600 | 112 | H100 (GPU 1) |
| fold 1 | 13 | 9 | 0.8863 | 112 | H100 (GPU 1) |
| fold 2 | 15 | 13 | 0.8597 | 112 | H100 (GPU 1) |
| fold 3 | 15 | 15 | 0.8926 | 112 | H100 (GPU 1) |
| fold 4 | 13 | 9 | 0.8596 | 112 | H100 (GPU 1) |

Settings (config.json, identical to the baseline except the architecture): arch fm_dinov2, epochs max 15, patience 4, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/dinov2/predictions/ - 5 of 5 prediction folders finished, 2.0 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/dinov2_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It starts after the nnU-Net site CV and all-public run, and only if the estimated end is before 2026-11-09 08:00 (the queue decides when it gets there; otherwise it is logged as not started).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

