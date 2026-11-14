# Virchow2 + head: 47-patient 5-fold patient CV, tile mode

Written 2026-11-10 22:13 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.
- Training machines (config.json): fold 0: H100 (GPU 0, shared); fold 1: H100 (GPU 0, shared); fold 2: H100 (GPU 0, shared); fold 3: H100 (GPU 0, shared); fold 4: H100 (GPU 0, shared).

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| Virchow2 + head | 47 | 0.781 [0.738, 0.821] | 0.149 | 0.829 | 0.812 [0.776, 0.847] | 0.840 [0.795, 0.879] | 0.811 [0.768, 0.850] | 0.091 [0.065, 0.122] | 0.963 [0.949, 0.974] |
| baseline U-Net (4090) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| Virchow2 + head | 0.876 | 0.861 | 0.851 | 0.902 | 3.8% |
| baseline U-Net (4090) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | baseline per-patient Dice | difference | model pooled Dice | baseline pooled Dice | model clean AUC | baseline clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.795 | 0.722 | +0.074 | 0.827 | 0.760 | 0.970 | 0.940 |
| large section | 22 | 0.765 | 0.718 | +0.046 | 0.796 | 0.751 | 0.954 | 0.924 |
| fold 0 | 9 | 0.831 | 0.762 | +0.069 | 0.854 | 0.781 | 0.977 | 0.943 |
| fold 1 | 9 | 0.770 | 0.719 | +0.050 | 0.812 | 0.765 | 0.951 | 0.918 |
| fold 2 | 10 | 0.799 | 0.719 | +0.080 | 0.819 | 0.743 | 0.976 | 0.957 |
| fold 3 | 10 | 0.709 | 0.636 | +0.073 | 0.725 | 0.666 | 0.934 | 0.888 |
| fold 4 | 9 | 0.803 | 0.773 | +0.029 | 0.861 | 0.833 | 0.977 | 0.958 |

## Paired differences: model minus baseline U-Net (4090) (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | +0.061 [+0.040, +0.084] | 42 / 5 | 5.43e-08 |
| patient-pooled Dice | 47 | +0.057 [+0.035, +0.081] | 41 / 6 | 3.52e-07 |
| sensitivity | 47 | +0.057 [+0.037, +0.077] | 42 / 5 | 4.18e-08 |
| precision | 47 | +0.029 [-0.001, +0.061] | 33 / 14 | 0.00938 |
| false-tumour rate (lower is better) | 47 | -0.005 [-0.019, +0.008] | 22 / 25 | 0.532 |
| clean-tile AUC | 47 | +0.030 [+0.021, +0.042] | 44 / 3 | 8.15e-11 |

Headline wording: **Virchow2 + head ahead**. higher / lower = number of patients on which the model scored above / below baseline.

## Pairwise differences between the finished models (same 47 patients, tile mode)
| difference | measure | patients | difference [95% CI] | higher / lower | Wilcoxon p | wording |
|---|---|---|---|---|---|---|
| UNI + head minus DINOv2-L + head | per-patient Dice | 47 | +0.062 [+0.046, +0.079] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus DINOv2-L + head | patient-pooled Dice | 47 | +0.060 [+0.042, +0.080] | 46 / 1 | 1.08e-11 |  |
| UNI + head minus DINOv2-L + head | clean-tile AUC | 47 | +0.008 [+0.001, +0.017] | 26 / 21 | 0.0542 |  |
| UNI + head minus CONCH + head | per-patient Dice | 47 | +0.012 [+0.004, +0.020] | 37 / 10 | 0.000448 | tie (difference below 0.03) |
| UNI + head minus CONCH + head | patient-pooled Dice | 47 | +0.011 [+0.004, +0.019] | 38 / 9 | 0.000118 |  |
| UNI + head minus CONCH + head | clean-tile AUC | 47 | +0.007 [+0.002, +0.013] | 35 / 12 | 0.000125 |  |
| UNI + head minus Virchow2 + head | per-patient Dice | 47 | +0.012 [+0.002, +0.022] | 27 / 20 | 0.0304 | tie (difference below 0.03) |
| UNI + head minus Virchow2 + head | patient-pooled Dice | 47 | +0.013 [+0.003, +0.024] | 26 / 21 | 0.0409 |  |
| UNI + head minus Virchow2 + head | clean-tile AUC | 47 | +0.005 [+0.001, +0.010] | 29 / 17 | 0.0125 |  |
| UNI + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.090 [+0.071, +0.113] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.082 [+0.062, +0.105] | 46 / 1 | 2.4e-12 |  |
| UNI + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.045 [+0.033, +0.058] | 46 / 1 | 3.93e-11 |  |
| UNI + head minus SegFormer-B2 | per-patient Dice | 47 | +0.045 [+0.032, +0.060] | 43 / 4 | 2.4e-10 | UNI + head ahead |
| UNI + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.045 [+0.031, +0.061] | 42 / 5 | 3.5e-10 |  |
| UNI + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.036 [+0.026, +0.047] | 45 / 2 | 2.7e-13 |  |
| DINOv2-L + head minus CONCH + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 4 / 43 | 1.41e-10 | CONCH + head ahead |
| DINOv2-L + head minus CONCH + head | patient-pooled Dice | 47 | -0.048 [-0.067, -0.032] | 4 / 43 | 1.79e-11 |  |
| DINOv2-L + head minus CONCH + head | clean-tile AUC | 47 | -0.001 [-0.010, +0.007] | 32 / 15 | 0.0798 |  |
| DINOv2-L + head minus Virchow2 + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 2 / 45 | 1.15e-09 | Virchow2 + head ahead |
| DINOv2-L + head minus Virchow2 + head | patient-pooled Dice | 47 | -0.047 [-0.065, -0.031] | 3 / 44 | 1.62e-09 |  |
| DINOv2-L + head minus Virchow2 + head | clean-tile AUC | 47 | -0.003 [-0.010, +0.003] | 23 / 24 | 0.814 |  |
| DINOv2-L + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.029 [+0.013, +0.045] | 35 / 12 | 0.000145 | tie (difference below 0.03) |
| DINOv2-L + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.022 [+0.007, +0.037] | 33 / 14 | 0.000862 |  |
| DINOv2-L + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.026, +0.049] | 44 / 3 | 2.01e-09 |  |
| DINOv2-L + head minus SegFormer-B2 | per-patient Dice | 47 | -0.016 [-0.029, -0.004] | 14 / 33 | 0.00695 | tie (difference below 0.03) |
| DINOv2-L + head minus SegFormer-B2 | patient-pooled Dice | 47 | -0.015 [-0.029, -0.002] | 13 / 34 | 0.016 |  |
| DINOv2-L + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.028 [+0.019, +0.038] | 43 / 4 | 7.77e-09 |  |
| CONCH + head minus Virchow2 + head | per-patient Dice | 47 | -0.000 [-0.010, +0.011] | 17 / 30 | 0.153 | tie (difference below 0.03) |
| CONCH + head minus Virchow2 + head | patient-pooled Dice | 47 | +0.002 [-0.007, +0.012] | 17 / 30 | 0.119 |  |
| CONCH + head minus Virchow2 + head | clean-tile AUC | 47 | -0.002 [-0.006, +0.002] | 16 / 31 | 0.127 |  |
| CONCH + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.078 [+0.059, +0.100] | 47 / 0 | 1.42e-14 | CONCH + head ahead |
| CONCH + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.070 [+0.052, +0.092] | 45 / 2 | 1.42e-13 |  |
| CONCH + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.029, +0.046] | 46 / 1 | 3.55e-13 |  |
| CONCH + head minus SegFormer-B2 | per-patient Dice | 47 | +0.033 [+0.019, +0.049] | 38 / 9 | 4.38e-06 | CONCH + head ahead |
| CONCH + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.033 [+0.019, +0.049] | 35 / 12 | 1.05e-05 |  |
| CONCH + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.029 [+0.021, +0.037] | 44 / 3 | 2.4e-12 |  |
| Virchow2 + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.079 [+0.058, +0.102] | 44 / 3 | 4.49e-10 | Virchow2 + head ahead |
| Virchow2 + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.068 [+0.049, +0.091] | 45 / 2 | 4.49e-10 |  |
| Virchow2 + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.039 [+0.031, +0.049] | 45 / 2 | 4.69e-13 |  |
| Virchow2 + head minus SegFormer-B2 | per-patient Dice | 47 | +0.034 [+0.019, +0.050] | 38 / 9 | 5.65e-06 | Virchow2 + head ahead |
| Virchow2 + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.032 [+0.018, +0.048] | 36 / 11 | 8.21e-06 |  |
| Virchow2 + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.031 [+0.023, +0.039] | 45 / 2 | 9.95e-13 |  |
| nnU-Net-style 2D minus SegFormer-B2 | per-patient Dice | 47 | -0.045 [-0.062, -0.030] | 6 / 41 | 8.32e-08 | SegFormer-B2 ahead |
| nnU-Net-style 2D minus SegFormer-B2 | patient-pooled Dice | 47 | -0.037 [-0.052, -0.023] | 7 / 40 | 6.41e-07 |  |
| nnU-Net-style 2D minus SegFormer-B2 | clean-tile AUC | 47 | -0.009 [-0.015, -0.002] | 13 / 34 | 0.00383 |  |

Wording applies to the per-patient Dice row: a difference below 0.03 is a tie between single runs.

## Training
| run | epochs run | best epoch | best val Dice | train img/s (mean) | machine | weights evaluated |
|---|---|---|---|---|---|---|
| fold 0 | 6 | 2 | 0.8942 | 26 | H100 (GPU 0, shared) | best epoch |
| fold 1 | 7 | 3 | 0.9119 | 26 | H100 (GPU 0, shared) | best epoch |
| fold 2 | 6 | 2 | 0.8894 | 26 | H100 (GPU 0, shared) | best epoch |
| fold 3 | 6 | 2 | 0.9215 | 26 | H100 (GPU 0, shared) | best epoch |
| fold 4 | 6 | 2 | 0.8861 | 27 | H100 (GPU 0, shared) | best epoch |

Settings (config.json, identical to the baseline except the architecture): arch fm_virchow2, epochs max 15, patience 4, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/virchow2/predictions/ - 5 of 5 prediction folders finished, 1.9 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/virchow2_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences)

## Status of the other planned models
- **Virchow2**: finished, see its own summary.
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

