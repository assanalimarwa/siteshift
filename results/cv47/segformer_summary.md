# SegFormer-B2: 47-patient 5-fold patient CV, tile mode

Written 2026-11-09 03:58 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.
- Training machines (config.json): fold 0: H100 (GPU 0, shared); fold 1: H100 (GPU 0, shared); fold 2: H100 (GPU 0, shared); fold 3: H100 (GPU 0, shared); fold 4: H100 (GPU 0, shared).

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B2 | 47 | 0.747 [0.697, 0.794] | 0.172 | 0.808 | 0.781 [0.735, 0.823] | 0.816 [0.767, 0.860] | 0.785 [0.733, 0.834] | 0.097 [0.072, 0.124] | 0.932 [0.912, 0.949] |
| baseline U-Net (4090) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| SegFormer-B2 | 0.858 | 0.841 | 0.827 | 0.891 | 4.2% |
| baseline U-Net (4090) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | baseline per-patient Dice | difference | model pooled Dice | baseline pooled Dice | model clean AUC | baseline clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.763 | 0.722 | +0.041 | 0.797 | 0.760 | 0.943 | 0.940 |
| large section | 22 | 0.730 | 0.718 | +0.012 | 0.762 | 0.751 | 0.920 | 0.924 |
| fold 0 | 9 | 0.807 | 0.762 | +0.045 | 0.834 | 0.781 | 0.946 | 0.943 |
| fold 1 | 9 | 0.742 | 0.719 | +0.023 | 0.791 | 0.765 | 0.914 | 0.918 |
| fold 2 | 10 | 0.740 | 0.719 | +0.021 | 0.761 | 0.743 | 0.947 | 0.957 |
| fold 3 | 10 | 0.668 | 0.636 | +0.032 | 0.686 | 0.666 | 0.890 | 0.888 |
| fold 4 | 9 | 0.790 | 0.773 | +0.017 | 0.843 | 0.833 | 0.965 | 0.958 |

## Paired differences: model minus baseline U-Net (4090) (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | +0.027 [+0.014, +0.043] | 38 / 9 | 1.41e-05 |
| patient-pooled Dice | 47 | +0.025 [+0.010, +0.041] | 39 / 8 | 2.12e-05 |
| sensitivity | 47 | +0.034 [+0.021, +0.047] | 39 / 8 | 1.73e-06 |
| precision | 47 | +0.004 [-0.017, +0.026] | 18 / 29 | 0.838 |
| false-tumour rate (lower is better) | 47 | +0.000 [-0.012, +0.010] | 30 / 17 | 0.172 |
| clean-tile AUC | 47 | -0.000 [-0.006, +0.006] | 20 / 27 | 0.416 |

Headline wording: **tie (difference below 0.03)**. higher / lower = number of patients on which the model scored above / below baseline.

## Pairwise differences between the finished models (same 47 patients, tile mode)
| difference | measure | patients | difference [95% CI] | higher / lower | Wilcoxon p | wording |
|---|---|---|---|---|---|---|
| UNI + head minus DINOv2-L + head | per-patient Dice | 47 | +0.062 [+0.046, +0.079] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus DINOv2-L + head | patient-pooled Dice | 47 | +0.060 [+0.042, +0.080] | 46 / 1 | 1.08e-11 |  |
| UNI + head minus DINOv2-L + head | clean-tile AUC | 47 | +0.008 [+0.001, +0.017] | 26 / 21 | 0.0542 |  |
| UNI + head minus CONCH + head | per-patient Dice | 47 | +0.012 [+0.004, +0.020] | 37 / 10 | 0.000448 | tie (difference below 0.03) |
| UNI + head minus CONCH + head | patient-pooled Dice | 47 | +0.011 [+0.004, +0.019] | 38 / 9 | 0.000118 |  |
| UNI + head minus CONCH + head | clean-tile AUC | 47 | +0.007 [+0.002, +0.013] | 35 / 12 | 0.000125 |  |
| UNI + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.090 [+0.071, +0.113] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.082 [+0.062, +0.105] | 46 / 1 | 2.4e-12 |  |
| UNI + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.045 [+0.033, +0.058] | 46 / 1 | 3.93e-11 |  |
| UNI + head minus SegFormer-B2 | per-patient Dice | 47 | +0.045 [+0.032, +0.060] | 43 / 4 | 2.4e-10 | UNI + head ahead |
| UNI + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.045 [+0.031, +0.061] | 42 / 5 | 3.5e-10 |  |
| UNI + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.036 [+0.026, +0.047] | 45 / 2 | 2.7e-13 |  |
| DINOv2-L + head minus CONCH + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 4 / 43 | 1.41e-10 | CONCH + head ahead |
| DINOv2-L + head minus CONCH + head | patient-pooled Dice | 47 | -0.048 [-0.067, -0.032] | 4 / 43 | 1.79e-11 |  |
| DINOv2-L + head minus CONCH + head | clean-tile AUC | 47 | -0.001 [-0.010, +0.007] | 32 / 15 | 0.0798 |  |
| DINOv2-L + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.029 [+0.013, +0.045] | 35 / 12 | 0.000145 | tie (difference below 0.03) |
| DINOv2-L + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.022 [+0.007, +0.037] | 33 / 14 | 0.000862 |  |
| DINOv2-L + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.026, +0.049] | 44 / 3 | 2.01e-09 |  |
| DINOv2-L + head minus SegFormer-B2 | per-patient Dice | 47 | -0.016 [-0.029, -0.004] | 14 / 33 | 0.00695 | tie (difference below 0.03) |
| DINOv2-L + head minus SegFormer-B2 | patient-pooled Dice | 47 | -0.015 [-0.029, -0.002] | 13 / 34 | 0.016 |  |
| DINOv2-L + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.028 [+0.019, +0.038] | 43 / 4 | 7.77e-09 |  |
| CONCH + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.078 [+0.059, +0.100] | 47 / 0 | 1.42e-14 | CONCH + head ahead |
| CONCH + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.070 [+0.052, +0.092] | 45 / 2 | 1.42e-13 |  |
| CONCH + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.029, +0.046] | 46 / 1 | 3.55e-13 |  |
| CONCH + head minus SegFormer-B2 | per-patient Dice | 47 | +0.033 [+0.019, +0.049] | 38 / 9 | 4.38e-06 | CONCH + head ahead |
| CONCH + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.033 [+0.019, +0.049] | 35 / 12 | 1.05e-05 |  |
| CONCH + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.029 [+0.021, +0.037] | 44 / 3 | 2.4e-12 |  |
| nnU-Net-style 2D minus SegFormer-B2 | per-patient Dice | 47 | -0.045 [-0.062, -0.030] | 6 / 41 | 8.32e-08 | SegFormer-B2 ahead |
| nnU-Net-style 2D minus SegFormer-B2 | patient-pooled Dice | 47 | -0.037 [-0.052, -0.023] | 7 / 40 | 6.41e-07 |  |
| nnU-Net-style 2D minus SegFormer-B2 | clean-tile AUC | 47 | -0.009 [-0.015, -0.002] | 13 / 34 | 0.00383 |  |

Wording applies to the per-patient Dice row: a difference below 0.03 is a tie between single runs.

## Training
| run | epochs run | best epoch | best val Dice | train img/s (mean) | machine | weights evaluated |
|---|---|---|---|---|---|---|
| fold 0 | 33 | 25 | 0.8882 | 50 | H100 (GPU 0, shared) | best epoch |
| fold 1 | 22 | 14 | 0.8970 | 49 | H100 (GPU 0, shared) | best epoch |
| fold 2 | 21 | 13 | 0.8846 | 51 | H100 (GPU 0, shared) | best epoch |
| fold 3 | 34 | 26 | 0.9037 | 47 | H100 (GPU 0, shared) | best epoch |
| fold 4 | 24 | 16 | 0.8870 | 50 | H100 (GPU 0, shared) | best epoch |

Settings (config.json, identical to the baseline except the architecture): arch segformer, epochs max 40, patience 8, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/segformer/predictions/ - 5 of 5 prediction folders finished, 1.9 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/segformer_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It runs after SegFormer and UniMed-CLIP in the queue order and has no cutoff gate any more (user, 2026-11-08: finish all the training).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

