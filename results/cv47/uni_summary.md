# UNI + head: 47-patient 5-fold patient CV, tile mode

Written 2026-11-05 14:01 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server (GPU 1)**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.

- UNI fold 0: its first 4 epochs ran on the shared GPU 0 before the GPU 1 reset, then resumed from that checkpoint on GPU 1; the config.json machine label of that run is the resumed one.

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| UNI + head | 47 | 0.793 [0.752, 0.830] | 0.139 | 0.838 | 0.826 [0.792, 0.857] | 0.834 [0.792, 0.872] | 0.831 [0.796, 0.865] | 0.076 [0.056, 0.101] | 0.968 [0.956, 0.977] |
| baseline U-Net (4090) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| UNI + head | 0.879 | 0.866 | 0.844 | 0.917 | 3.2% |
| baseline U-Net (4090) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | baseline per-patient Dice | difference | model pooled Dice | baseline pooled Dice | model clean AUC | baseline clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.811 | 0.722 | +0.089 | 0.845 | 0.760 | 0.979 | 0.940 |
| large section | 22 | 0.772 | 0.718 | +0.054 | 0.804 | 0.751 | 0.955 | 0.924 |
| fold 0 | 9 | 0.836 | 0.762 | +0.075 | 0.858 | 0.781 | 0.980 | 0.943 |
| fold 1 | 9 | 0.789 | 0.719 | +0.070 | 0.834 | 0.765 | 0.964 | 0.918 |
| fold 2 | 10 | 0.798 | 0.719 | +0.079 | 0.820 | 0.743 | 0.973 | 0.957 |
| fold 3 | 10 | 0.731 | 0.636 | +0.095 | 0.756 | 0.666 | 0.944 | 0.888 |
| fold 4 | 9 | 0.816 | 0.773 | +0.043 | 0.869 | 0.833 | 0.980 | 0.958 |

## Paired differences: model minus baseline U-Net (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | +0.073 [+0.054, +0.095] | 45 / 2 | 6.35e-12 |
| patient-pooled Dice | 47 | +0.070 [+0.050, +0.094] | 45 / 2 | 7.07e-11 |
| sensitivity | 47 | +0.052 [+0.024, +0.081] | 40 / 7 | 4.1e-07 |
| precision | 47 | +0.049 [+0.022, +0.079] | 37 / 10 | 4.89e-05 |
| false-tumour rate (lower is better) | 47 | -0.020 [-0.034, -0.008] | 15 / 32 | 0.0021 |
| clean-tile AUC | 47 | +0.035 [+0.024, +0.050] | 43 / 3 | 3.78e-08 |

Headline wording: **UNI + head ahead**. higher / lower = number of patients on which the model scored above / below the baseline.

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
| fold 0 | 6 | 2 | 0.8866 | 99 | H100 (GPU 1) |
| fold 1 | 6 | 2 | 0.9161 | 154 | H100 (GPU 1) |
| fold 2 | 10 | 6 | 0.8869 | 154 | H100 (GPU 1) |
| fold 3 | 7 | 3 | 0.9200 | 154 | H100 (GPU 1) |
| fold 4 | 7 | 3 | 0.8766 | 154 | H100 (GPU 1) |

Settings (config.json, identical to the baseline except the architecture): arch fm_uni, epochs max 15, patience 4, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/uni/predictions/ - 5 of 5 prediction folders finished, 1.9 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/uni_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It starts after the nnU-Net site CV and all-public run, and only if the estimated end is before 2026-11-09 08:00 (the queue decides when it gets there; otherwise it is logged as not started).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

