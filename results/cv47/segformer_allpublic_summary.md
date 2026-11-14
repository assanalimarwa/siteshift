# SegFormer-B2, all-public training: one model trained on public data only, tested on all 47 patients, tile mode

Written 2026-11-09 16:03 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.
- **Wording reference**: this run is compared with its own site CV (SegFormer-B2, same architecture, recipe, epochs and seed), not with the baseline U-Net. The baseline row below is context only.
- The two differ by design in the training data: the site-CV fold models train on 35-36 of our patients and are tested on the held-out 9-10 (five models, every patient once); the all-public model trains on SICAPv2 + Karolinska + Radboud (equal tile draws per source, no tile of ours) and is one model tested on all 47 patients.
- Training machines (config.json): all public: H100 (GPU 0, shared).
- Its site CV (SegFormer-B2) trained on: fold 0: H100 (GPU 0, shared); fold 1: H100 (GPU 0, shared); fold 2: H100 (GPU 0, shared); fold 3: H100 (GPU 0, shared); fold 4: H100 (GPU 0, shared). Runs on the shared GPU 0 ran at about a third of the speed of GPU 1; a paused and resumed run continues from its last checkpoint (same settings, same seed).

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B2, all-public training | 47 | 0.568 [0.516, 0.616] | 0.177 | 0.575 | 0.602 [0.551, 0.650] | 0.630 [0.567, 0.693] | 0.649 [0.585, 0.709] | 0.128 [0.102, 0.157] | 0.910 [0.888, 0.929] |
| SegFormer-B2, site CV (same architecture) | 47 | 0.747 [0.697, 0.794] | 0.172 | 0.808 | 0.781 [0.735, 0.823] | 0.816 [0.767, 0.860] | 0.785 [0.733, 0.834] | 0.097 [0.072, 0.124] | 0.932 [0.912, 0.949] |
| baseline U-Net (4090, context only) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| SegFormer-B2, all-public training | 0.619 | 0.609 | 0.537 | 0.730 | 7.9% |
| SegFormer-B2, site CV (same architecture) | 0.858 | 0.841 | 0.827 | 0.891 | 4.2% |
| baseline U-Net (4090, context only) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | its own site CV per-patient Dice | difference | model pooled Dice | its own site CV pooled Dice | model clean AUC | its own site CV clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.625 | 0.763 | -0.137 | 0.653 | 0.797 | 0.944 | 0.943 |
| large section | 22 | 0.502 | 0.730 | -0.228 | 0.545 | 0.762 | 0.871 | 0.920 |
| fold 0 | 9 | 0.610 | 0.807 | -0.196 | 0.669 | 0.834 | 0.927 | 0.946 |
| fold 1 | 9 | 0.591 | 0.742 | -0.152 | 0.616 | 0.791 | 0.912 | 0.914 |
| fold 2 | 10 | 0.558 | 0.740 | -0.182 | 0.594 | 0.761 | 0.892 | 0.947 |
| fold 3 | 10 | 0.502 | 0.668 | -0.166 | 0.520 | 0.686 | 0.896 | 0.890 |
| fold 4 | 9 | 0.585 | 0.790 | -0.205 | 0.623 | 0.843 | 0.928 | 0.965 |

## Paired differences: model minus SegFormer-B2, site CV (same architecture) (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | -0.180 [-0.219, -0.140] | 4 / 43 | 7.07e-11 |
| patient-pooled Dice | 47 | -0.178 [-0.218, -0.139] | 3 / 44 | 3.09e-10 |
| sensitivity | 47 | -0.186 [-0.249, -0.124] | 13 / 34 | 3.16e-06 |
| precision | 47 | -0.136 [-0.171, -0.103] | 6 / 41 | 1.29e-09 |
| false-tumour rate (lower is better) | 47 | +0.031 [+0.006, +0.055] | 36 / 11 | 0.00255 |
| clean-tile AUC | 47 | -0.022 [-0.043, -0.003] | 18 / 29 | 0.0236 |

Headline wording: **SegFormer-B2, site CV (same architecture) ahead**. higher / lower = number of patients on which the model scored above / below its own site CV.

## Training
| run | epochs run | best epoch | best val Dice | train img/s (mean) | machine | weights evaluated |
|---|---|---|---|---|---|---|
| all public | 27 | 19 | 0.8294 | 45 | H100 (GPU 0, shared) | best epoch |

Settings (config.json, identical to the baseline except the architecture): arch segformer, epochs max 40, patience 8, batch 16, lr 0.0003, weight decay 0.01, EMA 0.999, epoch = 60000 tiles, seed 0, recipe default.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/segformer_allpublic/predictions/ - 1 of 1 prediction folders finished, 2.2 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/segformer_allpublic_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences, site-CV values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It runs after SegFormer and UniMed-CLIP in the queue order and has no cutoff gate any more (user, 2026-11-08: finish all the training).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

