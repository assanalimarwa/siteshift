# nnU-Net-style 2D, all-public training: one model trained on public data only, tested on all 47 patients, tile mode

Written 2026-11-08 07:15 by export_model.py. Metric definitions: see the top of export_model.py (the same as the local reports; the baseline is scored from its own per_slide.csv / per_tile.csv).

## Machine and comparability
- The encoders and the other models here were trained and evaluated on the **H100 server**; the baseline U-Net was trained and evaluated on the **4090** (local, Python 3.10). Fold 0 trained on both machines with identical settings differs by about **+0.016** (H100 higher). Differences below **0.03** between single runs are therefore worded as ties.
- One run (seed 0) per model and fold; no repeat runs, so the confidence intervals cover patient sampling only, not training noise.
- **Wording reference**: this run is compared with its own site CV (nnU-Net-style 2D, same architecture, recipe, epochs and seed), not with the baseline U-Net. The baseline row below is context only.
- The two differ by design in the training data: the site-CV fold models train on 35-36 of our patients and are tested on the held-out 9-10 (five models, every patient once); the all-public model trains on SICAPv2 + Karolinska + Radboud (equal tile draws per source, no tile of ours) and is one model tested on all 47 patients.
- Training machines (config.json): all public: H100 (GPU 0, shared).
- Its site CV (nnU-Net-style 2D) trained on: fold 0: H100 (GPU 1); fold 1: H100 (GPU 1); fold 2: H100 (GPU 1); fold 3: H100 (GPU 0, shared); fold 4: H100 (GPU 0, shared). Runs on the shared GPU 0 ran at about a third of the speed of GPU 1; a paused and resumed run continues from its last checkpoint (same settings, same seed).

## Main result (headline: per-patient Dice = mean of the patient's slide Dice)
| model | patients | **per-patient Dice** (mean of slide Dice) [95% CI] | SD | median | patient-pooled Dice [95% CI] (secondary) | sensitivity | precision | false-tumour rate | clean-tile AUC |
|---|---|---|---|---|---|---|---|---|---|
| nnU-Net-style 2D, all-public training | 47 | 0.604 [0.550, 0.654] | 0.186 | 0.627 | 0.642 [0.589, 0.689] | 0.673 [0.615, 0.731] | 0.667 [0.604, 0.725] | 0.151 [0.118, 0.189] | 0.911 [0.892, 0.928] |
| nnU-Net-style 2D, site CV (same architecture) | 47 | 0.702 [0.650, 0.751] | 0.178 | 0.762 | 0.744 [0.694, 0.789] | 0.791 [0.740, 0.836] | 0.754 [0.696, 0.808] | 0.108 [0.083, 0.137] | 0.923 [0.903, 0.941] |
| baseline U-Net (4090, context only) | 47 | 0.720 [0.668, 0.768] | 0.178 | 0.775 | 0.755 [0.706, 0.800] | 0.783 [0.728, 0.831] | 0.782 [0.728, 0.831] | 0.097 [0.071, 0.125] | 0.932 [0.911, 0.951] |

Sensitivity, precision, false-tumour rate and clean-tile AUC: mean over patients of each patient's pooled counts (clean-tile AUC: tiles with >= 25% tissue, tumour tiles >= 50% under the outline vs tumour-free tiles).

### All pixels pooled (as in cv47/baseline_summary.md)
| model | pooled Dice (all pixels) | median slide Dice | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|
| nnU-Net-style 2D, all-public training | 0.686 | 0.670 | 0.628 | 0.756 | 8.6% |
| nnU-Net-style 2D, site CV (same architecture) | 0.830 | 0.804 | 0.784 | 0.880 | 4.4% |
| baseline U-Net (4090, context only) | 0.841 | 0.825 | 0.786 | 0.905 | 3.4% |

## Per specimen type and per fold (means over the patients of the group)
| group | patients | model per-patient Dice | its own site CV per-patient Dice | difference | model pooled Dice | its own site CV pooled Dice | model clean AUC | its own site CV clean AUC |
|---|---|---|---|---|---|---|---|---|
| biopsy | 25 | 0.659 | 0.713 | -0.055 | 0.691 | 0.755 | 0.946 | 0.933 |
| large section | 22 | 0.541 | 0.690 | -0.149 | 0.586 | 0.731 | 0.871 | 0.913 |
| fold 0 | 9 | 0.647 | 0.747 | -0.100 | 0.712 | 0.782 | 0.932 | 0.936 |
| fold 1 | 9 | 0.615 | 0.723 | -0.109 | 0.646 | 0.771 | 0.916 | 0.905 |
| fold 2 | 10 | 0.589 | 0.706 | -0.116 | 0.618 | 0.740 | 0.916 | 0.946 |
| fold 3 | 10 | 0.518 | 0.593 | -0.074 | 0.541 | 0.623 | 0.868 | 0.879 |
| fold 4 | 9 | 0.660 | 0.756 | -0.096 | 0.706 | 0.817 | 0.927 | 0.952 |

## Paired differences: model minus nnU-Net-style 2D, site CV (same architecture) (same patients)
| measure | patients | difference [95% CI] | higher / lower | Wilcoxon p |
|---|---|---|---|---|
| per-patient Dice (headline) | 47 | -0.099 [-0.132, -0.065] | 4 / 43 | 8.58e-09 |
| patient-pooled Dice | 47 | -0.102 [-0.135, -0.068] | 8 / 39 | 4.56e-08 |
| sensitivity | 47 | -0.118 [-0.176, -0.062] | 14 / 33 | 0.000374 |
| precision | 47 | -0.087 [-0.114, -0.060] | 8 / 39 | 1.61e-07 |
| false-tumour rate (lower is better) | 47 | +0.043 [+0.018, +0.069] | 35 / 12 | 0.000448 |
| clean-tile AUC | 47 | -0.012 [-0.030, +0.004] | 20 / 27 | 0.147 |

Headline wording: **nnU-Net-style 2D, site CV (same architecture) ahead**. higher / lower = number of patients on which the model scored above / below its own site CV.

## Training
| run | epochs run | best epoch | best val Dice | train img/s (mean) | machine | weights evaluated |
|---|---|---|---|---|---|---|
| all public | 250 | 240 | 0.8143 | 34 | H100 (GPU 0, shared) | last epoch (no early stopping) |

Settings (config.json, identical to the baseline except the architecture): arch nnunet, epochs max 250, patience 0, batch 12, lr 0.01, weight decay 3e-05, EMA 0.0, epoch = 3000 tiles, seed 0, recipe nnunet.
Versions: python 3.12.3, torch 2.11.0+cu128, cuda 12.8, cudnn 91900, gpu NVIDIA H100, torchvision 0.26.0+cu128, numpy 2.2.6, pandas 2.3.3, pillow 12.3.0, scipy 1.15.3, timm 1.0.30, transformers 5.20.1, huggingface_hub 2.2.0

## Files
- predictions (protocol format, tile mode, threshold 0.5): export/nnunet_allpublic/predictions/ - 1 of 1 prediction folders finished, 2.3 GB so far (one folder per fold; per slide a .npz with image, pred, truth, tissue as packed masks, plus done.json)
- cv47/nnunet_allpublic_by_patient.csv: one row per patient (fold, type, per-patient Dice, pooled Dice, sensitivity, precision, false-tumour rate, clean-tile AUC, baseline values and differences, site-CV values and differences)

## Status of the other planned models
- **Virchow2**: not started yet. It runs after SegFormer and UniMed-CLIP in the queue order and has no cutoff gate any more (user, 2026-11-08: finish all the training).
- **UniMed-CLIP**: not run on the 47 patients (the model was cancelled; the paper reports it on the 17-patient pilot only).

