# SiteShift: A Cross-Site Benchmark for Prostate Tumour Segmentation

Anonymous code release for a paper under review.

## What it is

SiteShift tests whether a prostate tumour segmentation model trained on public data works at a new site. The site data are 47 patients and 194 H&E
slides (89 biopsy slides from 25 patients, 105 large-section slides from 22 patients), cut into 449,623 tiles of 512 px at 0.5 um/px, with
whole-tumour outlines. Models are evaluated with DDA (detection / delineation attribution), which splits 1 - Dice exactly into detection errors
(missed regions, false tumour) and delineation errors (unfilled interior, boundary).

## Repository layout

```
data/              dataset conversion (site tile index, SICAPv2 region rebuilding, PANDA tiling), patient folds (data/cv47/), few-shot draws (data/fewshot/), public-data splits
dataset_analysis/  per-dataset measures (stain, label geometry, domain distance) of the site data, SICAPv2, PANDA-Karolinska and PANDA-Radboud
environment/       pinned requirements (requirements.txt; requirements-smoke.txt for the smoke test only)
eval/              scoring (run_eval.py), DDA (protocol.py) and its independent reference (attribution.py), validation, sensitivity, report scripts
figures/           regenerates the convention figure from results/
results/           per-patient CSVs and summary tables behind the paper (results/cv47, results/analysis, results/datasets), fold-2 leaderboard
tests/             smoke_test.py and e2e_test.py (synthetic data, no download needed)
train/             models, training drivers, one config per training run (train/configs/)
```

Top-level files: `siteshift_paths.py` (every location the scripts read or write; environment variables `SITESHIFT_DATA`, `SITESHIFT_RAW`,
`SITESHIFT_RUNS`, `SITESHIFT_PRED`, `SITESHIFT_RESULTS`, `SITESHIFT_OUT`, `SITESHIFT_INDEX`, defaults are folders inside the repository),
`setup_env.sh`, `LICENSE`.

## Quick start

Install (CPU is enough for the tests and for scoring; for training use the CUDA wheels of the same torch / torchvision versions):

```
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r environment/requirements.txt
```

Tests, from the repository root:

```
python tests/smoke_test.py             # seconds; last line: ALL CHECKS PASSED
python tests/e2e_test.py               # about 1 min on a CPU; last line: ALL STAGES PASSED
```

The smoke test checks that missed + false + fill + boundary = FP + FN holds exactly in pixels on synthetic masks. The end-to-end test runs two synthetic
slides through the dataset layout, the tile index, one training step, evaluation, DDA, `run_eval.py` on PNG masks and the PANDA tiler.

Scoring a folder of predicted masks. Build the tile index once from the site data, then score:

```
python data/prepare.py --data <DATA>/dataset47 --index-only --index data/tiles47.csv
python eval/run_eval.py --pred <submission> --track A --data <DATA>/dataset47 --out scores
```

`--track A` and `--track C` score the 10 patients of fold 2, and `--track B` scores fold 2's 5 large-section patients. The command writes
`scores_per_patient.csv` (per-patient Dice, clean-tile AUC, region recall, the four DDA bins) and `scores_per_slide.csv` (pixel sums), and prints the means.

Mask format (`eval/SUBMISSION_FORMAT.md`): one PNG per tile, `<submission>/<slide>/<tile image name>.png`. For the tile
`dataset47/<slide>/images/<name>.png`, the mask is `<submission>/<slide>/<name>.png`. Masks are 512 x 512 px, 8-bit single channel, 0 = not tumour,
255 = tumour (any value > 0 counts as tumour). Binary masks only; every tile of the scored patients must be present (a missing file is an error).

## Data

* **Site data (47 patients).** Gated under a data-use agreement; link withheld for review. Not in this repository. Expected layout after access:
  `<DATA>/dataset47/<slide>/images/*.png` and `masks/*.png` (0 / 255), 512 px tiles at 0.5 um/px. The patient is the part of the slide name before
  the first dot.
* **Public sources (not redistributed).** Download them yourself and convert them with the scripts of this repository:
  * SICAPv2 (Mendeley Data, doi 10.17632/9xxm58dvs3, CC BY 4.0), into `$SITESHIFT_RAW/SICAPv2_Kaggle`, then `python data/prepare_sicap.py`.
    It rebuilds the overlap regions of the half-overlapping patches and writes the tile index; `data/sicap_split.json` is the region split used.
  * PANDA-Karolinska and PANDA-Radboud (Kaggle competition `prostate-cancer-grade-assessment`; its terms apply), into
    `$SITESHIFT_RAW/prostate-cancer-grade-assessment`, then
    `python data/make_provider_labels.py --panda <folder with train.csv> --out $SITESHIFT_RAW/prostate-cancer-grade-assessment`,
    `python data/prepare_karolinska.py` and `python data/prepare_radboud.py`. `data/karolinska_*_split.json`, `data/radboud_*_split.json` and
    `data/all_public/` are the slide splits used.
* **Weights.** Not distributed. The frozen encoders (UNI, CONCH, Virchow2: gated by their owners; DINOv2-L: `facebook/dinov2-large`) load from the
  Hugging Face cache; the ResNet-34 start is torchvision's ImageNet model. Models trained on the site data are available on request after review,
  because they may leak patient information.

## Tracks

| track | task | training / adaptation data | evaluated on |
|---|---|---|---|
| **A, cross-site transfer** | segment the site's tumours without any site data | anything except the site data (public data, private data, foundation models) | fold 2 (10 patients) |
| **B, large sections** | Track A on large sections | as Track A | fold 2's 5 large-section patients |
| **C, few-shot adaptation** | adapt any starting model with 2, 4 or 8 site patients | fold 2's fixed draws `data/fewshot/fold2_n2.json`, `fold2_n4.json`, `fold2_n8.json` (all adaptation patients are outside fold 2) | fold 2 (10 patients) |
| **D, label-free model selection** (proposed, no baseline) | rank a set of models for the site without site labels | none from the site | Kendall's tau-b between the submitted ranking and the ranking measured on fold 2 |

Fold 2 (`data/cv47/fold2.json`: patients 6, 10, 23, 25, 28, 30, 34, 36, 38, 46; 5 biopsy, 5 large-section) is the test fold for every track. Its labels
(masks and tile-level tumour fractions) are withheld; the result tables in `results/` still list how the paper's models scored on these patients.
It is the fold closest to the whole cohort in specimen mix and baseline Dice (`data/withheld_fold_selection.md`). The draws of the other folds
(`data/fewshot/fold{0,1,3,4}_n*.json`) contain fold-2 patients; they reproduce the paper's five-fold tables and are not part of the benchmark.

## Rules

* No training or model selection on test labels.
* Differences below 0.03 Dice are ties.
* An improvement counts only if its paired 95% bootstrap interval over patients (10,000 resamples, seed 0) excludes zero.
* With 10 test patients, the 95% interval of one model's Dice is about +-0.14, so only large differences separate.

## Baselines

Mean per-patient Dice (mean of each patient's slide Dice), all 47 patients, tile mode (each tile predicted on its own, threshold 0.5). Columns:
trained on SICAPv2 only; trained on all public data (SICAPv2 + PANDA-Karolinska + PANDA-Radboud); adapted with 8 site patients (few-shot, fresh
start); trained on the site (five-fold cross-validation).

| Model | SICAPv2 | All public | 8 patients | Full site |
|---|---|---|---|---|
| U-Net (ImageNet) | 0.577 | 0.569 | 0.636 | 0.720 |
| nnU-Net-style | - | 0.604 | - | 0.702 |
| SegFormer-B2 | - | 0.568 | - | 0.747 |
| UNI (frozen) | 0.364 | 0.605 | 0.756 | 0.793 |
| CONCH (frozen) | 0.318 | - | - | 0.781 |
| Virchow2 (frozen) | 0.512 | - | - | 0.781 |
| DINOv2-L (frozen) | 0.386 | - | - | 0.731 |

Sources: `results/cv47/public_summary.md` (SICAPv2 and all-public U-Net and encoder heads; site U-Net: row "U-Net site CV (baseline)"),
`nnunet_allpublic_summary.md`, `segformer_allpublic_summary.md`, `fewshot_summary.md` ("U-Net from ImageNet + 8 patients", "UNI fresh head + 8
patients"), and `<model>_summary.md` for the other site cross-validations (`nnunet`, `segformer`, `uni`, `conch`, `virchow2`, `dinov2`).

Fold-2 baselines for the tracks (every model on fold 2 alone, with intervals): `results/leaderboard_fold2.md` and `.csv`, made by
`python eval/leaderboard_fold2.py` from the released per-patient results.

## Reproducing the paper

Numbering follows the current draft and may change in the final version. "Gated data" = the result needs the site data (masks, tiles) or the
prediction files made from it; its numbers are in the released tables, but they cannot be recomputed from this repository alone.

| paper item | content | script | input / output in this repository | gated data |
|---|---|---|---|---|
| Tab. 2 | leaderboard (Baselines above) | `eval/report47.py public`, `eval/report47.py fewshot`, `eval/export_model.py <model>` | `results/cv47/public_summary.md`, `fewshot_summary.md`, `<model>_summary.md` | yes |
| Tab. 3 | public models at the site | `eval/report47.py public` | `results/cv47/public_summary.md`, `public_by_patient.csv`, `radboud_row.md`, `nnunet_allpublic_summary.md`, `segformer_allpublic_summary.md` | yes |
| Fig. 5 | scale, sensitivity-precision, gap by specimen | figure drawn outside this repository from the released table | `results/cv47/public_summary.md` (Karolinska 20 to 3,814 slides; sensitivity, precision; biopsies vs large sections) | yes |
| Fig. 4 | DDA bars | `eval/report_protocol47.py` (bins); the drawing script is not released (it also renders tissue panels) | `results/cv47/protocol_by_patient.csv` (`err_missed`, `err_false`, `err_fill`, `err_boundary`), `protocol_summary.md` | yes |
| Tab. 8 | sensitivity to detection threshold and band width | `eval/sensitivity.py run`, `eval/sensitivity.py report` | `results/analysis/sensitivity.md`, `sensitivity_attribution.csv`, `sensitivity_model_bins.csv`, `sensitivity_data/` | yes |
| Fig. 16 | DDA validation | `eval/validation_suite.py` | `results/analysis/protocol_validation.md`, `validation_rows.csv`, `validation_pairs.csv`, `fig_validation.pdf` | yes |
| Tab. 9, Fig. 17 | convention curves | `eval/convention_curves.py`; figure: `python figures/make_convention_figure.py` | `results/analysis/convention_curves.md`, `convention_curves.csv`, `convention_data/`, `fig_convention.pdf` | table: yes; figure: no (regenerates from the CSV) |
| Tab. 10 | threshold analysis | `train/run_threshold47.py`, `eval/threshold47.py report` | `results/cv47/threshold_summary.md`, `threshold_by_patient.csv`, `threshold_curve.csv` | yes |
| Tab. 11 | label-convention control (site outlines dilated / eroded by 50 um; SICAPv2 relabelled by closing at 40 / 80 um) | `train/run_labels47.py`, `eval/labels47.py report`; `data/prepare_sicap_closed.py` | `results/cv47/labels_summary.md`, `labels_by_patient.csv`; SICAPv2 r40 / r80 rows in `public_summary.md`; configs `train/configs/cv47_labels__*.json`, `public__sicap_r40.json`, `public__sicap_r80.json` | yes |
| Tab. 12 | few-shot attribution | `eval/report_protocol47.py` | "Few-shot attribution" in `results/cv47/protocol_summary.md` | yes |
| Tab. 13 | transfer (site U-Nets on public test data) | `eval/report47.py reverse` | `results/cv47/reverse_summary.md` | yes |
| Tab. 14 | encoders | `eval/export_model.py <model>` | `results/cv47/uni_summary.md`, `conch_summary.md`, `virchow2_summary.md`, `dinov2_summary.md`, `pairwise_models.md` | yes |
| Tab. 15 | transfer-gap attribution | `eval/rq6_report.py`; U-Net row: `python eval/rq6_unet_row.py` | `results/rq6_table.md` (= `results/cv47/rq6_table.md`), `results/cv47/protocol_*_by_patient.csv` | nnU-Net-style and SegFormer-B2 rows: yes; U-Net row: no (computed from the released CSVs) |
| Tab. 16 | few-shot | `train/run_fewshot.py`, `eval/report47.py fewshot` | `results/cv47/fewshot_summary.md`, `fewshot_by_patient.csv`; draws `data/fewshot/`; configs `train/configs/fewshot47__*.json` | yes |
| Tab. 5, Figs. 10-11, Tab. 7 | dataset analysis | `dataset_analysis/ds_props.py`, `embed_uni.py`, `ds_distance.py`, `ds_report.py`, `ds_writeup.py`, `ds_link.py` | `results/datasets/`: `comparison_table.tex`, `eligibility_table.tex`, `eligibility.md`, `<dataset>_report.md`, `fig_stain.pdf`, `fig_label_geometry.pdf`, `fig_domain_distance.pdf`, `labels_vs_errors.md`, `data/summary.csv`, `data/distance*.csv` | yes (and the public downloads) |
| - | fold-2 leaderboard baselines | `python eval/leaderboard_fold2.py` | `results/leaderboard_fold2.md`, `.csv` | no |

Training: `train/run_cv.py` (site cross-validation), `train/run_public_only.py` and `train/run_all_public.py` (public-data models),
`train/run_fewshot.py`, `train/run_labels47.py`; the exact settings of each run are in `train/configs/*.json`.

## What cannot be reproduced without the gated data

Every training run, every prediction, every DDA analysis of a model, the threshold, label-convention and few-shot experiments, the validation suite
(it perturbs real annotation masks) and the site-data measures of the dataset analysis (the public-data measures need the public downloads). From this
repository alone, the following run: the smoke and end-to-end tests, the convention figure, the U-Net row of the transfer-gap table and the fold-2
leaderboard; all other numbers can be read from the released tables in `results/`.

## Licence and citation

MIT (see `LICENSE`). Citation: withheld for review.
