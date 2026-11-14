# Eligibility of other public prostate datasets for this study

_2026-11-04. Official pages, papers and metadata APIs only; nothing was downloaded or registered for. Criterion: usable
for prostate **tumour segmentation** next to ours, i.e. pixel-level labels that convert to tumour vs non-tumour,
ideally on whole slides or large sections (ours: needle biopsies + large sections, region-level whole-tumour outlines,
0.5 um/px)._

| Dataset | Specimen | Annotation -> tumour mask? | Slides / cores, patients | um/px, scanner | Size | License | Access, approval time | Verdict |
|---|---|---|---|---|---|---|---|---|
| AGGC22 | whole-mount prostatectomy (subsets 1, 3) and needle biopsy (subset 2) | pixel masks Stroma / Normal / G3 / G4 / G5, drawn on **parts** of the tissue only -> yes: G3-G5 tumour, Normal + Stroma non-tumour, unannotated = ignore | S1 105 train + 45 test, S2 37 + 16, S3 144 + 67 images (38 specimens x 6 scanners); 214 patients | ~0.44-0.55 (Akoya 0.50; Olympus, Zeiss, Leica, KFBio, Philips) | not stated | CC BY-NC-SA 4.0 | request form, links within 15 working days | **Eligible**, with an ignore label |
| Gleason2019 | TMA cores (prostatectomy) | pixel Gleason maps by 6 pathologists (majority vote) -> yes: G3-G5 vs benign | 333 (or 331) cores, 231 patients; 244 labelled (test labels withheld) | not stated (40x), Leica SCN400 | not stated | none; rules forbid sharing the data | public download links (browser), no approval | **Eligible with caveats**: TMA only, small |
| Arvaniti TMAs | TMA spots (prostatectomy) | pixel masks benign / G3 / G4 / G5 / unlabelled -> yes: G3-G5 vs benign | 886 spots (~886 patients) on 5 TMAs | 0.23, Hamamatsu NanoZoomer-XR | 2.58 GB | CC0 1.0 | Harvard Dataverse, direct | **Eligible with caveats**: TMA only |
| PESO | prostatectomy | epithelium masks only; tumour only as a benign / cancer label per test region -> no | 102 slides, 102 patients; 160 test regions | 0.48, 3DHistech Pannoramic Flash II 250 | 138.2 GB | CC BY-NC-SA 4.0 | Zenodo, direct | **Not eligible**: no tumour pixel masks |
| DiagSet | needle biopsy | region annotations distributed only as labelled patches (>= 75% one class) -> no masks | A 430 scans (2.6 M patches), B 4,675, C 46; patients not stated | 0.25 (40x), Hamamatsu C12000-22 | not stated (unofficial torrent of A: 378 GB) | none stated | registration, activation < 24 h | **Not eligible**: patch labels only, no license |
| TCGA-PRAD | prostatectomy (FFPE + frozen) | none (clinical Gleason only) -> no | 449 diagnostic + 723 frozen slides, 500 patients | not in metadata (SVS) | 588.5 GB (diagnostic 437.6) | GDC open-access policy (no CC license) | GDC, direct | **Not eligible** for supervised segmentation (unlabelled use only) |

## AGGC22 in detail

| Subset | Content | Images (train / test) | Specimen | Annotation |
|---|---|---|---|---|
| 1 | Akoya scanner | 105 / 45 | whole-mount radical prostatectomy | hand-drawn masks (one of 3 NUH pathologists per slide) |
| 2 | Akoya scanner | 37 / 16 | needle biopsy | hand-drawn masks |
| 3 | same prostatectomies re-scanned on 6 scanners | 144 / 67 (26 / 12 per scanner; Olympus 25, Zeiss 15 / 7) | whole-mount radical prostatectomy | Akoya masks transferred by image registration (registration error possible) |

- **Whole-slide or large-section annotations: large sections yes, exhaustive annotation no.** Subsets 1 and 3 are
  whole-mount sections, but "the ground-truth annotations are not exhaustive and pathologists might not annotate all the
  relevant regions"; only annotated pixels are scored. Annotated area: 22,148 of 112,400 mm^2 of prostatectomy tissue
  (~20%) and 2,223 of 7,723 mm^2 of biopsy tissue (~29%). Some cases have only G3, G4 and Normal annotated; masks may
  overlap.
- Test labels: release not stated (the post-challenge note promises the "complete data set").
- Inconsistency: the paper counts 156 biopsy specimens, subset 2 has 53 images (several biopsies per image possible, not
  stated). Prostatectomies: 150 (S1) + 38 (S3) = 188 vs 187 in the paper.

## Recommendation

- **AGGC22 is the only candidate that matches our setting** (whole-mount large sections and biopsies, ~0.5 um/px,
  pixel-level tumour classes, CC BY-NC-SA). Its labels are partial region annotations, so it would need an ignore label
  (as for Radboud background) and could test large-section transfer, which no current public set covers. Cost: a request
  form with up to 15 working days' wait, unknown size (likely hundreds of GB for whole-mount slides), no confirmed test
  masks. Worth requesting now if it may enter the paper; otherwise cite it as related work.
- **Gleason2019 and the Arvaniti TMAs** convert to tumour masks but are TMA cores (0.6-1 mm spots), neither biopsies nor
  large sections; useful at most as an extra transfer test, not for the main comparison. Arvaniti is the easier of the
  two (2.6 GB, CC0, direct download, 0.23 um/px).
- **PESO, DiagSet and TCGA-PRAD** have no tumour pixel masks: related work only.

## Sources

- AGGC22: [main page](https://aggc22.grand-challenge.org/), [Data](https://aggc22.grand-challenge.org/Data/),
  [Evaluation](https://aggc22.grand-challenge.org/Evaluation/), Huo et al. 2024 ([PMC11082180](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11082180/fullTextXML))
- Gleason2019: [download page](https://gleason2019.grand-challenge.org/Register/), [Rules](https://gleason2019.grand-challenge.org/Rules/),
  Nir et al., JAMA Netw Open ([PMC6484626](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6484626/fullTextXML)),
  [arXiv 2005.04886](https://arxiv.org/abs/2005.04886), [arXiv 2212.12844](https://arxiv.org/abs/2212.12844)
- Arvaniti: [Sci Rep 2018](https://www.nature.com/articles/s41598-018-30535-1), [Dataverse](https://doi.org/10.7910/DVN/OCYCMP)
- PESO: [Zenodo v1.1](https://zenodo.org/records/5137717), [Zenodo 1485967](https://zenodo.org/record/1485967), [paper](https://arxiv.org/abs/1808.05883)
- DiagSet: [Sci Rep 2024](https://www.nature.com/articles/s41598-024-52183-4), [GitHub](https://github.com/michalkoziarski/DiagSet),
  [loader](https://github.com/michalkoziarski/DiagSet/blob/master/diagset-a-container/container.py)
- TCGA-PRAD: [GDC API](https://api.gdc.cancer.gov/files), [GDC data access policies](https://gdc.cancer.gov/access-data/data-access-policies)
