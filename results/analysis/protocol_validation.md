# Protocol validation, step 2: perturbation suite on real annotation masks

10 real annotation masks (5 biopsies, 5 large sections), fixed seed 0, windows of up to 4096 x 4096 px (2.05 mm) of the stitched annotation at 0.5 um/px; metric engine: the reference attribution.py (equal to protocol.py, parity.md). Bins are shares of 2 TP + FP + FN, so they add up to 1 - Dice.

## Mask selection (rule and seed)

Slides of the 47-patient set with at least 8 countable regions (>= 2,500 um^2) on the whole slide (region counts of the nnU-Net-style site-CV protocol analysis; they do not depend on the band), a mean tumour tile fraction > 0.02, at least 60 tiles (biopsies) or 600 tiles (large sections), one slide per patient; 5 biopsies and 5 large sections drawn with numpy default_rng(0) (pick_slides in validation_suite.py). Each mask is the 4096 x 4096 px window (2.05 mm) of the stitched annotation holding the most regions (coarse search at 1/8 scale). **Limit:** the annotated regions are large and sparse, so a window holds only 2 to 4 of the slide's regions (even 8192-px windows hold 3 to 6 in the candidate slides). 'Delete 10 / 30 / 50% of the regions' therefore removes max(1, round(f n)) regions of the n in the window; the achieved fractions are in the table below and the deletion rows are best read by their achieved area fraction.

| mask | slide | type | regions in the window |
|---|---|---|---|
| 0 | 28.1 | biopsy | 4 |
| 1 | 46.1 | biopsy | 3 |
| 2 | 43.2 | biopsy | 4 |
| 3 | 16.1 | biopsy | 3 |
| 4 | 14.1 | biopsy | 4 |
| 5 | 24.1 | large | 4 |
| 6 | 20.1 | large | 2 |
| 7 | 19.1 | large | 4 |
| 8 | 33.3 | large | 3 |
| 9 | 30.5 | large | 3 |

Achieved deletions (mean over the masks):

| requested | regions deleted | share of the regions | share of the tumour area |
|---|---|---|---|
| 10% | 1.0 of 3.4 | 31% | 36% |
| 30% | 1.0 of 3.4 | 31% | 36% |
| 50% | 1.9 of 3.4 | 57% | 74% |

**Identity.** 1 - Dice = miss + false + fill + boundary holds exactly (integer pixel counts) in all 150 scored predictions (YES); the identical prediction gives Dice 1.000 and all bins 0.000.

## Which bins each perturbation moves (mean over the masks; shares of the error)

| perturbation | Dice | missed | false tumour | interior | boundary | share in the intended bin(s) | leakage |
|---|---|---|---|---|---|---|---|
| delete 10% | 0.753 | 0.247 | 0.000 | 0.000 | 0.000 | miss: 100% | 0.0% |
| delete 30% | 0.753 | 0.247 | 0.000 | 0.000 | 0.000 | miss: 100% | 0.0% |
| delete 50% | 0.389 | 0.611 | 0.000 | 0.000 | 0.000 | miss: 100% | 0.0% |
| false blobs | 0.944 | 0.000 | 0.056 | 0.000 | 0.000 | false: 100% | 0.0% |
| erode 20 um | 0.865 | 0.002 | 0.000 | 0.000 | 0.133 | bnd: 98% | 1.8% |
| erode 50 um | 0.637 | 0.014 | 0.000 | 0.000 | 0.349 | bnd: 96% | 3.8% |
| erode 100 um | 0.340 | 0.354 | 0.000 | 0.123 | 0.183 | bnd, fill: 46% | 53.6% |
| dilate 20 um | 0.892 | 0.000 | 0.000 | 0.000 | 0.108 | bnd: 100% | 0.0% |
| dilate 50 um | 0.773 | 0.000 | 0.000 | 0.000 | 0.227 | bnd: 100% | 0.0% |
| dilate 100 um | 0.639 | 0.000 | 0.177 | 0.000 | 0.184 | bnd, false: 100% | 0.0% |
| opening 20 um | 0.996 | 0.000 | 0.000 | 0.000 | 0.004 | bnd: 100% | 0.4% |
| opening 40 um | 0.980 | 0.004 | 0.000 | 0.000 | 0.016 | bnd: 79% | 20.8% |
| closing 40 um | 0.988 | 0.000 | 0.000 | 0.000 | 0.012 | bnd: 100% | 0.0% |
| closing 80 um | 0.961 | 0.000 | 0.002 | 0.000 | 0.037 | bnd, false: 100% | 0.0% |

Expected leakage: tiny regions erased by erosion or opening (-> missed), gaps wider than 100 um filled by closing 80 um or dilation 100 um (-> false tumour), deleted regions are replaced one for one (no leakage into other bins except through sub-2,500 um^2 pieces cut by the window border).

### by specimen type (mean Dice and share in the intended bins)

| perturbation | biopsy Dice | biopsy intended | large Dice | large intended |
|---|---|---|---|---|
| delete 10% | 0.740 | 100% | 0.767 | 100% |
| delete 30% | 0.740 | 100% | 0.767 | 100% |
| delete 50% | 0.335 | 100% | 0.442 | 100% |
| false blobs | 0.932 | 100% | 0.956 | 100% |
| erode 20 um | 0.838 | 97% | 0.891 | 100% |
| erode 50 um | 0.580 | 94% | 0.694 | 99% |
| erode 100 um | 0.297 | 36% | 0.383 | 58% |
| dilate 20 um | 0.877 | 100% | 0.908 | 100% |
| dilate 50 um | 0.750 | 100% | 0.795 | 100% |
| dilate 100 um | 0.618 | 100% | 0.660 | 100% |
| opening 20 um | 0.993 | 100% | 0.998 | 100% |
| opening 40 um | 0.964 | 77% | 0.996 | 100% |
| closing 40 um | 0.978 | 100% | 0.998 | 100% |
| closing 80 um | 0.937 | 100% | 0.986 | 100% |

## Equal-Dice pairs: deletion of regions (A) against erosion tuned to the same Dice (B)

Erosion radius found by bisection so that |Dice(B) - Dice(A)| <= 0.002 (the pairs below all satisfy the requested 0.005: YES); 30 of 30 pairs reachable.

| deleted | pairs | Dice A | Dice B | A: detection (miss + false) | A: delineation | B: detection | B: delineation | erosion radius (um) | separated |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 10 | 0.753 | 0.753 | 0.247 | 0.000 | 0.011 | 0.237 | 38 | 10 of 10 |
| 30% | 10 | 0.753 | 0.753 | 0.247 | 0.000 | 0.011 | 0.237 | 38 | 10 of 10 |
| 50% | 10 | 0.389 | 0.388 | 0.611 | 0.000 | 0.225 | 0.387 | 101 | 8 of 10 |

separated = the pair has the same Dice but the deletion error is mostly detection (> 50%) and the erosion error mostly delineation (> 50%).

