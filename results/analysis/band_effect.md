# Effect of the three fixes on the 5 parity slides (old pipeline vs final pipeline)

Old: square 50 um band, a sub-2,500 um^2 region counted as found when >= 10% marked, inner band = distance to the nearest non-tumour pixel.
Final: Euclidean 50 um band, a sub-2,500 um^2 component can never be detected (its unmarked pixels are missed), inner band = distance to the outline pixels
(the reference attribution.py). Same predictions; tp, fp, fn and the region counts are identical. Change (final minus old) as a share of 2 tp + fp + fn of the slide,
i.e. in units of (1 - Dice) of the slide, in percentage points.

| model | slide | type | missed | interior (fill) | boundary | false tumour |
|---|---|---|---|---|---|---|
| nnU-Net-style site CV | 39.2 | biopsy | +0.00 | +2.80 | -2.81 | +0.01 |
| nnU-Net-style site CV | 9.1 | biopsy | +0.12 | +0.21 | -0.40 | +0.06 |
| nnU-Net-style site CV | 23.3 | biopsy | +0.00 | +0.33 | -2.09 | +1.76 |
| nnU-Net-style site CV | 44.2 | large | +0.00 | +0.26 | -0.70 | +0.43 |
| nnU-Net-style site CV | 44.1 | large | +0.00 | +0.52 | -0.89 | +0.37 |
| nnU-Net-style all-public | 39.2 | biopsy | +0.00 | +0.25 | -0.59 | +0.35 |
| nnU-Net-style all-public | 9.1 | biopsy | +0.12 | +0.21 | -0.37 | +0.04 |
| nnU-Net-style all-public | 23.3 | biopsy | +0.00 | +1.45 | -1.94 | +0.49 |
| nnU-Net-style all-public | 44.2 | large | +0.00 | +0.14 | -0.79 | +0.65 |
| nnU-Net-style all-public | 44.1 | large | +0.00 | +0.22 | -0.86 | +0.64 |

Mean over the slides of a type (points of the error):

| type | missed | interior (fill) | boundary | false tumour |
|---|---|---|---|---|
| biopsy | +0.04 | +0.87 | -1.37 | +0.45 |
| large | +0.00 | +0.29 | -0.81 | +0.52 |
