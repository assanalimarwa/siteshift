# Parity verdict: our protocol pipeline vs the reference attribution.py

**Verdict (2026-11-08): EXACT, 10 of 10 comparisons.** For every slide and model, tp, fp, fn, the four error bins (missed, false tumour, interior / fill, boundary),
the number of countable regions and the number of detected regions are identical, pixel for pixel, between protocol.py (analyse_slide, tile predictions with 3x3 neighbour
stitching) and the reference attribution.py run on the full stitched slide arrays (missing tiles = glass). The reference's own identity miss + false + fill + boundary = FP + FN
holds exactly on all 10. The reference files were not edited (attribution.py, scenarios.py, part2.py).

Ruling of 2026-11-08 (user): a component below 2,500 um^2 is not a region, can never be detected, and its unmarked pixels are missed (the paper states this).

## Slides and models

5 slides of the 47-patient set (mixed specimen types), 0.5 um/px, tile mode, threshold 0.5: three needle biopsies (39.2, 9.1, 23.3) and two large-section slides (44.2, 44.1),
each for two models: nnU-Net-style site CV (fold models) and nnU-Net-style all-public (the second, public-data model; no Karolinska model on the server).

| model | slide | full-slide array (Mpx) | regions >= 2,500 um^2 | detected | tp + fp + fn | all counts equal |
|---|---|---|---|---|---|---|
| nnU-Net-style site CV | 39.2 | 47 | 5 | 3 | 8,223,115 | yes |
| nnU-Net-style site CV | 9.1 | 173 | 6 | 6 | 17,424,400 | yes |
| nnU-Net-style site CV | 23.3 | 180 | 16 | 15 | 13,976,900 | yes |
| nnU-Net-style site CV | 44.2 | 957 | 6 | 6 | 30,367,455 | yes |
| nnU-Net-style site CV | 44.1 | 1098 | 8 | 7 | 36,073,422 | yes |
| nnU-Net-style all-public | 39.2 | 47 | 5 | 4 | 9,065,607 | yes |
| nnU-Net-style all-public | 9.1 | 173 | 6 | 6 | 17,200,743 | yes |
| nnU-Net-style all-public | 23.3 | 180 | 16 | 15 | 10,759,147 | yes |
| nnU-Net-style all-public | 44.2 | 957 | 6 | 6 | 36,659,532 | yes |
| nnU-Net-style all-public | 44.1 | 1098 | 8 | 8 | 45,763,718 | yes |

## Deviations found and fixed in OUR pipeline (found by the first comparison, after the band had already been made Euclidean)

First comparison (nnU-Net-style site CV, slides 39.2 and 9.1; tp, fp, fn, false tumour and region counts already equal): the missed, interior and boundary bins differed.

| slide | missed (ours - ref) | interior (ours - ref) | boundary (ours - ref) |
|---|---|---|---|
| 39.2 | -361 | +22,732 | -22,371 |
| 9.1 | -37,396 | +5,118 | +32,278 |
| 23.3 | -603 | +4,775 | -4,172 |

1. **Small regions** (sub-2,500 um^2 components). Ours binned their unmarked pixels by the 10% detection rule, so a marked small component counted as found;
   the reference never detects it, its unmarked pixels are missed. Seen in the missed bin: 361 px on 39.2 and 37,396 px on 9.1 (ours lower), moved from interior / boundary. Fixed:
   found regions = detected AND >= 2,500 um^2; everything else is missed.
2. **Inner band** (tumour side of the 50 um band). Ours measured the distance to the nearest non-tumour pixel; the reference measures the distance to the OUTLINE pixels (tumour
   pixels with a non-tumour 8-neighbour), which is about one pixel wider. Seen between interior and boundary: about 22,000 px on 39.2 (interior +22,732 / boundary -22,371 once the
   361 px of item 1 are accounted for). Fixed: outline = tumour pixels with a known non-tumour 8-neighbour (unknown pixels and the canvas border create no outline).

Earlier, before the first comparison (decision of the same day): the band had been a square neighbourhood, now a Euclidean distance in um (see band_effect.md).

## Tile-border case (found by the second comparison)

After fixes 1 and 2 all but one slide matched. Slide 9.1 differed by 3 px (interior +3, boundary -3, of 2.1 million error pixels). Cause: our per-tile canvas extended exactly 100 px
(50 um) beyond the tile, so a tile-edge pixel whose nearest outline pixel lies exactly 100 px away on the canvas border was missing that outline pixel's status (its neighbours
lie beyond the canvas). Fixed: the canvas padding is the band radius + 1 px. Third comparison: exact.

Files: parity.py (the comparison), parity.csv (all counts, 10 rows), parity_v1_square_vs_outline_mismatch.csv and parity_v2_residual_3px.csv (the two earlier runs), parity.log.
