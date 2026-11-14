# Label geometry vs error type at our site

_paper_figs/src/ds_link.py; errors from cv47/protocol_by_patient.csv (47 patients, parity-verified protocol.py, report_protocol47.py); label geometry from the per-dataset analysis (median over units [95% CI])._

"All public combined" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 + 1,800 + 202 = 5,816; the rows below are the single-source models.

## Label geometry of the training data

| dataset | gap fill at 40 um | glass in tumour | regions per window | roughness P^2/(4 pi A) |
|---|---|---|---|---|
| Ours | 1.4 [1.1, 2.6]% | 0.2 [0.1, 0.3]% | 1.05 [1.05, 1.05] | 2.54 [2.40, 2.82] |
| SICAPv2 | 10.3 [7.0, 13.0]% | 0.7 [0.4, 1.4]% | 1.32 [1.20, 1.40] | 3.01 [2.75, 3.31] |
| PANDA-Karolinska | 2.4 [2.1, 2.7]% | 5.3 [3.7, 6.9]% | 1.00 [1.00, 1.00] | 2.38 [2.20, 2.89] |
| PANDA-Radboud | 89.4 [79.6, 97.2]% | 0.2 [0.1, 0.2]% | 6.17 [5.92, 7.00] | 3.97 [3.88, 4.06] |

## Error breakdown at our site, 47 patients, tile mode (contribution to 1 - Dice [95% CI]; share of the mean error in brackets)

| model (trained on) | 1 - Dice | missed | false tumour | incomplete fill | boundary | false: isolated / attached |
|---|---|---|---|---|---|---|
| ours (site CV, reference) | 0.245 [0.200, 0.294] | 0.021 [0.003, 0.046] (8%) | 0.096 [0.064, 0.132] (39%) | 0.067 [0.046, 0.092] (27%) | 0.061 [0.051, 0.073] (25%) | 0.070 / 0.026 |
| SICAPv2-only U-Net | 0.380 [0.321, 0.444] | 0.050 [0.026, 0.079] (13%) | 0.146 [0.101, 0.197] (38%) | 0.119 [0.090, 0.150] (31%) | 0.065 [0.053, 0.079] (17%) | 0.121 / 0.025 |
| PANDA-Karolinska-only U-Net | 0.420 [0.370, 0.476] | 0.005 [0.001, 0.012] (1%) | 0.299 [0.240, 0.364] (71%) | 0.045 [0.030, 0.062] (11%) | 0.071 [0.057, 0.085] (17%) | 0.180 / 0.120 |
| PANDA-Radboud-only U-Net | 0.530 [0.484, 0.578] | 0.051 [0.026, 0.081] (10%) | 0.154 [0.111, 0.200] (29%) | 0.239 [0.205, 0.275] (45%) | 0.087 [0.070, 0.104] (16%) | 0.136 / 0.018 |

## Error breakdown at our site, 47 patients, full mode (contribution to 1 - Dice [95% CI]; share of the mean error in brackets)

| model (trained on) | 1 - Dice | missed | false tumour | incomplete fill | boundary | false: isolated / attached |
|---|---|---|---|---|---|---|
| ours (site CV, reference) | 0.221 [0.178, 0.267] | 0.021 [0.003, 0.045] (9%) | 0.095 [0.066, 0.128] (43%) | 0.048 [0.030, 0.070] (22%) | 0.057 [0.047, 0.068] (26%) | 0.062 / 0.033 |
| SICAPv2-only U-Net | 0.358 [0.298, 0.422] | 0.053 [0.028, 0.082] (15%) | 0.130 [0.088, 0.178] (36%) | 0.112 [0.085, 0.142] (31%) | 0.063 [0.052, 0.076] (18%) | 0.107 / 0.022 |
| PANDA-Karolinska-only U-Net | 0.494 [0.440, 0.550] | 0.001 [0.000, 0.001] (0%) | 0.414 [0.354, 0.477] (84%) | 0.018 [0.011, 0.027] (4%) | 0.062 [0.049, 0.075] (13%) | 0.177 / 0.237 |
| PANDA-Radboud-only U-Net | 0.526 [0.480, 0.572] | 0.067 [0.033, 0.109] (13%) | 0.124 [0.088, 0.163] (24%) | 0.247 [0.210, 0.285] (47%) | 0.088 [0.072, 0.105] (17%) | 0.106 / 0.017 |

## Reading (47 patients, tile mode unless stated)

- Gland-level labels do not by themselves predict a fill-dominated error. SICAPv2 (gland-level: gap fill 10.3 [7.0, 13.0]%) transfers with its error spread over missed regions (13%), false tumour (38%), fill (31%) and boundary (17%).
- Coarse region labels that spill onto glass go with over-painting: Karolinska has almost no missed tumour (1%) and little fill error (11%); false tumour is 71% (tile) / 84% (full) of its error.
- Radboud (the most gland-tight labels) is fill-dominated: fill 45%, false tumour 29% (radboud_prediction.md, Result section).
- Ours (site CV) has the most even mix: missed 8%, false 39%, fill 27%, boundary 25%.
