# Ranking test, 47 patients: SICAPv2 held-out vs our site

_ranking47.py, 2026-11-12 14:25. Rules in its docstring. SICAPv2 held-out: 19 regions with cancer; site: 47 patients (25 biopsies, 22 large sections). Tile mode, threshold 0.5; 95% bootstrap CIs, 10,000 resamples, seed 0. UniMed-CLIP has no 47-patient site model and is left out (SICAPv2 held-out Dice 0.793)._

## Scores

| method            | SICAPv2 held-out (trained on SICAPv2)   |   rank SICAPv2 | our site, 47 patients (site CV)   |   rank site |   site: biopsies (25) |   site: large sections (22) |
|:------------------|:----------------------------------------|---------------:|:----------------------------------|------------:|----------------------:|----------------------------:|
| U-Net (ResNet-34) | 0.803 [0.745, 0.856]                    |              2 | 0.720 [0.667, 0.768]              |           5 |                 0.722 |                       0.718 |
| UNI               | 0.787 [0.722, 0.846]                    |              4 | 0.793 [0.752, 0.831]              |           1 |                 0.811 |                       0.772 |
| Virchow2          | 0.804 [0.745, 0.857]                    |              1 | 0.781 [0.737, 0.822]              |           2 |                 0.795 |                       0.765 |
| CONCH             | 0.794 [0.725, 0.855]                    |              3 | 0.781 [0.740, 0.819]              |           3 |                 0.798 |                       0.761 |
| DINOv2-L          | 0.785 [0.709, 0.849]                    |              5 | 0.731 [0.679, 0.779]              |           4 |                 0.749 |                       0.71  |

## Separation

**SICAPv2's held-out set separates 0 of 10 pairs; our site separates 8 of 10; the order flips in 5 pairs.**

| pair                         | SICAPv2 difference [CI]   | SICAPv2 separates   | site difference [CI]    | site separates   | order flips   |
|:-----------------------------|:--------------------------|:--------------------|:------------------------|:-----------------|:--------------|
| U-Net (ResNet-34) - UNI      | +0.016 [-0.010, +0.050]   | no                  | -0.073 [-0.095, -0.054] | yes              | yes           |
| U-Net (ResNet-34) - Virchow2 | -0.002 [-0.027, +0.020]   | no                  | -0.061 [-0.084, -0.039] | yes              | no            |
| U-Net (ResNet-34) - CONCH    | +0.008 [-0.028, +0.052]   | no                  | -0.061 [-0.083, -0.042] | yes              | yes           |
| U-Net (ResNet-34) - DINOv2-L | +0.018 [-0.019, +0.068]   | no                  | -0.011 [-0.024, +0.002] | no               | yes           |
| UNI - Virchow2               | -0.017 [-0.049, +0.009]   | no                  | +0.012 [+0.002, +0.022] | yes              | yes           |
| UNI - CONCH                  | -0.007 [-0.035, +0.017]   | no                  | +0.012 [+0.004, +0.020] | yes              | yes           |
| UNI - DINOv2-L               | +0.002 [-0.025, +0.027]   | no                  | +0.062 [+0.047, +0.079] | yes              | no            |
| Virchow2 - CONCH             | +0.010 [-0.009, +0.038]   | no                  | +0.000 [-0.011, +0.010] | no               | no            |
| Virchow2 - DINOv2-L          | +0.020 [-0.008, +0.059]   | no                  | +0.050 [+0.034, +0.068] | yes              | no            |
| CONCH - DINOv2-L             | +0.010 [-0.011, +0.029]   | no                  | +0.050 [+0.034, +0.068] | yes              | no            |

## Agreement

- Kendall's tau-b between the two orders: +0.00 [-0.60, +0.60]
- tau-b with ties for pairs a side cannot tell apart: undefined (separated pairs 0 / 8 of 10: one benchmark ties every pair)
- Best on SICAPv2: Virchow2; best on our site: UNI
