## Pairwise differences between the finished models (same 47 patients, tile mode)
| difference | measure | patients | difference [95% CI] | higher / lower | Wilcoxon p | wording |
|---|---|---|---|---|---|---|
| UNI + head minus DINOv2-L + head | per-patient Dice | 47 | +0.062 [+0.046, +0.079] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus DINOv2-L + head | patient-pooled Dice | 47 | +0.060 [+0.042, +0.080] | 46 / 1 | 1.08e-11 |  |
| UNI + head minus DINOv2-L + head | clean-tile AUC | 47 | +0.008 [+0.001, +0.017] | 26 / 21 | 0.0542 |  |
| UNI + head minus CONCH + head | per-patient Dice | 47 | +0.012 [+0.004, +0.020] | 37 / 10 | 0.000448 | tie (difference below 0.03) |
| UNI + head minus CONCH + head | patient-pooled Dice | 47 | +0.011 [+0.004, +0.019] | 38 / 9 | 0.000118 |  |
| UNI + head minus CONCH + head | clean-tile AUC | 47 | +0.007 [+0.002, +0.013] | 35 / 12 | 0.000125 |  |
| UNI + head minus Virchow2 + head | per-patient Dice | 47 | +0.012 [+0.002, +0.022] | 27 / 20 | 0.0304 | tie (difference below 0.03) |
| UNI + head minus Virchow2 + head | patient-pooled Dice | 47 | +0.013 [+0.003, +0.024] | 26 / 21 | 0.0409 |  |
| UNI + head minus Virchow2 + head | clean-tile AUC | 47 | +0.005 [+0.001, +0.010] | 29 / 17 | 0.0125 |  |
| UNI + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.090 [+0.071, +0.113] | 47 / 0 | 1.42e-14 | UNI + head ahead |
| UNI + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.082 [+0.062, +0.105] | 46 / 1 | 2.4e-12 |  |
| UNI + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.045 [+0.033, +0.058] | 46 / 1 | 3.93e-11 |  |
| UNI + head minus SegFormer-B2 | per-patient Dice | 47 | +0.045 [+0.032, +0.060] | 43 / 4 | 2.4e-10 | UNI + head ahead |
| UNI + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.045 [+0.031, +0.061] | 42 / 5 | 3.5e-10 |  |
| UNI + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.036 [+0.026, +0.047] | 45 / 2 | 2.7e-13 |  |
| DINOv2-L + head minus CONCH + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 4 / 43 | 1.41e-10 | CONCH + head ahead |
| DINOv2-L + head minus CONCH + head | patient-pooled Dice | 47 | -0.048 [-0.067, -0.032] | 4 / 43 | 1.79e-11 |  |
| DINOv2-L + head minus CONCH + head | clean-tile AUC | 47 | -0.001 [-0.010, +0.007] | 32 / 15 | 0.0798 |  |
| DINOv2-L + head minus Virchow2 + head | per-patient Dice | 47 | -0.050 [-0.068, -0.034] | 2 / 45 | 1.15e-09 | Virchow2 + head ahead |
| DINOv2-L + head minus Virchow2 + head | patient-pooled Dice | 47 | -0.047 [-0.065, -0.031] | 3 / 44 | 1.62e-09 |  |
| DINOv2-L + head minus Virchow2 + head | clean-tile AUC | 47 | -0.003 [-0.010, +0.003] | 23 / 24 | 0.814 |  |
| DINOv2-L + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.029 [+0.013, +0.045] | 35 / 12 | 0.000145 | tie (difference below 0.03) |
| DINOv2-L + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.022 [+0.007, +0.037] | 33 / 14 | 0.000862 |  |
| DINOv2-L + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.026, +0.049] | 44 / 3 | 2.01e-09 |  |
| DINOv2-L + head minus SegFormer-B2 | per-patient Dice | 47 | -0.016 [-0.029, -0.004] | 14 / 33 | 0.00695 | tie (difference below 0.03) |
| DINOv2-L + head minus SegFormer-B2 | patient-pooled Dice | 47 | -0.015 [-0.029, -0.002] | 13 / 34 | 0.016 |  |
| DINOv2-L + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.028 [+0.019, +0.038] | 43 / 4 | 7.77e-09 |  |
| CONCH + head minus Virchow2 + head | per-patient Dice | 47 | -0.000 [-0.010, +0.011] | 17 / 30 | 0.153 | tie (difference below 0.03) |
| CONCH + head minus Virchow2 + head | patient-pooled Dice | 47 | +0.002 [-0.007, +0.012] | 17 / 30 | 0.119 |  |
| CONCH + head minus Virchow2 + head | clean-tile AUC | 47 | -0.002 [-0.006, +0.002] | 16 / 31 | 0.127 |  |
| CONCH + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.078 [+0.059, +0.100] | 47 / 0 | 1.42e-14 | CONCH + head ahead |
| CONCH + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.070 [+0.052, +0.092] | 45 / 2 | 1.42e-13 |  |
| CONCH + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.037 [+0.029, +0.046] | 46 / 1 | 3.55e-13 |  |
| CONCH + head minus SegFormer-B2 | per-patient Dice | 47 | +0.033 [+0.019, +0.049] | 38 / 9 | 4.38e-06 | CONCH + head ahead |
| CONCH + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.033 [+0.019, +0.049] | 35 / 12 | 1.05e-05 |  |
| CONCH + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.029 [+0.021, +0.037] | 44 / 3 | 2.4e-12 |  |
| Virchow2 + head minus nnU-Net-style 2D | per-patient Dice | 47 | +0.079 [+0.058, +0.102] | 44 / 3 | 4.49e-10 | Virchow2 + head ahead |
| Virchow2 + head minus nnU-Net-style 2D | patient-pooled Dice | 47 | +0.068 [+0.049, +0.091] | 45 / 2 | 4.49e-10 |  |
| Virchow2 + head minus nnU-Net-style 2D | clean-tile AUC | 47 | +0.039 [+0.031, +0.049] | 45 / 2 | 4.69e-13 |  |
| Virchow2 + head minus SegFormer-B2 | per-patient Dice | 47 | +0.034 [+0.019, +0.050] | 38 / 9 | 5.65e-06 | Virchow2 + head ahead |
| Virchow2 + head minus SegFormer-B2 | patient-pooled Dice | 47 | +0.032 [+0.018, +0.048] | 36 / 11 | 8.21e-06 |  |
| Virchow2 + head minus SegFormer-B2 | clean-tile AUC | 47 | +0.031 [+0.023, +0.039] | 45 / 2 | 9.95e-13 |  |
| nnU-Net-style 2D minus SegFormer-B2 | per-patient Dice | 47 | -0.045 [-0.062, -0.030] | 6 / 41 | 8.32e-08 | SegFormer-B2 ahead |
| nnU-Net-style 2D minus SegFormer-B2 | patient-pooled Dice | 47 | -0.037 [-0.052, -0.023] | 7 / 40 | 6.41e-07 |  |
| nnU-Net-style 2D minus SegFormer-B2 | clean-tile AUC | 47 | -0.009 [-0.015, -0.002] | 13 / 34 | 0.00383 |  |

Wording applies to the per-patient Dice row: a difference below 0.03 is a tie between single runs.

