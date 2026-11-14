# Protocol validation, step 3: convention curves on real predictions

Per patient: mean over the patient's slides; table cells: mean over the patients [95% bootstrap CI, 10,000 resamples, seed 0]. Predictions: the protocol exports (tile mode, threshold 0.5). Masks reduced from 0.5 to 4 um/px (8 x 8 block majority) for speed. Closing and dilation are Euclidean, in um; signed boundary distance: + outside the annotation, - inside, median over the boundary pixels of predicted components touching a detected annotated region (>= 2,500 um^2 and >= 10% marked). Sets not on this server (SICAPv2, Karolinska 3,814, 600 mixed) are not included.

Resolution check: |Dice at 4 um/px - native Dice| per slide, UNI + head: mean 0.0001, max 0.0006; CONCH + head: mean 0.0001, max 0.0007; DINOv2-L + head: mean 0.0004, max 0.0028; nnU-Net-style, site CV: mean 0.0003, max 0.0014; SegFormer, site CV: mean 0.0001, max 0.0006; nnU-Net-style, all-public: mean 0.0007, max 0.0026; SegFormer, all-public: mean 0.0006, max 0.0017.

## Dice after closing both masks (r in um)

| model | Dice r=0 | r=20 | r=40 | r=60 | r=80 | gain r=0 -> 80 |
|---|---|---|---|---|---|---|
| UNI + head | 0.793 [0.752, 0.830] | 0.796 [0.755, 0.833] | 0.799 [0.759, 0.836] | 0.801 [0.761, 0.839] | 0.804 [0.763, 0.842] | +0.011 |
| CONCH + head | 0.781 [0.741, 0.818] | 0.784 [0.744, 0.821] | 0.787 [0.747, 0.824] | 0.789 [0.749, 0.827] | 0.791 [0.751, 0.829] | +0.010 |
| DINOv2-L + head | 0.732 [0.680, 0.779] | 0.737 [0.685, 0.784] | 0.739 [0.687, 0.787] | 0.739 [0.686, 0.787] | 0.737 [0.683, 0.786] | +0.005 |
| nnU-Net-style, site CV | 0.703 [0.651, 0.751] | 0.703 [0.651, 0.752] | 0.704 [0.652, 0.753] | 0.706 [0.654, 0.755] | 0.708 [0.656, 0.757] | +0.005 |
| SegFormer, site CV | 0.748 [0.697, 0.794] | 0.748 [0.697, 0.795] | 0.750 [0.699, 0.796] | 0.752 [0.702, 0.799] | 0.755 [0.705, 0.802] | +0.008 |
| U-Net baseline, site CV | 0.720 [0.668, 0.768] | 0.722 [0.670, 0.771] | 0.725 [0.673, 0.774] | 0.728 [0.676, 0.777] | 0.731 [0.678, 0.780] | +0.011 |
| nnU-Net-style, all-public | 0.604 [0.550, 0.654] | 0.633 [0.578, 0.684] | 0.648 [0.592, 0.699] | 0.655 [0.597, 0.707] | 0.658 [0.600, 0.711] | +0.053 |
| SegFormer, all-public | 0.568 [0.516, 0.616] | 0.615 [0.562, 0.663] | 0.644 [0.588, 0.693] | 0.655 [0.598, 0.706] | 0.661 [0.604, 0.712] | +0.092 |

## Normalized surface Dice (tolerance in um)

| model | 20 | 40 | 80 |
|---|---|---|---|
| UNI + head | 0.378 [0.316, 0.439] | 0.490 [0.423, 0.556] | 0.597 [0.530, 0.662] |
| CONCH + head | 0.355 [0.297, 0.412] | 0.466 [0.403, 0.529] | 0.579 [0.514, 0.643] |
| DINOv2-L + head | 0.288 [0.235, 0.343] | 0.379 [0.316, 0.442] | 0.474 [0.406, 0.543] |
| nnU-Net-style, site CV | 0.272 [0.225, 0.320] | 0.368 [0.313, 0.422] | 0.482 [0.422, 0.540] |
| SegFormer, site CV | 0.340 [0.284, 0.397] | 0.445 [0.383, 0.506] | 0.556 [0.492, 0.617] |
| U-Net baseline, site CV | 0.296 [0.246, 0.346] | 0.401 [0.343, 0.459] | 0.518 [0.454, 0.580] |
| nnU-Net-style, all-public | 0.200 [0.155, 0.244] | 0.270 [0.214, 0.325] | 0.362 [0.295, 0.429] |
| SegFormer, all-public | 0.174 [0.134, 0.215] | 0.254 [0.201, 0.307] | 0.365 [0.296, 0.433] |

## Symmetric dilation tolerance (radius in um)

| model | 20 | 40 | 80 |
|---|---|---|---|
| UNI + head | 0.840 [0.802, 0.875] | 0.865 [0.830, 0.898] | 0.892 [0.859, 0.921] |
| CONCH + head | 0.829 [0.790, 0.864] | 0.854 [0.818, 0.887] | 0.881 [0.847, 0.911] |
| DINOv2-L + head | 0.794 [0.744, 0.838] | 0.822 [0.775, 0.864] | 0.851 [0.807, 0.890] |
| nnU-Net-style, site CV | 0.746 [0.694, 0.794] | 0.771 [0.720, 0.818] | 0.803 [0.754, 0.848] |
| SegFormer, site CV | 0.786 [0.737, 0.832] | 0.809 [0.760, 0.854] | 0.836 [0.788, 0.880] |
| U-Net baseline, site CV | 0.763 [0.711, 0.810] | 0.789 [0.738, 0.835] | 0.820 [0.770, 0.865] |
| nnU-Net-style, all-public | 0.698 [0.642, 0.749] | 0.733 [0.676, 0.785] | 0.768 [0.711, 0.819] |
| SegFormer, all-public | 0.685 [0.629, 0.735] | 0.729 [0.672, 0.781] | 0.771 [0.712, 0.822] |

## Median signed boundary distance (um; + outside the annotation; detected regions only)

| model | all patients | biopsies | large sections |
|---|---|---|---|
| UNI + head | -68.375 [-101.325, -39.845] | -4.262 [-8.179, -0.228] | -141.231 [-196.255, -94.252] |
| CONCH + head | -81.738 [-118.332, -49.308] | -6.684 [-11.739, -1.782] | -167.026 [-225.926, -113.165] |
| DINOv2-L + head | -118.517 [-175.253, -69.634] | -5.306 [-19.039, 13.354] | -247.165 [-339.643, -167.445] |
| nnU-Net-style, site CV | -63.797 [-99.730, -31.232] | 7.190 [-6.930, 27.496] | -144.464 [-201.833, -93.385] |
| SegFormer, site CV | -60.930 [-94.404, -32.267] | -0.472 [-7.142, 6.072] | -129.632 [-186.842, -79.171] |
| U-Net baseline, site CV | -84.581 [-126.238, -47.807] | 2.190 [-9.077, 15.288] | -183.183 [-249.108, -125.471] |
| nnU-Net-style, all-public | -163.141 [-235.995, -98.804] | -17.489 [-33.308, -1.108] | -328.656 [-449.541, -222.798] |
| SegFormer, all-public | -187.050 [-270.264, -112.837] | -8.643 [-31.593, 16.777] | -389.786 [-516.741, -276.331] |

## Average symmetric surface distance (ASSD) and 95th-percentile Hausdorff distance (HD95), um (lower is better)

Boundaries = outline pixels of the masks at 4 um/px. ASSD = mean of all boundary-to-boundary distances in both directions; HD95 = the larger of the two directed 95th percentiles. Global: every annotated and predicted boundary of the slide (missed regions and false tumour add large distances); within detected regions: annotated boundary of the detected regions (>= 2,500 um^2 and >= 10% marked) against the boundary of the predicted components that touch them. A slide where one boundary set is empty has no value and is left out of that patient's mean.

| model | ASSD global | ASSD within detected regions | HD95 global | HD95 within detected regions |
|---|---|---|---|---|
| UNI + head | 1095.976 [723.950, 1524.703] | 137.771 [95.395, 185.400] | 4105.176 [3142.967, 5165.842] | 598.044 [422.156, 796.179] |
| CONCH + head | 1097.498 [728.649, 1522.246] | 147.365 [102.011, 198.556] | 4280.990 [3288.913, 5413.674] | 626.348 [447.028, 827.234] |
| DINOv2-L + head | 1533.769 [1068.483, 2071.360] | 190.475 [130.819, 258.291] | 4591.979 [3583.111, 5728.678] | 696.174 [501.157, 916.690] |
| nnU-Net-style, site CV | 1485.496 [1024.559, 2029.118] | 162.929 [118.893, 212.314] | 4981.352 [3941.070, 6135.738] | 689.359 [509.738, 890.365] |
| SegFormer, site CV | 1242.869 [817.815, 1728.299] | 142.703 [100.641, 189.844] | 4571.360 [3543.608, 5720.918] | 641.722 [462.749, 844.463] |
| U-Net baseline, site CV | 1235.413 [824.202, 1722.057] | 169.133 [122.215, 221.005] | 4225.352 [3217.491, 5361.041] | 705.850 [522.985, 914.767] |
| nnU-Net-style, all-public | 1917.670 [1373.914, 2526.683] | 253.224 [178.137, 337.508] | 5395.076 [4247.028, 6636.598] | 804.097 [599.962, 1029.500] |
| SegFormer, all-public | 1646.830 [1149.889, 2219.621] | 273.263 [195.431, 360.487] | 4959.052 [3875.451, 6159.927] | 809.345 [612.213, 1023.652] |

By specimen type (within detected regions):

| model | ASSD biopsies | ASSD large sections | HD95 biopsies | HD95 large sections |
|---|---|---|---|---|
| UNI + head | 33.128 [29.181, 37.292] | 256.684 [191.692, 327.809] | 164.352 [141.395, 190.234] | 1090.875 [824.682, 1386.674] |
| CONCH + head | 36.137 [31.082, 42.037] | 273.760 [203.366, 350.555] | 177.587 [149.826, 215.047] | 1136.302 [859.381, 1439.988] |
| DINOv2-L + head | 50.320 [40.321, 64.619] | 349.742 [252.626, 457.161] | 235.325 [192.255, 286.685] | 1219.866 [906.371, 1558.748] |
| nnU-Net-style, site CV | 57.521 [46.515, 71.541] | 282.711 [213.298, 358.261] | 265.822 [219.135, 319.636] | 1170.651 [890.174, 1480.252] |
| SegFormer, site CV | 42.453 [36.363, 49.133] | 256.624 [190.075, 329.035] | 213.104 [182.662, 246.668] | 1128.788 [841.547, 1444.356] |
| U-Net baseline, site CV | 58.364 [46.296, 72.454] | 295.006 [221.141, 374.390] | 277.732 [217.473, 344.739] | 1192.348 [896.233, 1512.719] |
| nnU-Net-style, all-public | 69.817 [60.147, 80.477] | 461.642 [344.149, 591.271] | 304.497 [246.117, 372.780] | 1371.825 [1060.098, 1698.816] |
| SegFormer, all-public | 80.035 [67.762, 95.166] | 492.840 [376.157, 624.004] | 314.238 [251.201, 388.818] | 1371.968 [1086.323, 1672.648] |

## Signed boundary bias (predicted boundary against the annotation; + = outside the annotation, i.e. over-segmentation; detected regions only)

| model | median signed distance (um) | mean signed distance (um) | share of predicted boundary outside the annotation |
|---|---|---|---|
| UNI + head | -68.375 [-101.325, -39.845] | -109.047 [-160.430, -62.018] | 0.343 [0.302, 0.385] |
| CONCH + head | -81.738 [-118.332, -49.308] | -120.453 [-176.309, -69.590] | 0.332 [0.292, 0.374] |
| DINOv2-L + head | -118.517 [-175.253, -69.634] | -157.434 [-232.014, -91.088] | 0.271 [0.227, 0.317] |
| nnU-Net-style, site CV | -63.797 [-99.730, -31.232] | -102.704 [-159.756, -50.471] | 0.368 [0.324, 0.414] |
| SegFormer, site CV | -60.930 [-94.404, -32.267] | -98.388 [-154.413, -47.838] | 0.381 [0.337, 0.426] |
| U-Net baseline, site CV | -84.581 [-126.238, -47.807] | -122.499 [-188.709, -61.222] | 0.345 [0.298, 0.394] |
| nnU-Net-style, all-public | -163.141 [-235.995, -98.804] | -179.309 [-270.168, -97.873] | 0.277 [0.220, 0.339] |
| SegFormer, all-public | -187.050 [-270.264, -112.837] | -202.403 [-299.463, -114.511] | 0.299 [0.225, 0.381] |

Slides without a value (global ASSD / HD95): UNI + head: 0 of 194; CONCH + head: 0 of 194; DINOv2-L + head: 0 of 194; nnU-Net-style, site CV: 0 of 194; SegFormer, site CV: 0 of 194; U-Net baseline, site CV: 1 of 194; nnU-Net-style, all-public: 0 of 194; SegFormer, all-public: 0 of 194.
