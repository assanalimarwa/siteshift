# Baseline U-Net (ImageNet ResNet-34), 47-patient 5-fold patient CV - finished 2026-11-02 11:20

Data $SITESHIFT_DATA/dataset47, tiles47.csv, folds cv47/ (194 slides, 449,623 tiles).
Training: train.py defaults (resnet34_v2 settings) + --epoch-tiles 60000 --limit-val 20000; 40 max epochs, patience 8. Total 14.5 h.
Epochs (best): fold0 16 (8), fold1 32 (24), fold2 15 (7), fold3 26 (18), fold4 19 (11).

| mode | pooled Dice | median patient | mean +- sd | IQR | median slide | sensitivity | precision | false-tumour rate |
|---|---|---|---|---|---|---|---|---|
| tile | 0.841 | 0.798 | 0.755 +- 0.166 | 0.696-0.872 | 0.825 | 0.786 | 0.905 | 3.4% |
| ctx256_blend_tta | 0.869 | 0.821 | 0.779 +- 0.155 | 0.717-0.900 | 0.851 | 0.841 | 0.898 | 3.9% |

ctx mode: biopsy 25 pts pooled 0.824 / median 0.794; large 22 pts pooled 0.872 / median 0.844.
Folds (pooled/median): 0 0.885/0.845, 1 0.862/0.794, 2 0.899/0.786, 3 0.798/0.741, 4 0.902/0.868. ctx > tile for 40/47 patients.
Per patient: cv47/baseline_imagenet_by_patient.csv. Weakest: 39 (0.34, sens 0.21), 25 (0.42, 0.6% tumour), 27 (0.46, sens 0.31), 24 (0.47, sens 0.35), 2 (0.49, 2.9% tumour).
