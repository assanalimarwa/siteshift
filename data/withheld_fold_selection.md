# Withheld fold: selection

Fold 2 is the withheld leaderboard fold: its labels (masks, tile-level tumour fractions) are not released; it is scored by the organisers. It was chosen as the cv47 fold closest to the whole cohort in specimen mix and in the baseline's per-patient Dice.

Rule used for the choice: for each fold, the sum of four absolute deviations from the cohort value: the biopsy share of the patients, the mean baseline per-patient Dice (all patients), and the mean baseline per-patient Dice among biopsy patients and among large-section patients. The fold with the smallest sum is withheld; a tie would go to fold 2. Baseline = ResNet-34 U-Net, tile mode, per-patient Dice = mean of the patient's slide Dice.

| fold | patients | biopsy / large patients | slides | biopsy / large slides | biopsy share of patients | baseline per-patient Dice | Dice biopsies | Dice large sections | distance to the cohort |
|---|---|---|---|---|---|---|---|---|---|
| **cohort** | 47 | 25 / 22 | 194 | 89 / 105 | 53.2% | 0.720 | 0.722 | 0.718 | 0 |
| 0 | 9 | 5 / 4 | 41 | 20 / 21 | 55.6% | 0.762 | 0.715 | 0.819 | 0.173 |
| 1 | 9 | 5 / 4 | 39 | 18 / 21 | 55.6% | 0.719 | 0.638 | 0.821 | 0.212 |
| 2 | 10 | 5 / 5 | 42 | 20 / 22 | 50.0% | 0.719 | 0.758 | 0.680 | 0.107 |
| 3 | 10 | 5 / 5 | 33 | 13 / 20 | 50.0% | 0.636 | 0.732 | 0.540 | 0.304 |
| 4 | 9 | 5 / 4 | 39 | 18 / 21 | 55.6% | 0.773 | 0.765 | 0.784 | 0.186 |

Folds 1 and 2 tie on the overall baseline Dice (0.719); fold 2 is clearly closer once the two specimen types are looked at separately (fold 1: biopsies 0.638 against 0.722, large sections 0.821 against 0.718). Fold 2 holds patients 6, 10, 23, 25, 28, 30, 34, 36, 38, 46. All result tables of this release cover all 47 patients.
