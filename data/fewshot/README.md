# Few-shot draws

`fold<k>_n<n>.json` (k = 0-4, n = 2, 4, 8) are the adaptation draws used in the paper's few-shot experiment (`train/run_fewshot.py`).
Each file is `data/cv47/fold<k>.json` with the training slides cut down to the first n patients of a fixed random order of the
fold's train patients (`patient_order`, numpy `default_rng(k)`; nested: the 2 patients are in the 4, the 4 in the 8). `val` and
`test` are the fold's own.

| fold | n = 2 | n = 4 adds | n = 8 adds |
|---|---|---|---|
| 0 | 8, 46 | 40, 4 | 6, 30, 36, 28 |
| 1 | 9, 2 | 12, 10 | 46, 6, 32, 23 |
| **2 (benchmark, Track C)** | **35, 9** | **37, 4** | **32, 8, 33, 18** |
| 3 | 36, 42 | 19, 37 | 34, 25, 7, 29 |
| 4 | 15, 45 | 2, 23 | 38, 10, 33, 37 |

**Public benchmark: fold 2 only.** Track C adapts on `fold2_n2.json`, `fold2_n4.json` or `fold2_n8.json` (all adaptation patients are
outside fold 2) and is scored on fold 2. The draws of folds 0, 1, 3 and 4 contain fold-2 patients (between them all ten: 6, 10, 23, 25, 28, 30, 34, 36, 38, 46)
as adaptation patients and are **not part of the public benchmark**; they are released only to reproduce the paper's five-fold tables.
