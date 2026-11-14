# Submission format

A submission is a folder of predicted tumour masks, one PNG per tile of the evaluated patients:

```
<submission>/<slide>/<tile image name>.png
```

* `<slide>` and `<tile image name>` are the names of the dataset: for the tile `dataset47/<slide>/images/<name>.png` the mask is `<submission>/<slide>/<name>.png`.
* 512 x 512 pixels at 0.5 um/px (the tile size of the dataset), 8-bit, single channel. 0 = not tumour, 255 = tumour (any value > 0 counts as tumour).
* Binary masks only. The clean-tile AUC of the benchmark is computed from the share of tumour pixels of each tile (the paper's definition of the
  detection score), so soft scores cannot be submitted.
* Every tile of the evaluated split must be present; a missing file is an error, never "no tumour".
* Tile mode: each tile is predicted on its own (no information from the other tiles of the slide is required by the format; what a method uses
  internally is declared by the participant in the track description).

Score a submission:

```
python eval/run_eval.py --pred <submission> --track A --data $SITESHIFT_DATA/dataset47 --out scores
# --track A / C: the 10 patients of fold 2 (the test fold of every track); --track B: fold 2's 5 large-section patients.
# Without --track: every tile of --index (default data/tiles47.csv), or one split of --split-file (e.g. data/cv47/fold0.json --split test).
```

Output: `scores_per_patient.csv` (per-patient pooled Dice, mean-of-slides Dice, clean-tile AUC, region recall, dice_found, the four error bins
missed / false / fill / boundary as shares of 1 - Dice, and their shares of the errors) and `scores_per_slide.csv` (pixel sums per slide).

The scoring code is the code that produced the numbers of the paper (`eval/protocol.py`, checked against the reference `eval/attribution.py` by
`eval/parity.py`); the identity missed + false + fill + boundary = FP + FN holds exactly in pixels (`tests/smoke_test.py`).
`run_eval.py` was checked to reproduce every pixel sum of the saved protocol analysis of one UNI fold-0 slide pair exactly (difference 0).
