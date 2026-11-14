"""
Split the patients into cross-validation folds, so every patient is tested exactly once.

    python make_folds.py                 # 5 folds -> cv/folds.json, cv/fold0.json ... cv/fold4.json, cv/all.json

Patient = the number before the first dot of the slide name (10.1, 10.2, 10.3 -> patient 10);
all slides of a patient stay in one place. Each fold's test patients are scored by a model
trained on the other patients, of which two (one biopsy, one large section) are held out as
val for early stopping. cv/fold<k>.json has the format of split.json, so train.py --split and
evaluate.py --split-file read it unchanged; cv/all.json puts every slide in "test", for models
that never trained on our tiles (e.g. runs/sicap_pretrain).

Slide types: needle-biopsy strips (mean tissue share of the tiles < 0.65: 7, 12, 38, 40, 43, 44,
45, 46) and large sections (10, 13, 26, 30, 31, 34, 35, 36, 37). Each fold's test set gets 1-2
patients of each type; among the assignments that allow, the one whose tile and tumour-pixel
shares are most even across folds is kept (seeded random search). Patient 36 holds a quarter
of all tiles, so the shares cannot be even; per-patient results matter more than pooled ones.

Never overwrites cv/folds.json without --force: the folds define every cross-validated result.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = P.INDEX
BIOPSY_TISSUE = 0.65          # biopsy strips cover less of their tiles than large sections


def split_file(slides, train, val, test):
    pick = lambda ps: sorted(slides[slides.patient.isin(ps)].slide, key=lambda s: [int(n) for n in s.split(".")[:2]])
    return dict(splits=dict(train=pick(train), val=pick(val), test=pick(test)),
                patients=dict(train=sorted(train), val=sorted(val), test=sorted(test)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=HERE / "tiles.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val", type=float, default=0.15, help="target share of a fold's training tiles held out as val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=50_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "cv", help="folder for folds.json, fold<k>.json, all.json")
    a = ap.parse_args()

    out = a.out
    if (out / "folds.json").exists() and not a.force:
        raise SystemExit(f"{out / 'folds.json'} exists; pass --force to replace it (this changes every CV result)")
    out.mkdir(exist_ok=True)

    t = pd.read_csv(a.index)
    t["patient"] = t.slide.str.split(".").str[0].astype(int)
    p = t.groupby("patient").agg(tiles=("image", "size"), tumour=("tumour_frac", "sum"), tissue=("tissue_frac", "mean"))
    p["biopsy"] = p.tissue < BIOPSY_TISSUE
    slides = t.groupby("slide").patient.first().reset_index()
    K = a.folds

    rng = np.random.default_rng(a.seed)
    groups = [p.index[p.biopsy].to_numpy(), p.index[~p.biopsy].to_numpy()]
    best, best_score = None, np.inf
    for _ in range(a.trials):
        fold = {}
        for g in groups:                                         # deal each type round-robin after a shuffle
            order = rng.permutation(g)
            start = rng.integers(K)
            for i, pat in enumerate(order):
                fold[pat] = (start + i) % K
        f = p.index.map(fold).to_numpy()
        tiles = np.bincount(f, weights=p.tiles, minlength=K) / p.tiles.sum()
        tumour = np.bincount(f, weights=p.tumour, minlength=K) / p.tumour.sum()
        score = tiles.std() + tumour.std()
        if score < best_score:
            best, best_score = f.copy(), score
    p["fold"] = best

    folds = []
    for k in range(K):
        test = p.index[p.fold == k].tolist()
        rest = p[p.fold != k]
        total, total_tumour = rest.tiles.sum(), rest.tumour.sum()
        cand = rest[rest.tiles < 0.2 * total]                   # never hold out a patient this large as val
        pairs = itertools.product(cand.index[cand.biopsy], cand.index[~cand.biopsy])
        val = min(pairs, key=lambda vp: abs(rest.tiles[list(vp)].sum() / total - a.val)
                  + abs(rest.tumour[list(vp)].sum() / total_tumour - a.val))
        val = [int(v) for v in val]
        train = [int(x) for x in rest.index if x not in val]
        test = [int(x) for x in test]
        sf = split_file(slides, train, val, test)
        sf.update(fold=k, seed=a.seed)
        (out / f"fold{k}.json").write_text(json.dumps(sf, indent=2))
        folds.append(dict(fold=k, test=test, val=val, train=train,
                          test_tiles=int(p.tiles[test].sum()), test_tumour_share=round(p.tumour[test].sum() / p.tumour.sum(), 3),
                          val_tiles=int(p.tiles[val].sum()), train_tiles=int(p.tiles[train].sum())))
    all_slides = sorted(slides.slide, key=lambda s: [int(n) for n in s.split(".")[:2]])
    (out / "all.json").write_text(json.dumps(dict(splits=dict(train=[], val=[], test=all_slides)), indent=2))
    (out / "folds.json").write_text(json.dumps(dict(
        seed=a.seed, folds=folds, biopsy_patients=sorted(int(x) for x in p.index[p.biopsy]),
        large_section_patients=sorted(int(x) for x in p.index[~p.biopsy])), indent=2))

    for f in folds:
        print(f"fold {f['fold']}: test {f['test']} ({f['test_tiles']} tiles, {f['test_tumour_share']:.0%} of tumour)  "
              f"val {f['val']} ({f['val_tiles']} tiles)  train {len(f['train'])} patients ({f['train_tiles']} tiles)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
