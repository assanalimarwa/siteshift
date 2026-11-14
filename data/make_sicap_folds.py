"""
5-fold cross-validation over SICAPv2 regions, so every region is a test region exactly once
(reverse-direction experiments, run_reverse.py).

    python make_sicap_folds.py          # cv/sicap_folds.json, cv/sicap_fold0.json ... cv/sicap_fold4.json

Built from prepare_sicap.py's 30 stretches of the tile numbering (cut only where no region of
overlapping tiles crosses; the numbering follows the original file names, so one slide's tiles sit
in one stretch):
  test   fold k: stretches c with c % 5 == k (6 stretches, spread over the whole numbering)
  val    fold k: 3 stretches of fold (k + 1) % 5, used only for early stopping
  guard  regions within --guard tile positions of a test stretch: left out of that fold's training,
         so a slide cut by a stretch border is never on both sides
  train  everything else
cv/sicap_fold<k>.json has the format of split.json (train / val / test region lists), so
train.py --split and evaluate_reverse.py --split-file read it. Never overwritten without --force.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = P.INDEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--guard", type=int, default=150)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = HERE / "cv" / "sicap_folds.json"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; pass --force to replace it")

    t = pd.read_csv(HERE / "sicap_tiles.csv")
    t["pos"] = t.image.str.extract(r"(\d+)\.jpg")[0].astype(int)
    t = t.sort_values("pos").reset_index(drop=True)
    if (t.groupby("slide").chunk.nunique() > 1).any():
        raise SystemExit("a region spans two stretches; stretches must be cut where no region crosses")
    K, n = a.folds, len(t)
    t["fold"] = t.chunk % K
    summary = []
    for k in range(K):
        test = t.fold == k
        val_chunks = sorted(t.chunk[t.fold == (k + 1) % K].unique())[::2]
        val = t.chunk.isin(val_chunks)
        near = np.zeros(n, bool)
        for c in t.chunk[test].unique():
            idx = np.where(t.chunk == c)[0]
            near[max(0, idx.min() - a.guard):idx.min()] = True
            near[idx.max() + 1:min(n, idx.max() + 1 + a.guard)] = True
        guard_regions = set(t.slide[near & ~test])
        guard = t.slide.isin(guard_regions) & ~test
        train = ~test & ~val & ~guard
        split = dict(splits=dict(train=sorted(t.slide[train].unique()), val=sorted(t.slide[val & ~guard].unique()),
                                 test=sorted(t.slide[test].unique()), guard=sorted(guard_regions)),
                     fold=k, test_chunks=sorted(int(c) for c in t.chunk[test].unique()), val_chunks=[int(c) for c in val_chunks])
        (HERE / "cv" / f"sicap_fold{k}.json").write_text(json.dumps(split, indent=1))
        s = {name: dict(tiles=int(m.sum()), regions=int(t.slide[m].nunique()),
                        cancer_regions=int((t[m].groupby("slide").tumour_frac.sum() > 0).sum()))
             for name, m in [("train", train), ("val", val & ~guard), ("test", test), ("guard", guard)]}
        summary.append(dict(fold=k, **s))
        print(f"fold {k}: " + "  ".join(f"{name} {v['tiles']} tiles / {v['regions']} regions ({v['cancer_regions']} with cancer)"
                                        for name, v in s.items()), flush=True)
    tests = [set(json.loads((HERE / "cv" / f"sicap_fold{k}.json").read_text())["splits"]["test"]) for k in range(K)]
    assert sum(len(x) for x in tests) == t.slide.nunique() and len(set().union(*tests)) == t.slide.nunique(), "every region once"
    out.write_text(json.dumps(dict(folds=summary, guard=a.guard), indent=1))
    print(f"every one of {t.slide.nunique()} regions is tested exactly once -> {out}")


if __name__ == "__main__":
    main()
