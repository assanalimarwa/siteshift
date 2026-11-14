"""
Image and label properties per dataset (CPU, one process; run at below-normal priority beside the GPU queue):

    python ds_props.py ours sicapv2 karolinska radboud rings [--units N] [--fields N]

Per dataset, up to 100 random units (seed 0; all if fewer) and per unit up to 20 tissue fields and 20 tumour fields
(256 um at 0.5 um/px, ds_lib). Writes to paper_figs/datasets/data/:
  <ds>_units_all.csv   every unit: tiles, tissue and tumour area from the tile index (tumour share per slide / patient)
  <ds>_fields.csv      one row per field: image measures (tissue fields) or label geometry (tumour fields)
  <ds>_units.csv       one row per sampled unit: Macenko stain vectors and mean H / E optical density, region counts
  <ds>_regions.csv     tumour regions (>= 2,500 um^2) of the sampled units at 4 um/px: area, roughness, cut
  <ds>_hist.npz        L*, a*, b* histograms of tissue pixels (summed over the dataset's fields)
Resumable: units already in <ds>_units.csv are skipped.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import argparse
import time

import numpy as np
import pandas as pd

from ds_lib import ADAPTERS, AB_BINS, DATA, FIELD_COLS, L_BINS, image_measures, macenko, region_stats, tumour_measures


def append(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run(name, n_units, n_fields):
    ad = ADAPTERS[name]()
    DATA.mkdir(parents=True, exist_ok=True)
    units = ad.units()
    units.to_csv(DATA / f"{name}_units_all.csv", index=False)
    pick = units.sample(min(n_units, len(units)), random_state=0).reset_index(drop=True)
    up, fp, rp, hp = (DATA / f"{name}_{k}" for k in ["units.csv", "fields.csv", "regions.csv", "hist.npz"])
    done = set(pd.read_csv(up).unit.astype(str)) if up.exists() else set()
    hist = dict(np.load(hp)) if hp.exists() and done else {"L": np.zeros(len(L_BINS) - 1), "a": np.zeros(len(AB_BINS) - 1),
                                                           "b": np.zeros(len(AB_BINS) - 1)}
    t0 = time.time()
    for i, row in pick.iterrows():
        if str(row.unit) in done:
            continue
        rng = np.random.default_rng(i)
        fields, samples = [], []
        for rgb, tum, lab, key in ad.fields(row, "tissue", n_fields, rng):
            r, h, s = image_measures(rgb, rng)
            fields.append(dict(unit=row.unit, kind="tissue", key=key, **r))
            for k, v in zip("Lab", h):
                hist[k] += v
            samples.append(s)
        for rgb, tum, lab, key in ad.fields(row, "tumour", n_fields, rng):
            if tum.sum() * 0.25 >= 100:
                fields.append(dict(unit=row.unit, kind="tumour", key=key, **tumour_measures(rgb, tum, lab)))
        m = ad.lowres(row) if row.tumour_mm2 > 0 else None
        regions = region_stats(m, patch_px=getattr(ad, "patch_px", None)) if m is not None else []
        append(fp, [{c: f.get(c) for c in FIELD_COLS} for f in fields])      # one fixed layout
        append(rp, [dict(unit=row.unit, **r) for r in regions])
        np.savez(hp, **hist)
        append(up, [dict(unit=row.unit, patient=row.patient, type=row.type, tissue_mm2=row.tissue_mm2,
                         tumour_mm2=row.tumour_mm2, n_tissue_fields=sum(f["kind"] == "tissue" for f in fields),
                         n_tumour_fields=sum(f["kind"] == "tumour" for f in fields), regions=len(regions),
                         regions_uncut=sum(not r["cut"] for r in regions), placed=getattr(ad, "placed", np.nan),
                         **macenko(samples))])
        print(f"[{name}] {i + 1}/{len(pick)} {row.unit}: {len(fields)} fields, {len(regions)} regions "
              f"({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=list(ADAPTERS))
    ap.add_argument("--units", type=int, default=100)
    ap.add_argument("--fields", type=int, default=20)
    a = ap.parse_args()
    for d in a.datasets:
        run(d, a.units, a.fields)


if __name__ == "__main__":
    main()
