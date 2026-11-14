"""
Numbers behind tab_datasets.tex and fig_stats.pdf (CPU):

    python stats.py      ->  paper_figs/data/ours_slides.csv, ours_regions.csv, public_samples.csv

Ours: per slide tiles, tissue and annotated tumour (from tiles47.csv: tissue_frac, tumour_frac of each 512-px tile);
annotated tumour regions = 8-connected components of the slide's mask stitched at 16 um/px (each 512-px mask tile
reduced to 32 px by area, then > 50%), area in mm^2. Public sets: per sample (SICAPv2: overlap region; PANDA: slide)
tiles, tumour share of labelled tissue (PANDA: tumour_frac / mask_tissue_frac; SICAPv2: tumour_frac / tissue_frac).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import D47, FIGS, U, grid_pos, mask_rel

PER = 32
UM_PX = 0.5 * 512 / PER          # 16 um per px


def slide_regions(args):
    slide, images = args
    pos = np.array([grid_pos(i) for i in images])
    c0, r0 = pos.min(0)
    W, H = (pos.max(0) - pos.min(0) + 1) * PER
    m = np.zeros((H, W), bool)
    for rel, (c, r) in zip(images, pos):
        a = Image.open(D47 / mask_rel(rel)).convert("L").resize((PER, PER), Image.BOX)
        m[(r - r0) * PER:(r - r0 + 1) * PER, (c - c0) * PER:(c - c0 + 1) * PER] = np.asarray(a) > 127
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel())[1:] * (UM_PX / 1000) ** 2
    return slide, areas


def main():
    t = pd.read_csv(U / "tiles47.csv")
    info = pd.read_csv(U / "cv47" / "dataset_by_patient.csv").set_index("p").type
    t["p"] = t.slide.str.split(".").str[0].astype(int)
    g = t.groupby("slide")
    s = pd.DataFrame({"p": g.p.first(), "tiles": g.size(), "tissue_mm2": g.tissue_frac.sum() * (0.256 ** 2),
                      "tumour_mm2": g.tumour_frac.sum() * (0.256 ** 2)})
    s["type"] = s.p.map(info)
    s["tumour_share"] = (s.tumour_mm2 / s.tissue_mm2).clip(upper=1)
    s.to_csv(FIGS / "data" / "ours_slides.csv")
    jobs = [(sl, list(gr.image)) for sl, gr in g]
    with Pool(8) as pool:
        res = pool.map(slide_regions, jobs, chunksize=2)
    reg = pd.DataFrame([dict(slide=sl, area_mm2=a) for sl, areas in res for a in areas])
    reg["p"] = reg.slide.str.split(".").str[0].astype(int)
    reg["type"] = reg.p.map(info)
    reg.to_csv(FIGS / "data" / "ours_regions.csv", index=False)
    print(f"ours: {len(s)} slides, {len(reg)} annotated regions")
    rows = []
    for name, idx, denom in [("SICAPv2", "sicap_tiles.csv", "tissue_frac"), ("PANDA-Karolinska", "karolinska_all_tiles.csv", "mask_tissue_frac"),
                             ("PANDA-Radboud", "radboud_tiles.csv", "mask_tissue_frac"), ("PANDA-Radboud (all 5,060)", "radboud_all_tiles.csv", "mask_tissue_frac")]:
        x = pd.read_csv(U / idx)
        gg = x.groupby("slide")
        rows.append(pd.DataFrame({"dataset": name, "tiles": gg.size(),
                                  "tumour_share": (gg.tumour_frac.sum() / gg[denom].sum().clip(lower=1e-9)).clip(upper=1)}))
    pd.concat(rows).to_csv(FIGS / "data" / "public_samples.csv")
    print("public samples written")


if __name__ == "__main__":
    main()
