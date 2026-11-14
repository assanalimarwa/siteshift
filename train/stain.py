"""
Macenko stain normalisation: recolour our tiles to SICAPv2's typical H&E, to test whether the
SICAPv2-only models fail on our slides because of staining or because of the tissue itself.

    python stain.py fit          # cv/stain_macenko_sicap.json + cv/stain_examples.png

Fixed before any normalised result was seen (2026-10-21):
- Macenko et al. 2009 with the usual settings: optical density OD = -log((I + 1) / 256),
  pixels with OD < 0.15 in any channel ignored, stain vectors from the 1st / 99th percentile
  angles in the plane of the two main OD eigenvectors, concentrations scaled so that their
  99th percentiles match the reference.
- Reference: one fit on tissue pixels of 300 random SICAPv2 training tiles.
- Our slides: one fit per slide on tissue pixels of up to 150 random tissue tiles of the slide
  (one stain per slide, not per tile, so a tile's own tumour content does not shift its colours).
evaluate.py --stain cv/stain_macenko_sicap.json applies it to every tile (and context window).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = P.INDEX
BETA, ALPHA, PIX_PER_TILE = 0.15, 1.0, 3000


def optical_density(rgb):
    return -np.log((rgb.astype(np.float64) + 1) / 256)


def fit(od):
    """Macenko fit on N x 3 optical densities -> (3 x 2 stain matrix [H, E], 2 max concentrations)."""
    od = od[(od > BETA).all(1)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, [1, 2]]                                  # two largest eigenvectors
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)   # haematoxylin first (more red OD)
    he *= np.sign(he.sum(0, keepdims=True))                  # positive orientation
    he /= np.linalg.norm(he, axis=0, keepdims=True)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    return he, np.percentile(conc, 99, axis=1)


class StainNormalizer:
    """Callable (rgb uint8 H x W x 3, slide name) -> rgb uint8 recoloured to the reference stain."""

    def __init__(self, params):
        ref = params["reference"]
        self.he_ref, self.max_ref = np.array(ref["he"]), np.array(ref["max_conc"])
        self.slides = {s: (np.linalg.pinv(np.array(p["he"])), np.array(p["max_conc"])) for s, p in params["slides"].items()}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def __call__(self, rgb, slide):
        pinv, max_c = self.slides[slide]
        od = optical_density(rgb).reshape(-1, 3)
        conc = (pinv @ od.T) * (self.max_ref / max_c)[:, None]
        out = 256 * np.exp(-(self.he_ref @ conc)) - 1
        return np.clip(out.T.reshape(rgb.shape), 0, 255).astype(np.uint8)


def sample_pixels(paths, rng):
    from prepare import tissue_mask
    chunks = []
    for p in paths:
        rgb = np.array(Image.open(p).convert("RGB"))
        px = rgb[tissue_mask(rgb)]
        if len(px):
            chunks.append(px[rng.choice(len(px), min(PIX_PER_TILE, len(px)), replace=False)])
    return optical_density(np.concatenate(chunks))


def main_fit():
    rng = np.random.default_rng(0)
    sic = pd.read_csv(HERE / "sicap_tiles.csv")
    sic = sic[(sic.split == "train") & (sic.tissue_frac > 0.3)].sample(300, random_state=0)
    he, mx = fit(sample_pixels([Path(str(P.DATA_ROOT / "sicapv2")) / p for p in sic.image], rng))
    params = {"method": "macenko", "beta": BETA, "alpha": ALPHA,
              "reference": {"source": "SICAPv2 train, 300 tiles", "he": he.tolist(), "max_conc": mx.tolist()}, "slides": {}}
    tiles = pd.read_csv(HERE / "tiles.csv")
    for slide, g in tiles[tiles.tissue_frac > 0.3].groupby("slide"):
        g = g.sample(min(150, len(g)), random_state=0)
        he_s, mx_s = fit(sample_pixels([Path(str(P.DATASET47)) / p for p in g.image], rng))
        params["slides"][slide] = {"he": he_s.tolist(), "max_conc": mx_s.tolist(), "tiles": len(g)}
        print(f"{slide.split('.bif')[0]:6s} H {np.round(he_s[:, 0], 2)}  E {np.round(he_s[:, 1], 2)}  max {np.round(mx_s, 2)}", flush=True)
    print(f"SICAPv2 H {np.round(he[:, 0], 2)}  E {np.round(he[:, 1], 2)}  max {np.round(mx, 2)}")
    out = HERE / "cv" / "stain_macenko_sicap.json"
    out.write_text(json.dumps(params, indent=1))

    # picture: for 6 slides, original / normalised, next to SICAPv2 tiles
    norm = StainNormalizer(params)
    picks = ["7.5.bif - Series 0", "44.1.bif - Series 0", "46.2.bif - Series 0", "26.3.bif - Series 0",
             "31.1.bif - Series 0", "30.4.bif - Series 0"]
    S = 220
    sheet = Image.new("RGB", (S * 6, S * 3), "white")
    for c, slide in enumerate(picks):
        rel = tiles[(tiles.slide == slide) & (tiles.tumour_frac > 0.3) & (tiles.tissue_frac > 0.6)].image.iloc[0]
        rgb = np.array(Image.open(Path(str(P.DATASET47)) / rel).convert("RGB"))
        sheet.paste(Image.fromarray(rgb).resize((S, S)), (c * S, 0))
        sheet.paste(Image.fromarray(norm(rgb, slide)).resize((S, S)), (c * S, S))
    for c, rel in enumerate(sic.image.iloc[:6]):
        sheet.paste(Image.open(Path(str(P.DATA_ROOT / "sicapv2")) / rel).convert("RGB").resize((S, S)), (c * S, 2 * S))
    sheet.save(HERE / "cv" / "stain_examples.png")
    print(f"-> {out}, {HERE / 'cv' / 'stain_examples.png'} (rows: ours, ours normalised, SICAPv2)")


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        main_fit()
    else:
        raise SystemExit("usage: python stain.py fit")
