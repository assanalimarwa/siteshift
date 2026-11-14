"""
Per-dataset analysis (results/datasets): the same measures for every dataset, through one adapter per dataset.

An adapter gives
  units()            one row per analysis unit (slide; SICAPv2: overlap region rebuilt from the patches, the copy has no
                     slide ids) with patient (where known), specimen type and tumour / tissue area from the tile index
  fields(row, kind, n, rng)
                     up to n random 256-um fields of the unit at 0.5 um/px (512 x 512 RGB, tumour mask, labelled-area
                     mask or None); kind "tissue" (tiles with tissue >= 25%) or "tumour" (tiles with tumour >= 10%)
  lowres(row)        the unit's tumour map at 4 um/px for region statistics (None where it has no tumour)
Native scales: ours 0.5; PANDA per slide from its TIFF tags (data/panda_um_per_px.csv: Karolinska 0.452 um/px on 3,263
slides (Hamamatsu) and 0.503 on 2,193 (Aperio), Radboud 0.486 on all); SICAPv2 taken as 0.875
(1.75x, as everywhere else in this project), RINGS see RINGS_UM. Images are resampled to 0.5 um/px with a box filter
when shrunk and bilinear when enlarged; masks by nearest neighbour.
PANDA fields are read from the original TIFFs (not the re-encoded training tiles), so stain, blur and compression
measures see the distributed images.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy import ndimage

from common import D47, FIGS, U, grid_pos, mask_rel

sys.path.insert(0, str(U))
from granularity import close, glass  # noqa: E402
from stain import fit  # noqa: E402

UM, FIELD = 0.5, 512                      # analysis scale, field side in px (256 um)
LR = 4.0                                  # um/px of the whole-unit tumour maps
MIN_REGION_UM2 = 2500                     # region statistics: connected tumour components of at least this area
SPECK_UM2 = 100                           # window geometry: components below this are ignored as specks
BETA = 0.15                               # optical density below which a pixel is empty (glass, lumen, fat)
OUT = FIGS / "datasets"
DATA = OUT / "data"
PANDA = P.PANDA_RAW
RINGS = P.RAW / "rings"                      # excluded from the paper; adapter kept
RINGS_UM = 0.5                            # ASSUMED (20x): no primary source states it (2026-11-04); rerun if it is found


def to_field(rgb, tum, lab, um):
    """Square native crops (side 256 um) -> FIELD px at 0.5 um/px."""
    if abs(um - UM) < 1e-3 and rgb.shape[0] == FIELD:
        return rgb, tum, lab
    flt = Image.BOX if um < UM else Image.BILINEAR
    rgb = np.asarray(Image.fromarray(rgb).resize((FIELD, FIELD), flt))
    near = lambda m: None if m is None else np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((FIELD, FIELD), Image.NEAREST)) > 127
    return rgb, near(tum), near(lab)


def downscale_bool(m, f):
    """Boolean map shrunk by factor f (> 1): a low-res pixel is set if more than half of its area is."""
    h, w = int(m.shape[0] / f), int(m.shape[1] / f)
    if h < 1 or w < 1:
        return np.zeros((max(h, 1), max(w, 1)), bool)
    img = Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BOX)
    return np.asarray(img) > 127


# ---------------------------------------------------------------------------------------------------- adapters
class Ours:
    name, label, um = "ours", "Ours", 0.5

    def __init__(self):
        self.t = pd.read_csv(U / "tiles47.csv")
        self.info = pd.read_csv(U / "cv47" / "dataset_by_patient.csv").set_index("p").type

    def units(self):
        g = self.t.groupby("slide")
        u = pd.DataFrame({"tiles": g.size(), "tissue_mm2": g.tissue_frac.sum() * 0.256 ** 2,
                          "tumour_mm2": g.tumour_frac.sum() * 0.256 ** 2}).reset_index().rename(columns={"slide": "unit"})
        u["patient"] = u.unit.str.split(".").str[0].astype(int)
        u["type"] = u.patient.map(self.info)
        return u

    def _tiles(self, unit, kind):
        t = self.t[self.t.slide == unit]
        return t[t.tissue_frac >= 0.25] if kind == "tissue" else t[t.tumour_frac >= 0.1]

    def fields(self, row, kind, n, rng):
        t = self._tiles(row.unit, kind)
        for rel in t.image.sample(min(n, len(t)), random_state=rng.integers(1 << 31)) if len(t) else []:
            rgb = np.asarray(Image.open(D47 / rel).convert("RGB"))
            yield rgb, np.asarray(Image.open(D47 / mask_rel(rel))) > 127, None, rel

    def lowres(self, row):
        t = self.t[self.t.slide == row.unit]
        pos = np.array([grid_pos(i) for i in t.image])
        c0, r0 = pos.min(0)
        per = int(256 / LR)                                   # 64 px per tile
        H, W = (pos.max(0) - pos.min(0) + 1)[::-1] * per
        m = np.zeros((H, W), bool)
        for rel, (c, r), tf in zip(t.image, pos, t.tumour_frac):
            if tf > 0:
                a = Image.open(D47 / mask_rel(rel)).convert("L").resize((per, per), Image.BOX)
                m[(r - r0) * per:(r - r0 + 1) * per, (c - c0) * per:(c - c0 + 1) * per] = np.asarray(a) > 127
        return m


class Sicap:
    name, label, um = "sicapv2", "SICAPv2", 0.875
    root = P.DATA_ROOT / "sicapv2"

    def __init__(self):
        self.t = pd.read_csv(U / "sicap_tiles.csv")
        self.side = round(256 / self.um)                      # 293 native px

    def units(self):
        g = self.t.groupby("slide")
        tile_mm2 = (512 * self.um / 1000) ** 2 / 4            # patches overlap by half in x and y: ~1/4 of each is new
        u = pd.DataFrame({"tiles": g.size(), "tissue_mm2": g.tissue_frac.sum() * tile_mm2,
                          "tumour_mm2": g.tumour_frac.sum() * tile_mm2}).reset_index().rename(columns={"slide": "unit"})
        u["patient"], u["type"] = np.nan, "biopsy"
        return u

    def fields(self, row, kind, n, rng):
        t = self.t[self.t.slide == row.unit]
        t = t[t.tissue_frac >= 0.25] if kind == "tissue" else t[t.tumour_frac >= 0.1]
        o = (512 - self.side) // 2
        for rel in t.image.sample(min(n, len(t)), random_state=rng.integers(1 << 31)) if len(t) else []:
            rgb = np.asarray(Image.open(self.root / rel).convert("RGB"))[o:o + self.side, o:o + self.side]
            m = np.asarray(Image.open(self.root / mask_rel(rel)))[o:o + self.side, o:o + self.side] > 127
            yield (*to_field(rgb, m, None, self.um), rel)

    def lowres(self, row):
        from fig_conventions import sicap_layout
        t = self.t[self.t.slide == row.unit].reset_index(drop=True)
        if not (t.tumour_frac > 0).any():
            return None
        pos = sicap_layout(t)
        f = LR / self.um                                      # native px per low-res px
        per, step = int(round(512 / f)), 256 / f
        xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        x0, y0 = min(xs), min(ys)
        m = np.zeros((int((max(ys) - y0) / f) + per + 1, int((max(xs) - x0) / f) + per + 1), bool)
        for i, (x, y) in pos.items():
            if t.tumour_frac[i] > 0:
                a = np.asarray(Image.open(self.root / mask_rel(t.image[i])).convert("L").resize((per, per), Image.BOX)) > 127
                yy, xx = int((y - y0) / f), int((x - x0) / f)
                m[yy:yy + per, xx:xx + per] |= a
        self.placed = len(pos) / len(t)
        return m


class Panda:
    def __init__(self, part):
        self.part = part
        if part == "karolinska":
            self.name, self.label = "karolinska", "PANDA-Karolinska"
            self.img, self.msk, self.cancer = PANDA / "Karolinska" / "images", PANDA / "Karolinska" / "masks", (2,)
            self.t = pd.read_csv(U / "karolinska_all_tiles.csv")
        else:
            self.name, self.label = "radboud", "PANDA-Radboud"
            self.img, self.msk, self.cancer = PANDA / "Panda" / "images", PANDA / "Panda" / "masks", (3, 4, 5)
            self.t = pd.read_csv(U / "radboud_all_tiles.csv")
        ums = pd.read_csv(DATA / "panda_um_per_px.csv")
        self.ums = ums[ums.part == part].set_index("slide").um

    def units(self):
        g = self.t.groupby("slide")
        u = pd.DataFrame({"tiles": g.size(), "tissue_frac": g.mask_tissue_frac.sum(), "tumour_frac": g.tumour_frac.sum(),
                          "isup": g.isup.first()}).reset_index().rename(columns={"slide": "unit"})
        u["um"] = u.unit.map(self.ums)
        tile_mm2 = (512 * u.um / 1000) ** 2
        u["tissue_mm2"], u["tumour_mm2"] = u.pop("tissue_frac") * tile_mm2, u.pop("tumour_frac") * tile_mm2
        u["patient"], u["type"] = np.nan, "biopsy"
        return u

    def fields(self, row, kind, n, rng):
        from prepare_karolinska import read_tiles
        t = self.t[self.t.slide == row.unit]
        t = t[t.tissue_frac >= 0.25] if kind == "tissue" else t[t.tumour_frac >= 0.1]
        if not len(t):
            return
        img, msk = self.img / f"{row.unit}.tiff", self.msk / f"{row.unit}_mask.tiff"
        import tifffile
        with tifffile.TiffFile(img) as tf:
            rows_, cols_ = -(-tf.pages[0].imagelength // 512), -(-tf.pages[0].imagewidth // 512)
        for rel in t.image.sample(min(n, len(t)), random_state=rng.integers(1 << 31)):
            c, r = grid_pos(rel)
            if r + 1 >= rows_ or c + 1 >= cols_:
                continue
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            a, b = read_tiles(img, cells), read_tiles(msk, cells)
            quad = lambda d: np.concatenate([np.concatenate([d[(r, c)], d[(r, c + 1)]], 1),
                                             np.concatenate([d[(r + 1, c)], d[(r + 1, c + 1)]], 1)], 0)
            rgb, lab = quad(a), quad(b)
            lab = lab[..., 0] if lab.ndim == 3 else lab
            side = round(256 / row.um)
            rgb, lab = rgb[:side, :side, :3], lab[:side, :side]
            yield (*to_field(np.ascontiguousarray(rgb), np.isin(lab, self.cancer), lab > 0, row.um), rel)

    def lowres(self, row):
        if row.tumour_mm2 <= 0:
            return None
        import tifffile
        with tifffile.TiffFile(self.msk / f"{row.unit}_mask.tiff") as tf:
            lab = tf.series[0].levels[1].asarray()                # 4x downsampled
        lab = lab[..., 0] if lab.ndim == 3 else lab
        return downscale_bool(np.isin(lab, self.cancer), LR / (4 * row.um))


class Rings:
    name, label = "rings", "RINGS"

    def __init__(self):
        if RINGS_UM is None:
            raise SystemExit("RINGS_UM not set")
        self.um = RINGS_UM
        idx = DATA / "rings_patches.csv"
        if not idx.exists():
            rows = []
            for split in ["TRAIN", "TEST"]:
                for p in sorted((RINGS / split / "IMAGES").glob("*.png")):
                    rgb = np.asarray(Image.open(p).convert("RGB").resize((375, 375), Image.BOX))
                    tum = np.asarray(Image.open(RINGS / split / "MANUAL TUMOR" / p.name).convert("L").resize((375, 375), Image.NEAREST)) > 127
                    tis = ~(-np.log((rgb.astype(float) + 1) / 256) < BETA).all(-1)
                    rows.append(dict(split=split, image=p.name, unit="_".join(p.stem.split("_")[:2]),
                                     patient=p.stem.split("_")[0], tissue_frac=tis.mean(), tumour_frac=tum.mean()))
            DATA.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(idx, index=False)
        self.t = pd.read_csv(idx)
        self.side = round(256 / self.um)

    def units(self):
        g = self.t.groupby("unit")
        mm2 = (1500 * self.um / 1000) ** 2
        u = pd.DataFrame({"tiles": g.size(), "tissue_mm2": g.tissue_frac.sum() * mm2, "tumour_mm2": g.tumour_frac.sum() * mm2,
                          "patient": g.patient.first()}).reset_index()
        u["type"] = "not reported"
        return u

    def _load(self, r):
        rgb = np.asarray(Image.open(RINGS / r.split / "IMAGES" / r.image).convert("RGB"))
        return rgb, np.asarray(Image.open(RINGS / r.split / "MANUAL TUMOR" / r.image).convert("L")) > 127

    def fields(self, row, kind, n, rng):
        t = self.t[self.t.unit == row.unit]
        t = t[t.tissue_frac >= 0.25] if kind == "tissue" else t[t.tumour_frac >= 0.1]
        for _, r in t.sample(min(n, len(t)), random_state=rng.integers(1 << 31)).iterrows() if len(t) else []:
            rgb, tum = self._load(r)
            y, x = rng.integers(0, 1500 - self.side + 1, 2)
            yield (*to_field(rgb[y:y + self.side, x:x + self.side], tum[y:y + self.side, x:x + self.side], None, self.um),
                   f"{r.split}/{r.image}@{x},{y}")

    def lowres(self, row):
        """Patches are separate fields of view: one map per patch, laid side by side with a 1-px gap (regions are
        flagged as cut when they touch a patch border)."""
        t = self.t[(self.t.unit == row.unit) & (self.t.tumour_frac > 0)]
        if not len(t):
            return None
        per = int(1500 * self.um / LR)
        maps = [downscale_bool(self._load(r)[1], LR / self.um)[:per, :per] for _, r in t.iterrows()]
        out = np.zeros((per + 2, len(maps) * (per + 1) + 1), bool)
        for i, m in enumerate(maps):
            out[1:1 + m.shape[0], 1 + i * (per + 1):1 + i * (per + 1) + m.shape[1]] = m
        self.patch_px = per
        return out


ADAPTERS = {"ours": Ours, "sicapv2": Sicap, "karolinska": lambda: Panda("karolinska"),
            "radboud": lambda: Panda("radboud"), "rings": Rings}


# ---------------------------------------------------------------------------------------------------- measures
def rgb_to_lab(rgb):
    """sRGB (uint8) -> CIE L*a*b* (D65)."""
    c = rgb.astype(np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


L_BINS, AB_BINS = np.linspace(0, 100, 51), np.linspace(-60, 60, 61)


def perimeter(comp):
    """Outline length in px of a boolean component: pixel-edge count x pi/4 (the average overcount of pixel edges
    for outlines at random orientation), so a disk scores ~1 in roughness() whatever the resolution."""
    p = np.pad(comp, 1)
    return ((p[:, 1:] != p[:, :-1]).sum() + (p[1:] != p[:-1]).sum()) * np.pi / 4


def detectors(rgb, od, tissue):
    """Simple artefact detectors (checked by eye on contact sheets, data/check_<detector>_<dataset>.png).
    pen: green/blue/cyan ink (hue 70-250 deg, saturation > 0.3) or black ink (value < 0.1);
    fold: tissue with mean optical density > 1.0 (two layers of tissue are much darker than one), not ink;
    bubble: round empty area >= 5,000 um^2 inside the field (circularity >= 0.75, not touching the border) with a
    dark rim (mean optical density > 0.6 in the 3-px ring around it)."""
    hsv = rgb_to_hsv(rgb / 255.0)
    hue = hsv[..., 0] * 360
    pen = ((hue >= 70) & (hue <= 250) & (hsv[..., 1] > 0.3) & (hsv[..., 2] < 0.9)) | (hsv[..., 2] < 0.1)
    dark = (od.mean(-1) > 1.0) & tissue & ~pen
    lab, n = ndimage.label(~tissue)
    bubble = 0
    if n:
        areas = np.bincount(lab.ravel())
        for k in np.nonzero(areas >= 5000 / UM ** 2)[0]:
            if k == 0:
                continue
            sl = ndimage.find_objects(lab == k)[0]
            if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == FIELD or sl[1].stop == FIELD:
                continue
            comp = lab == k
            circ = 4 * np.pi * areas[k] / perimeter(comp) ** 2
            if circ < 0.75:
                continue
            ring = ndimage.binary_dilation(comp, iterations=3) & ~comp
            if od[ring].mean() > 0.6:
                bubble += 1
    return pen.mean(), dark.sum() / max(tissue.sum(), 1), bubble


def image_measures(rgb, rng):
    od = -np.log((rgb.astype(np.float64) + 1) / 256)
    tissue = (od >= BETA).any(-1)
    grey = rgb.astype(np.float64) @ [0.299, 0.587, 0.114]
    pen, fold, bubble = detectors(rgb, od, tissue)
    lab = rgb_to_lab(rgb[tissue]) if tissue.any() else np.zeros((0, 3))
    hist = (np.histogram(lab[:, 0], L_BINS)[0], np.histogram(lab[:, 1], AB_BINS)[0], np.histogram(lab[:, 2], AB_BINS)[0])
    sample = od[tissue]
    sample = sample[rng.choice(len(sample), min(2000, len(sample)), replace=False)] if len(sample) else sample
    row = dict(tissue=tissue.mean(), blur=ndimage.laplace(grey).var(), pen=pen, fold=fold, bubbles=bubble,
               L=lab[:, 0].mean() if len(lab) else np.nan, a=lab[:, 1].mean() if len(lab) else np.nan,
               b=lab[:, 2].mean() if len(lab) else np.nan)
    return row, hist, sample


def tumour_measures(rgb, tum, labelled):
    """Label geometry of one field with tumour (the existing granularity measures, at 0.5 um/px):
    empty inside tumour (glass = empty areas >= 2,500 um^2, lumen = smaller), gap fill at 40 um (area a closing with a
    40-um disk would add, share of the labelled area), regions per window (8-connected, >= 100 um^2), outline density."""
    od = -np.log((rgb.astype(np.float64) + 1) / 256)
    empty = (od < BETA).all(-1)
    gl = glass(rgb, UM)
    t = tum
    a = t.sum()
    lab, n = ndimage.label(t, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    outline = (t[:, 1:] != t[:, :-1]).sum() + (t[1:] != t[:-1]).sum()
    return dict(tumour_px=int(a), tumour_share_field=a / t.size, glass_in_tumour=(t & gl).sum() / a,
                lumen_in_tumour=(t & empty & ~gl).sum() / a, gap_fill_40=close(t, 40 / UM).sum() / a - 1,
                regions=int((sizes * UM ** 2 >= SPECK_UM2).sum()),
                outline_per_1000um2=outline * UM / (a * UM ** 2) * 1000,
                labelled_share=np.nan if labelled is None else labelled.mean())


def region_stats(m, cut_rows=None, patch_px=None):
    """Connected tumour components (8-conn) of a 4-um/px map, area >= MIN_REGION_UM2: area, roughness
    perimeter^2 / (4 pi area) (1 = disk), cut = touches the map border (RINGS: a patch border)."""
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        area = comp.sum() * LR ** 2
        if area < MIN_REGION_UM2:
            continue
        per = perimeter(comp) * LR
        if patch_px:
            x0, x1 = sl[1].start - 1, sl[1].stop - 1
            cut = sl[0].start <= 1 or sl[0].stop >= m.shape[0] - 1 or x0 % (patch_px + 1) == 0 or x1 % (patch_px + 1) == 0
        else:
            cut = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == m.shape[0] or sl[1].stop == m.shape[1]
        out.append(dict(area_um2=area, roughness=per ** 2 / (4 * np.pi * area), cut=bool(cut)))
    return out


def macenko(samples):
    """Per-unit Macenko fit on the pooled tissue optical densities of its fields: H and E unit vectors, angle between
    them, and mean H / E concentration (optical density along each vector) over tissue pixels."""
    od = np.concatenate(samples) if samples else np.zeros((0, 3))
    od = od[(od > BETA).all(1)]
    if len(od) < 500:
        return {}
    he, _ = fit(od)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    ang = np.degrees(np.arccos(np.clip(he[:, 0] @ he[:, 1], -1, 1)))
    return dict(H_r=he[0, 0], H_g=he[1, 0], H_b=he[2, 0], E_r=he[0, 1], E_g=he[1, 1], E_b=he[2, 1], HE_angle=ang,
                H_od=np.clip(conc[0], 0, None).mean(), E_od=np.clip(conc[1], 0, None).mean())


TISSUE_COLS = ["tissue", "blur", "pen", "fold", "bubbles", "L", "a", "b"]
TUMOUR_COLS = ["tumour_px", "tumour_share_field", "glass_in_tumour", "lumen_in_tumour", "gap_fill_40", "regions",
               "outline_per_1000um2", "labelled_share"]
FIELD_COLS = ["unit", "kind", "key"] + TISSUE_COLS + TUMOUR_COLS


def load_fields(name):
    """<ds>_fields.csv as one table. Files written before 2026-11-04 22:40 mixed two layouts (a unit whose fields were
    all of one kind was written with only that kind's columns); each row is mapped by its length and kind. Rows of a
    unit that was interrupted and redone are dropped as duplicates."""
    import csv
    rows = []
    with open(DATA / f"{name}_fields.csv", newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if r[:2] == ["unit", "kind"]:
                continue
            if len(r) == len(FIELD_COLS):
                cols = FIELD_COLS
            else:
                cols = ["unit", "kind", "key"] + (TISSUE_COLS if r[1] == "tissue" else TUMOUR_COLS)
            rows.append(dict(zip(cols, r)))
    f = pd.DataFrame(rows, columns=FIELD_COLS)
    for c in TISSUE_COLS + TUMOUR_COLS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f.drop_duplicates(["unit", "kind", "key"]).reset_index(drop=True)


def read_field(name, ad, unit, key):
    """Re-read one field of <name> by its key (as stored in <name>_fields.csv): rgb (512 x 512 at 0.5 um/px), tumour mask,
    raw label (PANDA mask values; None elsewhere)."""
    if name == "ours":
        return (np.asarray(Image.open(D47 / key).convert("RGB")), np.asarray(Image.open(D47 / mask_rel(key))) > 127, None)
    if name == "sicapv2":
        o = (512 - ad.side) // 2
        rgb = np.asarray(Image.open(ad.root / key).convert("RGB"))[o:o + ad.side, o:o + ad.side]
        m = np.asarray(Image.open(ad.root / mask_rel(key)))[o:o + ad.side, o:o + ad.side] > 127
        rgb, m, _ = to_field(rgb, m, None, ad.um)
        return rgb, m, None
    from prepare_karolinska import read_tiles
    c, r = grid_pos(key)
    um = float(ad.ums[unit])
    cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    a, b = read_tiles(ad.img / f"{unit}.tiff", cells), read_tiles(ad.msk / f"{unit}_mask.tiff", cells)
    quad = lambda d: np.concatenate([np.concatenate([d[(r, c)], d[(r, c + 1)]], 1),
                                     np.concatenate([d[(r + 1, c)], d[(r + 1, c + 1)]], 1)], 0)
    rgb, lab = quad(a), quad(b)
    lab = lab[..., 0] if lab.ndim == 3 else lab
    side = round(256 / um)
    rgb, lab = np.ascontiguousarray(rgb[:side, :side, :3]), lab[:side, :side]
    flt = Image.BOX if um < UM else Image.BILINEAR
    rgb = np.asarray(Image.fromarray(rgb).resize((FIELD, FIELD), flt))
    lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize((FIELD, FIELD), Image.NEAREST))
    return rgb, np.isin(lab, ad.cancer), lab
