"""
How precisely the PANDA Radboud masks outline cancer, and whether they run onto glass, measured like
granularity.py next to ours, Karolinska and SICAPv2 (user question 2026-10-27):

    python radboud_labels.py      # cv/radboud_labels.md, cv/radboud_label_examples.png, cv/radboud_slides.png (local only)

Radboud (PANDA, Panda/ folder): 5,060 slides with masks, ~0.486 um/px (TIFF tags); labels 0 background or
unknown, 1 stroma, 2 benign epithelium, 3-5 Gleason 3-5 (cancer = 3-5). Karolinska: 0 background, 1 benign
tissue, 2 cancer.

1. Windows (as granularity.py, native labels only): 256 x 256 um at 1 um/px. Radboud: 320 slides with
   ISUP >= 1 (seed 0); per slide up to 7 blocks of 2 x 2 TIFF tiles (1,024 px) whose level-2 mask has
   cancer and whose level-2 image is >= 25% tissue; window centred on the block. Ours, Karolinska, SICAPv2:
   the samples of granularity.sample(). Added to granularity's measures:
     glass inside tumour   share of tumour-labelled pixels on glass (granularity.glass: empty areas
                           >= 2,500 um^2 and debris specks inside them)
     lumen inside tumour   the other empty pixels inside tumour (lumens, small gaps)
2. Whole label at ~2 um/px: share of the cancer label on glass / on any empty pixel. Radboud: the same
   320 slides, Karolinska: 320 test slides with ISUP >= 1 (seed 0), both at TIFF level 1 (4x), with all
   labels on glass and tissue left unlabelled; ours and SICAPv2: every tile with tumour, box-resampled to
   2 um/px (tiles cut glass areas at their border, so their glass share is a slight underestimate).
3. Pictures: cv/radboud_label_examples.png (12 Radboud windows, 3 Karolinska, 3 ours; 512 px at ~0.5 um/px,
   plain and with the labels) and cv/radboud_slides.png (whole slides at level 2).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cv_summary import bootstrap_ci
from data import load_mask, mask_path
from granularity import PROBES, SOURCES, W, close, empty_pixels, geometry, glass, per_unit, sample, window
from prepare_karolinska import T, read_tiles

HERE = P.INDEX
PANDA = Path(str(P.PANDA_RAW))
RAD, KAR = PANDA / "Panda", PANDA / "Karolinska"
SLIDES, BLOCKS = 320, 7
CANCER = {"Radboud": [3, 4, 5], "Karolinska": [2]}
COLOURS = {"Radboud": {1: (80, 140, 255), 2: (30, 190, 30), 3: (255, 210, 0), 4: (255, 120, 0), 5: (230, 0, 0)},
           "Karolinska": {1: (30, 190, 30), 2: (230, 0, 0)}, "ours": {1: (230, 0, 0)}}


def um_per_px(path):
    with tifffile.TiffFile(path) as tf:
        pg = tf.pages[0]
        num, den = pg.tags["XResolution"].value
        unit = pg.tags["ResolutionUnit"].value
    return (1e4 if unit == tifffile.RESUNIT.CENTIMETER else 25400) * den / num


def window_row(source, unit, image, rgb, mask, um):
    """granularity.measure at radius 0, plus glass and lumen inside the tumour label."""
    t = window(mask, um, Image.NEAREST)
    if not t.any():
        return None
    e = empty_pixels(window(rgb, um, Image.BOX))
    g = window(glass(rgb, um), um, Image.NEAREST)
    outline, regions = geometry(t)
    a = np.array(regions, float).reshape(-1, 3)
    cut = a[:, 1].astype(bool)
    return dict(source=source, unit=unit, image=image, tumour=int(t.sum()), empty_in_tumour=int((t & e).sum()),
                glass_in_tumour=int((t & g).sum()), lumen_in_tumour=int((t & e & ~g).sum()),
                outline=outline, regions=len(regions), regions_cut=int(cut.sum()),
                median_area_uncut=float(np.median(a[~cut, 0])) if (~cut).any() else np.nan,
                median_solidity=float(np.nanmedian(a[:, 2])) if np.isfinite(a[:, 2]).any() else np.nan,
                **{f"closed_{p}": int(close(t, p).sum()) for p in PROBES})


def tile_windows(job):
    source, data, um, image, unit, _ = job
    rgb = np.array(Image.open(data / image).convert("RGB"))
    row = window_row(source, unit, image, rgb, load_mask(data / mask_path(image)).astype(bool), um)
    return [row] if row else []


def read_block(img_p, msk_p, r, c):
    cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    im, mk = read_tiles(img_p, cells), read_tiles(msk_p, cells)
    join = lambda d, k: np.concatenate([np.concatenate([d[cells[0]], d[cells[1]]], 1),
                                        np.concatenate([d[cells[2]], d[cells[3]]], 1)], 0)[..., k]
    return join(im, slice(0, 3)), join(mk, 0)


def radboud_blocks(sid, seed):
    """Top-left TIFF tile (row, col) of up to BLOCKS 2 x 2 blocks with cancer and >= 25% tissue."""
    rgb2 = tifffile.imread(RAD / "images" / f"{sid}.tiff", level=2)[..., :3]
    m2 = tifffile.imread(RAD / "masks" / f"{sid}_mask.tiff", level=2)[..., 0]
    with tifffile.TiffFile(RAD / "images" / f"{sid}.tiff") as tf:
        H, Wd = tf.pages[0].imagelength, tf.pages[0].imagewidth
    s = 2 * T // 16                                                # a block is 64 px at level 2
    h = min(m2.shape[0], rgb2.shape[0], H // T * T // 16) // s
    w = min(m2.shape[1], rgb2.shape[1], Wd // T * T // 16) // s
    blk = lambda a: a[:h * s, :w * s].reshape(h, s, w, s).mean((1, 3))
    cand = np.argwhere((blk(np.isin(m2, CANCER["Radboud"])) > 0) & (blk(~empty_pixels(rgb2)) >= 0.25))
    pick = cand[np.random.default_rng(seed).permutation(len(cand))[:BLOCKS]]
    return [(2 * br, 2 * bc) for br, bc in pick]


def radboud_windows(job):
    sid, seed = job
    img_p, msk_p = RAD / "images" / f"{sid}.tiff", RAD / "masks" / f"{sid}_mask.tiff"
    um = um_per_px(img_p)
    rows = []
    for r, c in radboud_blocks(sid, seed):
        rgb, lab = read_block(img_p, msk_p, r, c)
        row = window_row("Radboud", sid, f"{sid}|{r}|{c}", rgb, np.isin(lab, CANCER["Radboud"]), um)
        if row:
            tissue = ~empty_pixels(rgb)
            row.update({f"label_{k}": int((lab == k).sum()) for k in range(6)},
                       **{f"tissue_label_{k}": int((tissue & (lab == k)).sum()) for k in range(6)})
            rows.append(row)
    return rows


def slide_row(job):
    """Cancer label and all labels against glass at TIFF level 1 (4x)."""
    source, sid, img_p, msk_p = job
    with tifffile.TiffFile(img_p) as tf:
        W0 = tf.pages[0].imagewidth
    rgb = tifffile.imread(img_p, level=1)[..., :3]
    lab = tifffile.imread(msk_p, level=1)[..., 0]
    um = um_per_px(img_p) * W0 / rgb.shape[1]
    h, w = min(rgb.shape[0], lab.shape[0]), min(rgb.shape[1], lab.shape[1])
    rgb, lab = rgb[:h, :w], lab[:h, :w]
    e, g = empty_pixels(rgb), glass(rgb, um)
    ca, an = np.isin(lab, CANCER[source]), lab > 0
    return dict(source=source, unit=sid, um=um, cancer=int(ca.sum()), cancer_glass=int((ca & g).sum()),
                cancer_empty=int((ca & e).sum()), labelled=int(an.sum()), labelled_glass=int((an & g).sum()),
                tissue=int((~e).sum()), tissue_unlabelled=int((~e & ~an).sum()))


def tile_glass(job):
    """The same for one exported tile with tumour, box-resampled to 2 um/px."""
    source, data, um, image, unit = job
    rgb = Image.open(data / image).convert("RGB")
    n = round(rgb.size[0] * um / 2.0)
    rgb = np.array(rgb.resize((n, n), Image.BOX))
    m = np.array(Image.fromarray(load_mask(data / mask_path(image)).astype(np.uint8) * 255).resize((n, n), Image.NEAREST)) > 127
    e, g = empty_pixels(rgb), glass(rgb, 2.0)
    return dict(source=source, unit=unit, um=2.0, cancer=int(m.sum()), cancer_glass=int((m & g).sum()), cancer_empty=int((m & e).sum()))


def overlay(rgb, lab, colours, alpha=0.4):
    o = rgb.astype(np.float64)
    for k, col in colours.items():
        sel = lab == k
        o[sel] = (1 - alpha) * o[sel] + alpha * np.array(col)
    o = o.astype(np.uint8)
    cancer = lab >= (3 if len(colours) == 5 else max(colours))
    edge = np.zeros_like(cancer)
    edge[:, 1:] |= cancer[:, 1:] != cancer[:, :-1]
    edge[1:] |= cancer[1:] != cancer[:-1]
    o[edge] = 0
    return o


def centre(a, n=512):
    o0, o1 = (a.shape[0] - n) // 2, (a.shape[1] - n) // 2
    return a[o0:o0 + n, o1:o1 + n]


def labelled(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(255, 255, 255))
    d.text((4, 2), text, fill=(0, 0, 0))
    return np.array(im)


def example_figure(w, isup):
    rng = np.random.default_rng(0)
    gap = lambda h, wd: np.full((h, wd, 3), 255, np.uint8)
    panels = []
    rad = w[w.source == "Radboud"].drop_duplicates("unit")
    for image in rad.image.iloc[rng.permutation(len(rad))[:12]]:
        sid, r, c = image.split("|")
        rgb, lab = read_block(RAD / "images" / f"{sid}.tiff", RAD / "masks" / f"{sid}_mask.tiff", int(r), int(c))
        rgb, lab = centre(rgb), centre(lab)
        panels.append((labelled(rgb, f"Radboud ISUP {isup.get(sid, '?')} {sid[:8]}"), overlay(rgb, lab, COLOURS["Radboud"])))
    for src, colours in (("Karolinska", {2: (230, 0, 0)}), ("ours", COLOURS["ours"])):
        d = w[w.source == src].drop_duplicates("unit")
        for image in d.image.iloc[rng.permutation(len(d))[:3]]:
            data = SOURCES[src]["data"]
            rgb = centre(np.array(Image.open(data / image).convert("RGB")))
            m = centre(load_mask(data / mask_path(image)).astype(np.uint8))
            lab = m * (2 if src == "Karolinska" else 1)
            panels.append((labelled(rgb, f"{src} {Path(image).name[:8]}"), overlay(rgb, lab, colours)))
    rows = []
    for i in range(0, len(panels), 3):
        cells = []
        for plain, over in panels[i:i + 3]:
            cells += [plain, gap(512, 4), over, gap(512, 24)]
        rows.append(np.concatenate(cells, 1))
        rows.append(gap(16, rows[-1].shape[1]))
    width = max(r.shape[1] for r in rows)
    rows = [np.concatenate([r, gap(r.shape[0], width - r.shape[1])], 1) for r in rows]
    Image.fromarray(np.concatenate(rows, 0)).save(HERE / "cv" / "radboud_label_examples.png")


def slide_figure(rad_ids, kar_ids, isup):
    cols = []
    for src, root, ids in (("Radboud", RAD, rad_ids), ("Karolinska", KAR, kar_ids)):
        for sid in ids:
            rgb = tifffile.imread(root / "images" / f"{sid}.tiff", level=2)[..., :3]
            lab = tifffile.imread(root / "masks" / f"{sid}_mask.tiff", level=2)[..., 0]
            h, w = min(rgb.shape[0], lab.shape[0]), min(rgb.shape[1], lab.shape[1])
            rgb, lab = rgb[:h, :w], lab[:h, :w]
            if h < w:
                rgb, lab = np.rot90(rgb), np.rot90(lab)
            pair = np.concatenate([rgb, np.full((rgb.shape[0], 4, 3), 255, np.uint8), overlay(rgb, lab, COLOURS[src])], 1)
            im = Image.fromarray(pair)
            im = im.resize((round(im.size[0] * 900 / im.size[1]), 900), Image.BOX)
            cols.append(labelled(np.array(im), f"{src} ISUP {isup.get(sid, '?')}"))
            cols.append(np.full((900, 24, 3), 255, np.uint8))
    Image.fromarray(np.concatenate(cols, 1)).save(HERE / "cv" / "radboud_slides.png")


def ci(v, fmt="{:.3f}"):
    v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = bootstrap_ci(v)
    return f"{fmt.format(v.mean())} [{fmt.format(lo)}, {fmt.format(hi)}]"


def main():
    rl = pd.read_csv(RAD / "labels.csv")
    kl = pd.read_csv(KAR / "labels.csv")
    isup = {**dict(zip(rl.image_id, rl.isup_grade)), **dict(zip(kl.image_id, kl.isup_grade))}
    rad = rl[(rl.isup_grade >= 1) & (rl.has_mask == 1)].sample(SLIDES, random_state=0).image_id.tolist()
    test = set(json.loads((HERE / "karolinska_all_split.json").read_text())["splits"]["test"])
    kar = kl[(kl.isup_grade >= 1) & kl.image_id.isin(test)].sample(SLIDES, random_state=0).image_id.tolist()

    others = sample()
    tiles = []
    for src in ("ours", "SICAPv2"):
        t = pd.read_csv(SOURCES[src]["index"])
        t = t[t.tumour_frac > 0]
        tiles += [(src, SOURCES[src]["data"], SOURCES[src]["um"], r.image, r.group) for r in t.itertuples()]
    slides = [("Radboud", s, RAD / "images" / f"{s}.tiff", RAD / "masks" / f"{s}_mask.tiff") for s in rad] + \
             [("Karolinska", s, KAR / "images" / f"{s}.tiff", KAR / "masks" / f"{s}_mask.tiff") for s in kar]
    with Pool(6) as pool:
        w = pd.DataFrame([r for rows in pool.imap_unordered(radboud_windows, [(s, i) for i, s in enumerate(rad)]) for r in rows]
                         + [r for rows in pool.imap_unordered(tile_windows, others, chunksize=8) for r in rows])
        s = pd.DataFrame(list(pool.imap_unordered(slide_row, slides)) + list(pool.imap_unordered(tile_glass, tiles, chunksize=16)))
    w.to_csv(HERE / "cv" / "radboud_label_windows.csv", index=False)
    s.to_csv(HERE / "cv" / "radboud_label_slides.csv", index=False)

    rows = []
    for src in ("Radboud", "Karolinska", "SICAPv2", "ours"):
        d = w[w.source == src]
        g = d.groupby("unit")
        u = per_unit(d).assign(glass=g.glass_in_tumour.sum() / g.tumour.sum(), lumen=g.lumen_in_tumour.sum() / g.tumour.sum())
        rows.append({"labels": src, "units": len(u), "windows": len(d),
                     "glass inside tumour": ci(u.glass), "lumen / small gaps inside tumour": ci(u.lumen),
                     **{f"gap fill at {p} um": ci(u[f"gap_fill_{p}"]) for p in PROBES},
                     "outline um per 1,000 um^2": ci(u.outline_density, "{:.1f}"), "median solidity": ci(u.median_solidity, "{:.2f}")})
    t1 = pd.DataFrame(rows)

    rows = []
    for src in ("Radboud", "Karolinska", "SICAPv2", "ours"):
        d = s[s.source == src]
        g = d.groupby("unit")
        r = {"labels": src, "units": g.ngroups, "um/px": f"{d.um.median():.2f}",
             "cancer label on glass": ci(g.cancer_glass.sum() / g.cancer.sum()),
             "cancer label on any empty pixel": ci(g.cancer_empty.sum() / g.cancer.sum())}
        if src in CANCER:
            r["all labels on glass"] = ci(g.labelled_glass.sum() / g.labelled.sum())
            r["tissue left unlabelled"] = ci(g.tissue_unlabelled.sum() / g.tissue.sum())
        rows.append(r)
    t2 = pd.DataFrame(rows).fillna("-")

    rw = w[w.source == "Radboud"]
    lab_tot = rw[[f"label_{k}" for k in range(6)]].sum()
    tis_tot = rw[[f"tissue_label_{k}" for k in range(6)]].sum()
    names = ["0 background / unknown", "1 stroma", "2 benign epithelium", "3 Gleason 3", "4 Gleason 4", "5 Gleason 5"]
    t3 = pd.DataFrame({"Radboud label": names, "share of block pixels": (lab_tot / lab_tot.sum()).round(3).values,
                       "share of tissue pixels": (tis_tot / tis_tot.sum()).round(3).values})

    ex_rad = rw.drop_duplicates("unit").sample(6, random_state=1).unit.tolist()
    ex_kar = s[s.source == "Karolinska"].sample(3, random_state=1).unit.tolist()
    example_figure(w, isup)
    slide_figure(ex_rad, ex_kar, isup)

    lines = ["# PANDA Radboud labels: how precise, and do they run onto glass?", "",
             "## 1. Windows centred on labelled cancer (256 x 256 um at 1 um/px, as cv/granularity.md)", "",
             "Mean over units (Radboud / Karolinska slides, SICAPv2 regions, our patients) [95% CI]. Glass = empty areas "
             ">= 2,500 um^2; lumen = other empty pixels. Gap fill at p um = tumour area a closing of radius p would add "
             "(high = gland-by-gland outline, low = region outline).", "",
             t1.to_markdown(index=False), "",
             "## 2. Whole cancer label at ~2 um/px", "",
             "Radboud and Karolinska: whole slides at TIFF level 1. SICAPv2 and ours: every tile with tumour (tile borders cut "
             "glass areas, so their glass share is a slight underestimate).", "",
             t2.to_markdown(index=False), "",
             "Slides with ISUP >= 1 but no cancer in the level-1 mask: "
             + ", ".join(f"{src} {int((s[s.source == src].cancer == 0).sum())} of {SLIDES}" for src in CANCER)
             + f"; Radboud slides without a usable cancer window: {SLIDES - rw.unit.nunique()}.", "",
             "## 3. Radboud label use in the sampled blocks (1,024 px around cancer)", "",
             t3.to_markdown(index=False), "",
             "Pictures (local only): cv/radboud_label_examples.png, cv/radboud_slides.png.", ""]
    (HERE / "cv" / "radboud_labels.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
