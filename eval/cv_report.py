"""
Build cv/report.html: the patient cross-validation of SICAPv2 pretraining in one page.

    python cv_report.py

Reads what the other scripts wrote (run them first):
    cv_summary.py          cv/summary_ctx256_blend_tta.csv + the per-tile results it reads
    evaluate_sicap.py      runs/sicap_pretrain/eval_sicap_{val,guard}_tile/
    evaluate.py            runs/sicap_pretrain/eval_all_test_tile/ (cv/all.json, via run_cv.py)
    miss_analysis.py       cv/miss_composition.csv, cv/miss_examples/
    outlier_check.py       cv/outlier_45.json
The page shows patient tissue (cv/miss_examples), so it stays on this machine: open it from
the cv folder, never upload it.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import html
import json
from pathlib import Path

import pandas as pd

from cv_summary import METRICS, REFERENCE, overview, paired, per_patient, rates, sources
from evaluate_sicap import comparable

HERE = P.INDEX
CV = HERE / "cv"
MODE = "ctx256_blend_tta"
NAMES = {"imagenet": "ImageNet start, trained on ours", "sicap": "SICAPv2 start, trained on ours",
         "sicap_only": "SICAPv2 only, never trained on ours"}
SHORT = {"imagenet": "ImageNet start", "sicap": "SICAPv2 start", "sicap_only": "SICAPv2 only"}
SERIES = {"imagenet": 1, "sicap": 2, "sicap_only": 3}          # fixed categorical slots
TYPE_LABEL = {True: "needle biopsy", False: "large section"}
TYPE_PLURAL = {True: "needle biopsies", False: "large sections"}


def table(df, cls=""):
    return df.to_html(index=False, border=0, classes=f"t {cls}", escape=True)


def dot_plot(per, biopsy, metric, domain, ticks, fmt, title):
    """Patients as rows (biopsies first, then large sections, each sorted by the ImageNet start),
    one dot per model, a line joining the two trained models of each patient."""
    order = []
    for is_biopsy in (True, False):
        pats = [p for p in per["imagenet"].index if (p in biopsy) == is_biopsy]
        order.append((TYPE_PLURAL[is_biopsy], sorted(pats, key=lambda p: -per["imagenet"].loc[p, metric])))
    row_h, top, left, width = 24, 30, 92, 560
    n_rows = sum(len(p) for _, p in order) + len(order)
    h = top + n_rows * row_h + 10
    x = lambda v: left + (min(max(v, domain[0]), domain[1]) - domain[0]) / (domain[1] - domain[0]) * width
    svg = [f'<svg class="dots" viewBox="0 0 {left + width + 20} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in ticks:
        svg.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{h - 6}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{top - 12}" text-anchor="middle">{fmt(t)}</text>')
    y = top
    for label, pats in order:
        svg.append(f'<text class="group" x="4" y="{y + row_h * 0.65:.1f}">{label}</text>')
        y += row_h
        for p in pats:
            cy = y + row_h / 2
            svg.append(f'<text class="label" x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end">patient {p}</text>')
            a, b = per["imagenet"].loc[p, metric], per["sicap"].loc[p, metric]
            svg.append(f'<line class="pair" x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
            for name in ["sicap_only", "imagenet", "sicap"]:
                v = per[name].loc[p, metric]
                svg.append(f'<circle class="s{SERIES[name]}" cx="{x(v):.1f}" cy="{cy:.1f}" r="5">'
                           f'<title>patient {p} ({label}) - {NAMES[name]}: {fmt(v)}</title></circle>')
            y += row_h
    svg.append("</svg>")
    return "\n".join(svg)


def legend():
    return '<div class="legend">' + "".join(
        f'<span><svg width="12" height="12"><circle class="s{SERIES[n]}" cx="6" cy="6" r="5"/></svg>{NAMES[n]}</span>'
        for n in ["imagenet", "sicap", "sicap_only"]) + '<span class="pairkey">line: the two trained models of one patient</span></div>'


def main():
    folds = json.loads((CV / "folds.json").read_text())
    biopsy = set(folds["biopsy_patients"])
    found = sources(MODE)
    per = {}
    for name, paths in found.items():
        counts = per_patient(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
        per[name] = pd.concat([counts, rates(counts)], axis=1)
    missing = [n for n in ["imagenet", "sicap", "sicap_only"] if n not in per or len(per[n]) < 17]
    if missing:
        raise SystemExit(f"cross-validation incomplete for {missing}; finish run_cv.py first")

    groups = {"All 17 patients": None, "8 needle-biopsy patients": True, "9 large-section patients": False}
    main_tables, paired_tables = [], []
    for title, want in groups.items():
        sub = {n: p.loc[[x for x in p.index if want is None or (x in biopsy) == want]] for n, p in per.items()}
        ov = overview(sub)
        ov["configuration"] = ov.configuration.map(NAMES)
        main_tables.append(f"<h3>{title}</h3>" + table(ov))
        pr = paired(sub)
        pr["configuration"] = pr.configuration.str.replace(" - " + REFERENCE, "").map(SHORT) + " minus ImageNet start"
        pr.columns = ["difference", "patients", "Dice [CI]", "Dice better / worse", "sensitivity [CI]",
                      "sensitivity better / worse", "false tumour [CI]", "false tumour better / worse"]
        paired_tables.append(f"<h3>{title}</h3>" + table(pr))

    fold_rows = pd.DataFrame([dict(fold=f["fold"], **{"test: needle biopsies": ", ".join(str(p) for p in f["test"] if p in biopsy),
                                                     "test: large sections": ", ".join(str(p) for p in f["test"] if p not in biopsy),
                                                     "val (early stopping)": ", ".join(str(p) for p in f["val"])})
                              for f in folds["folds"]])

    pp = pd.DataFrame({"patient": per["imagenet"].index,
                       "specimen": [TYPE_LABEL[p in biopsy] for p in per["imagenet"].index],
                       "tiles": per["imagenet"].tiles.values})
    for m in ["dice", "sensitivity", "false_tumour_rate"]:
        for n in ["imagenet", "sicap", "sicap_only"]:
            pp[f"{m} {n}"] = per[n].loc[pp.patient, m].round(3).values
    pp = pp.sort_values(["specimen", "patient"])

    # the drop: same metrics, each tile alone, SICAPv2 held-out vs our patients
    gap = {f"SICAPv2 {s} (per region)": comparable(pd.read_csv(P.RUNS / "sicap_pretrain" / f"eval_sicap_{s}_tile" / "per_tile.csv"), 896 ** 2)
           for s in ["val", "guard"]}
    ours = pd.read_csv(P.RUNS / "sicap_pretrain" / "eval_all_test_tile" / "per_tile.csv")
    ours = ours.assign(slide=ours.slide.str.split(".").str[0])
    gap["ours, all 17 patients"] = comparable(ours, 512 ** 2)
    for want in (True, False):
        gap[f"ours, {TYPE_LABEL[want]} patients"] = comparable(ours[ours.slide.astype(int).isin(biopsy) == want], 512 ** 2)
    gap = pd.DataFrame(gap).drop(["tiles"])
    gap.index = ["tumour share of pixels", "Dice, all pixels", "Dice, pixels of tumour tiles", "mean Dice of tumour tiles",
                 "mean Dice per region / patient", "tumour-free tiles left clean", "tumour missed", "false tumour (per tumour pixel)"]
    gap = gap.round(3).reset_index().rename(columns={"index": "metric"})

    mc = pd.read_csv(CV / "miss_composition.csv")
    mc["gland-like missed"] = mc.missed_gland_like / mc.tumour_gland_like
    mc["stroma-like missed"] = mc.missed_stroma_like / mc.tumour_stroma_like
    agg = mc.groupby(["type", "model"])[["tumour_gland_like", "tumour_stroma_like", "missed_gland_like", "missed_stroma_like"]].sum()
    agg["gland-like tumour missed"] = (agg.missed_gland_like / agg.tumour_gland_like).round(3)
    agg["stroma-like tumour missed"] = (agg.missed_stroma_like / agg.tumour_stroma_like).round(3)
    agg = agg.reset_index()[["type", "model", "gland-like tumour missed", "stroma-like tumour missed"]]
    agg["model"] = agg.model.map(NAMES)
    large = mc[mc.type == "large section"].pivot(index="patient", columns="model", values=["gland-like missed", "stroma-like missed"]).round(2)
    large.columns = [f"{m.split('-')[0]} missed, {SHORT[c]}" for m, c in large.columns]
    large = large.reset_index()

    o45 = json.loads((CV / "outlier_45.json").read_text())
    bios = pd.DataFrame(o45["biopsies"])[["patient", "tissue_tiles", "blurry_share", "sharpness_median", "R", "G", "B"]].round(2)
    focus = pd.DataFrame([dict(model=NAMES[c], tiles=v[k]["tiles"], focus=k, **{"false-tumour rate": round(v[k]["false_tumour_rate"], 3),
                               "share of the patient's false tumour": round(v[k]["share_of_false_tumour"], 3)})
                          for c, v in o45["by_focus"].items() for k in ["sharp", "blurry"]])
    means45 = pd.DataFrame([dict(model=NAMES[c], **{"mean Dice": round(v["dice"], 3), "without 45": round(v["dice_without"], 3),
                                 "mean false tumour": round(v["false_tumour"], 3), "without 45 ": round(v["false_tumour_without"], 3)})
                            for c, v in o45["means"].items()])

    examples = "".join(f'<figure><img src="miss_examples/{p}.png" alt="tiles of patient {p}"><figcaption>Patient {p}: the four tumour tiles '
                       f'the SICAPv2-only model misses most. Columns: image, annotation (white = tumour), SICAPv2-only prediction, '
                       f'prediction of the fold model trained on our tiles (ImageNet start), gland-like map.</figcaption></figure>'
                       for p in [31, 30, 26] if (CV / "miss_examples" / f"{p}.png").exists())

    body = f"""
<header><h1>SICAPv2 pretraining, patient cross-validation</h1>
<p class="sub">ResNet-34 U-Net, 17 patients / 26 slides / 52,877 tiles, 5 folds by patient, every patient tested once by a model that never saw it.
Inference: neighbour context + blending + flips (evaluate.py default) unless stated. Means are over patients; [95% CI] by bootstrap over patients.</p></header>

<section><h2>Findings</h2><ol class="findings">
<li><b>SICAPv2 pretraining does not raise Dice on our data.</b> Paired difference to the ImageNet start over 17 patients:
{paired({n: per[n] for n in ['imagenet', 'sicap']}).iloc[0]['dice difference']}, better on {paired({n: per[n] for n in ['imagenet', 'sicap']}).iloc[0]['dice: better / worse'].split(' / ')[0]} patients, worse on {paired({n: per[n] for n in ['imagenet', 'sicap']}).iloc[0]['dice: better / worse'].split(' / ')[1]}.
The +0.054 seen on the original 3-patient test split does not hold.</li>
<li><b>It shifts the errors by specimen type.</b> On needle biopsies (SICAPv2's own specimen type) it lowers false tumour with Dice unchanged; on large sections it lowers sensitivity a little. Per-type tables below; 8 and 9 patients per group, so treat these as hypotheses.</li>
<li><b>Without training on our tiles, the SICAPv2 model loses most on large sections.</b> Same metric on both datasets (mean Dice per region / patient, tiles alone): 0.79-0.84 on held-out SICAPv2, {gap.iloc[4][f'ours, {TYPE_LABEL[True]} patients']:.2f} on our biopsies, {gap.iloc[4][f'ours, {TYPE_LABEL[False]} patients']:.2f} on our large sections. Biopsies: it over-calls; large sections: it misses tumour.</li>
<li><b>The large-section misses are partly annotation style, partly real.</b> The SICAPv2-only model leaves out half of the stroma-like annotated tumour (in-domain models about a fifth), but also about a quarter of the gland-like tumour (in-domain about a tenth). Patients 10, 13 and 35: glands found, stroma between them left out. Patients 26, 30 and 31: whole tumour areas, glands included, missed.</li>
<li><b>Patient 45 is a focus outlier, not a staining or annotation problem.</b> {o45['biopsies'][[b['patient'] for b in o45['biopsies']].index(45)]['blurry_share']:.0%} of its tissue tiles are out of focus; almost all of its false tumour lies on them.</li>
</ol></section>

<section><h2>Folds</h2><p>Each fold's test set holds both specimen types; two other patients serve as val for early stopping.</p>{table(fold_rows)}</section>

<section><h2>Results per model</h2>{''.join(main_tables)}
<p class="note">dice / sensitivity / precision per patient; false_tumour_rate = share of the tumour-free tissue (glass excluded) predicted tumour. Pooled columns add up all pixels, so patient 36 (a quarter of all tiles) dominates them.</p></section>

<section><h2>Does pretraining help? Paired differences, same patients</h2>{''.join(paired_tables)}
<p class="note">Difference = SICAPv2 start minus ImageNet start (or SICAPv2 only minus ImageNet start) per patient, then averaged. For false tumour, lower is better.</p></section>

<section><h2>Every patient</h2>{legend()}
<h3>Dice per patient</h3>{dot_plot(per, biopsy, 'dice', (0, 1), [0, .25, .5, .75, 1], lambda v: f'{v:.2f}', 'Dice per patient and model')}
<h3>False-tumour rate per patient (share of tumour-free tissue predicted tumour)</h3>{dot_plot(per, biopsy, 'false_tumour_rate', (0, .5), [0, .1, .2, .3, .4, .5], lambda v: f'{v:.0%}', 'False-tumour rate per patient and model')}
<details><summary>Table view</summary>{table(pp, 'small')}</details></section>

<section><h2>The drop from SICAPv2 to our slides, same metrics</h2>
<p>SICAPv2-only model, each tile predicted alone (SICAPv2 tiles enlarged to our scale). SICAPv2's official test split cannot be recovered from the Kaggle copy;
val (1,883 tiles, used to pick the pretraining epoch) and guard (1,077 tiles, never used) are held out by prepare_sicap.py. Regions are groups of overlapping SICAPv2 tiles, about two thirds of a slide.</p>
{table(gap)}</section>

<section><h2>What the misses on large sections are</h2>
<p>Every annotated tumour pixel is classed by a stain estimate (haematoxylin density over 31 x 31 px, Otsu threshold per slide): gland-like (nuclei-dense) or stroma-like tissue (glass and lumen left out).
If a model only left out stroma inside coarse outlines, its gland-like miss rate would match the in-domain models'. Tile mode, threshold 0.5.</p>
<h3>All tumour tiles, by specimen type</h3>{table(agg)}
<h3>Large-section patients</h3>{table(large, 'small')}
<div class="figs">{examples}</div>
<p class="note">The stain estimate is no trained epithelium detector; on the very pale slide 26.3 it also marks some stroma bands as gland-like.</p></section>

<section><h2>Patient 45</h2>
<p>Needle biopsy with many false alarms even after training on our tiles. Out-of-focus = variance of the Laplacian over tissue below 30.</p>
<h3>Focus and colour of the biopsy patients</h3>{table(bios)}
<h3>False tumour on sharp vs blurry tiles of patient 45</h3>{table(focus)}
<h3>Means over patients with and without patient 45</h3>{table(means45)}</section>

<section><h2>Caveats</h2><ul>
<li>17 patients; per-type groups of 8 and 9. Several comparisons, none corrected for multiplicity.</li>
<li>One training run per fold and configuration: the intervals cover patient variation, not seed variation.</li>
<li>The gland / stroma split is a stain heuristic, not a trained detector.</li>
<li>SICAPv2 held-out sets are our own split of the Kaggle copy, not the official test partition.</li>
<li>Pretraining enlarges SICAPv2 tiles 1.5-2x to match our pixel size; this factor was estimated from nucleus sizes, not from scanner metadata.</li>
</ul></section>
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SICAPv2 Pretraining CV</title><style>{CSS}</style></head><body><main class="viz-root">{body}</main></body></html>"""
    (CV / "report.html").write_text(page, encoding="utf-8")
    print(f"-> {CV / 'report.html'}")


CSS = """
.viz-root { color-scheme: light; --surface-1: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark; --surface-1: #1a1a19; --page: #0d0d0d;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; }
html, body { margin: 0; background: #f9f9f7; }
@media (prefers-color-scheme: dark) { html, body { background: #0d0d0d; } }
.viz-root { background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 6px; } h2 { font-size: 19px; margin: 36px 0 10px; } h3 { font-size: 15px; margin: 20px 0 8px; color: var(--ink-2); }
.sub, .note { color: var(--ink-2); } .note { font-size: 13px; }
section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 4px 18px 16px; margin-top: 16px; }
.findings li { margin: 8px 0; }
.t { border-collapse: collapse; font-size: 13px; margin: 6px 0 10px; display: block; overflow-x: auto; max-width: 100%; }
.t th, .t td { padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; font-variant-numeric: tabular-nums; }
.t th { color: var(--ink-2); font-weight: 600; border-bottom-color: var(--axis); }
.t.small { font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); margin: 8px 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; } .pairkey::before { content: ""; width: 18px; border-top: 2px solid var(--muted); display: inline-block; }
svg.dots { width: 100%; max-width: 680px; height: auto; display: block; }
svg .grid { stroke: var(--grid); stroke-width: 1; } svg .tick, svg .label { fill: var(--muted); font-size: 11px; }
svg .label { fill: var(--ink-2); } svg .group { fill: var(--ink); font-size: 12px; font-weight: 600; }
svg .pair { stroke: var(--muted); stroke-width: 2; }
svg circle { stroke: var(--surface-1); stroke-width: 2; } svg circle:hover { stroke: var(--ink); }
.s1 { fill: var(--series-1); } .s2 { fill: var(--series-2); } .s3 { fill: var(--series-3); }
.figs figure { margin: 14px 0; } .figs img { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }
figcaption { font-size: 12px; color: var(--ink-2); }
details summary { cursor: pointer; color: var(--ink-2); margin: 10px 0; }
"""

if __name__ == "__main__":
    main()
