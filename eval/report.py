"""
Build results.html: scores, training curves and original vs predicted masks for the test slides.

    python report.py                       # compares runs/resnet34 and runs/resnet34_v2

Every number is read from the runs' summary.json, per_slide.csv and log.csv; the pictures are
linked from the eval folders (not copied), so open results.html from this folder. Nothing is
uploaded anywhere.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import json
from pathlib import Path

import pandas as pd

HERE = P.INDEX
OLD, NEW = "resnet34", "resnet34_v2"
CONFIGS = [  # (label, run, mode)
    ("Run 1, each tile alone", OLD, "tile"),
    ("Run 1 + context", OLD, "ctx256_tta"),
    ("New model, each tile alone", NEW, "tile"),
    ("New model + context", NEW, "ctx256_tta"),
    ("New model + context + blending", NEW, "ctx256_blend_tta"),
]
BEST = (NEW, "ctx256_blend_tta")


def load(run, split, mode):
    d = P.RUNS / run / f"eval_{split}_{mode}"
    if not (d / "summary.json").exists():
        return None, None
    per_slide = pd.read_csv(d / "per_slide.csv")
    per_slide["slide"] = per_slide.slide.str.replace(".bif - Series 0", "", regex=False)
    return json.loads((d / "summary.json").read_text()), per_slide.set_index("slide").dice.round(4).to_dict()


def main():
    configs = []
    for label, run, mode in CONFIGS:
        test, test_slides = load(run, "test", mode)
        val, _ = load(run, "val", mode)
        has_seams = "seam_jumps_pred" in test
        seams = {k: round(test[k], 3) for k in ("seam_ratio_pred", "seam_jumps_pred", "mid_jumps_pred",
                                                "seam_ratio_truth", "seam_jumps_truth", "mid_jumps_truth")} if has_seams else None
        configs.append(dict(label=label, run=run, mode=mode, epoch=test["epoch"],
                            test={k: round(test[k], 4) for k in ("dice", "iou", "slide_dice", "tumour_tile_dice", "clean_empty_tiles")},
                            seams=seams, val_dice=round(val["dice"], 4) if val else None, slides=test_slides))
    curves = {run: pd.read_csv(P.RUNS / run / "log.csv")[["epoch", "val_dice"]].round(4).values.tolist()
              for run in (OLD, NEW)}
    slides = sorted(configs[0]["slides"], key=lambda s: [int(n) for n in s.split(".")])
    stitching = None
    if (HERE / "stitching" / "report.csv").exists():   # from check_stitching.py
        rep = pd.read_csv(HERE / "stitching" / "report.csv", dtype={"slide": str})
        pairs = pd.read_csv(HERE / "stitching" / "seam_pairs.csv")
        busy = pairs[pairs.inner >= pairs.inner.quantile(0.75)]
        no_checker = rep[rep.slide != "26.3"]
        stitching = dict(
            slides=rep.fillna("").to_dict("records"), tiles=int(rep.tiles.sum()),
            bad_names=int(rep.bad_names.sum()), duplicates=int(rep.duplicates.sum()),
            aligned=round(float(pairs.aligned.mean()), 3), aligned_busy=round(float(busy.aligned.mean()), 3),
            pairs=len(pairs), seam_ratio=round(float((pairs.seam / pairs.inner).median()), 2),
            mask_seam_ratio=round(float(pairs.mask_seam.sum() / max(pairs.mask_inner.sum() / 2, 1)), 2),
            holes=int(rep.holes.sum()), holes_in_tissue=int(no_checker.holes_in_tissue.sum()),
            tissue_cutoffs=int(no_checker.tissue_cutoffs.sum()), tumour_cutoffs=int(no_checker.tumour_cutoffs.sum()))
    data = dict(configs=configs, curves=curves, slides=slides, old=OLD, new=NEW, best=BEST, stitching=stitching)
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    (HERE / "results.html").write_text(html, encoding="utf-8")
    print(f"-> {HERE / 'results.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>U-Net Segmentation Results</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --good: #006300;
  --series-1: #2a78d6; --series-2: #eb6834; --accent-wash: rgba(42,120,214,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --good: #0ca30c;
    --series-1: #3987e5; --series-2: #d95926; --accent-wash: rgba(57,135,229,0.16);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --good: #0ca30c;
  --series-1: #3987e5; --series-2: #d95926; --accent-wash: rgba(57,135,229,0.16);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 12px; }
p { margin: 0 0 12px; color: var(--ink-2); max-width: 75ch; }
.lede { margin-bottom: 24px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.stat .label { color: var(--ink-2); font-size: 13px; }
.stat .value { font-size: 34px; font-weight: 600; line-height: 1.2; }
.stat .note { color: var(--muted); font-size: 13px; }
.stat .up { color: var(--good); font-weight: 600; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap;
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
tr.best td { font-weight: 600; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); margin-bottom: 8px; }
.legend span::before { content: ""; display: inline-block; width: 14px; height: 3px; border-radius: 2px;
                       background: var(--c); vertical-align: middle; margin-right: 6px; }
.chart { position: relative; }
.chart svg { display: block; width: 100%; height: auto; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
       border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
       display: none; min-width: 170px; }
.tip .row { display: flex; justify-content: space-between; gap: 14px; }
.tip .row b { font-variant-numeric: tabular-nums; }
.tip .name::before { content: ""; display: inline-block; width: 10px; height: 3px; border-radius: 2px;
                     background: var(--c); vertical-align: middle; margin-right: 6px; }
details { margin-top: 10px; font-size: 14px; }
summary { cursor: pointer; color: var(--ink-2); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.tabs button, .toggle button { font: inherit; font-size: 14px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px; cursor: pointer; }
.tabs button[aria-pressed="true"], .toggle button[aria-pressed="true"] { background: var(--accent-wash);
  border-color: var(--series-1); font-weight: 600; }
.toggle { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 12px; }
.key { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); margin: 8px 0; }
.key i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 6px; }
.viewer img { display: block; width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border);
              background: #fff; }
.viewer a { color: var(--ink-2); font-size: 13px; }
.slidehead { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.slidehead .scores { font-size: 14px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.callout { border-left: 3px solid var(--series-2); padding: 8px 12px; margin: 12px 0; color: var(--ink-2);
           background: var(--surface); border-radius: 0 8px 8px 0; font-size: 14px; }
</style>
</head>
<body>
<main>
  <h1>U-Net tumour segmentation: results</h1>
  <p class="lede">ResNet-34 U-Net trained on 16 slides, scored on every tile of 7 held-out test slides
  (7,481 tiles) and 3 validation slides (8,426 tiles). Dice compares predicted tumour pixels with the
  annotated tumour pixels: 1 is a perfect match.</p>

  <div class="tiles" id="stats"></div>

  <h2>All four set-ups</h2>
  <p>"Context" predicts each tile inside a 1024 × 1024 window of its neighbours and averages over four flips.
  "Blending" also averages the overlapping windows of neighbouring tiles, each faded out towards its edges, so
  neighbouring tiles agree at their shared border. The new model adds stronger augmentation, crops shifted across
  tile borders, weight averaging and early stopping.</p>
  <div class="card scroll"><table id="configs"></table></div>

  <h2>Validation Dice during training</h2>
  <p>Run 1 swung between 0.67 and 0.82 and declined after epoch 8. The new recipe stays between 0.80 and 0.82
  and stopped early after epoch 12. Scores here are tile by tile.</p>
  <div class="card">
    <div class="legend" id="legend"></div>
    <div class="chart" id="chart"><div class="tip" id="tip"></div></div>
    <details><summary>Show as table</summary><div class="scroll"><table id="curve-table"></table></div></details>
  </div>

  <h2>Dice per test slide</h2>
  <div class="card scroll"><table id="slides-table"></table></div>

  <section id="stitching" hidden>
  <h2>How the tiles fit together</h2>
  <p>Checked on all exported tiles (<code>check_stitching.py</code>): whether every tile sits on the 1024-pixel grid,
  whether neighbouring tiles join without a gap, overlap or offset, and whether the export left out tiles that contain tissue.</p>
  <div class="tiles" id="stitch-stats"></div>
  <p style="margin-top:12px">Seam ratio compares the difference across a tile border with the difference between two neighbouring
  pixel columns inside a tile: about 1 means the border is invisible. "Cut-offs" are tile sides where tissue
  (or annotated tumour) runs into a cell the export left empty; they lie on the outer rim of the tissue.
  26.3 was exported as a sparse checkerboard, so every tile there borders empty cells.</p>
  <div class="card scroll"><table id="stitch-table"></table></div>
  <div id="seam-section" hidden>
  <h3 style="margin:24px 0 8px;font-size:15px">Is the merged prediction smooth?</h3>
  <p style="font-size:14px">At every border between two neighbouring test tiles, the pixels where the two touching lines of the
  mask disagree are counted, and compared with a virtual border through the middle of each tile, where the mask comes from
  a single prediction. A "visible jump" is 8 or more mismatched pixels along a border that tumour touches.</p>
  <div class="card scroll"><table id="seam-table"></table></div>
  </div>

  <h3 style="margin:24px 0 8px;font-size:15px">Coverage maps</h3>
  <p style="font-size:14px">Each slide drawn from its tiles. Hatched = no tile exported; red side = tissue runs into it;
  blue side = annotated tumour runs into it.</p>
  <div class="card viewer">
    <div class="tabs" id="map-tabs" role="group" aria-label="Slide"></div>
    <a id="map-link" target="_blank"><img id="map-img" alt=""></a>
  </div>
  </section>

  <h2>Original vs predicted masks</h2>
  <p>Pick a slide. Each picture shows the image, a comparison overlay, the original (annotated) mask and the
  predicted mask. Click a picture to open it at full size.</p>
  <div class="key"><span><i style="background:#ffc800"></i>tumour in both</span>
    <span><i style="background:#00be00"></i>missed (only in the original mask)</span>
    <span><i style="background:#e60000"></i>extra (only in the prediction)</span></div>
  <div class="card viewer">
    <div class="tabs" id="slide-tabs" role="group" aria-label="Test slide"></div>
    <div class="slidehead"><h3 id="slide-title" style="margin:4px 0"></h3><div class="scores" id="slide-scores"></div></div>
    <div class="toggle" id="model-toggle" role="group" aria-label="Model"></div>
    <div id="slide-note"></div>
    <a id="slide-link" target="_blank"><img id="slide-img" alt=""></a>
    <div id="seam-box">
      <h3 style="margin:20px 0 4px;font-size:15px">3 × 3 neighbouring tiles at full resolution</h3>
      <p style="font-size:14px">Stitched from nine tiles around a tumour border; the cyan dashed lines are the tile borders.
      A prediction that runs on smoothly across them is not blocky.</p>
      <a id="seam-link" target="_blank"><img id="seam-img" alt=""></a>
    </div>
    <h3 style="margin:20px 0 4px;font-size:15px">Example tiles at full resolution</h3>
    <p style="font-size:14px">Two tiles on a tumour border and one tile without tumour, from the same slide.</p>
    <a id="tile-link" target="_blank"><img id="tile-img" alt=""></a>
  </div>
</main>

<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const fmt = v => v == null ? "–" : v.toFixed(3);
const pct = v => Math.round(v * 100) + "%";
const pct1 = v => (v * 100).toFixed(1) + "%";
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const cfg = (run, mode) => DATA.configs.find(c => c.run === run && c.mode === mode);
const base = cfg(DATA.old, "tile"), best = cfg(...DATA.best);

// headline tiles
[["Test Dice", best.test.dice, base.test.dice], ["Validation Dice", best.val_dice, base.val_dice],
 ["Mean Dice per test slide", best.test.slide_dice, base.test.slide_dice],
 ["Test IoU", best.test.iou, base.test.iou]].forEach(([label, v, b]) => {
  const t = el("div", null, "card stat");
  t.append(el("div", label, "label"), el("div", fmt(v), "value"));
  const n = el("div", "first model: " + fmt(b) + " ", "note");
  n.append(el("span", "+" + (v - b).toFixed(3), "up"));
  t.append(n); $("stats").append(t);
});

// set-up table
(() => {
  const t = $("configs"), h = t.createTHead().insertRow();
  ["Set-up", "Test Dice", "IoU", "Mean slide Dice", "Dice on tumour tiles", "Clean tumour-free tiles", "Validation Dice"]
    .forEach(c => h.append(el("th", c)));
  const b = t.createTBody();
  DATA.configs.forEach(c => {
    const r = b.insertRow(); if (c === best) r.className = "best";
    [c.label, fmt(c.test.dice), fmt(c.test.iou), fmt(c.test.slide_dice), fmt(c.test.tumour_tile_dice),
     pct(c.test.clean_empty_tiles), fmt(c.val_dice)].forEach(v => r.append(el("td", v)));
  });
})();

// smoothness across tile borders
(() => {
  const withSeams = DATA.configs.filter(c => c.seams);
  if (!withSeams.length) return;
  $("seam-section").hidden = false;
  const t = $("seam-table"), h = t.createTHead().insertRow();
  ["Mask", "Seam ratio (1 = smooth)", "Tile borders with a visible jump", "Same, at virtual mid-tile borders"]
    .forEach(c => h.append(el("th", c)));
  const b = t.createTBody();
  const row = (label, ratio, jumps, mid, bold) => {
    const r = b.insertRow(); if (bold) r.className = "best";
    [label, ratio.toFixed(2), pct1(jumps), pct1(mid)].forEach(v => r.append(el("td", v)));
  };
  withSeams.forEach(c => row(c.label, c.seams.seam_ratio_pred, c.seams.seam_jumps_pred, c.seams.mid_jumps_pred, c === best));
  const s = withSeams[0].seams;
  row("Original masks (reference)", s.seam_ratio_truth, s.seam_jumps_truth, s.mid_jumps_truth, false);
})();

// stitching check
(() => {
  const S = DATA.stitching;
  if (!S) return;
  $("stitching").hidden = false;
  const n = v => v.toLocaleString("en-US");
  [["Tiles on the 1024-pixel grid", pct((S.tiles - S.bad_names) / S.tiles),
    `${n(S.tiles - S.bad_names)} of ${n(S.tiles)}; ${S.duplicates} duplicate positions`],
   ["Neighbours joining at shift 0", pct(S.aligned), `${pct(S.aligned_busy)} in well-textured tissue; seam ratio ${S.seam_ratio}`],
   ["Gaps inside the tissue", String(S.holes_in_tissue), `${S.holes} empty cells enclosed by tiles, all bordering glass (26.3 excluded)`],
   ["Tissue cut-offs at the rim", String(S.tissue_cutoffs), `tile sides out of ${n(S.tiles)} tiles; ${S.tumour_cutoffs} with annotated tumour`]]
    .forEach(([label, v, note]) => {
      const t = el("div", null, "card stat");
      t.append(el("div", label, "label"), el("div", v, "value"), el("div", note, "note"));
      $("stitch-stats").append(t);
    });
  const t = $("stitch-table"), h = t.createTHead().insertRow();
  const cols = [["slide", "Slide"], ["split", "Split"], ["tiles", "Tiles"], ["holes_in_tissue", "Gaps in tissue"],
                ["tissue_cutoffs", "Tissue cut-offs"], ["tumour_cutoffs", "Tumour cut-offs"],
                ["aligned_pairs", "Joined at shift 0"], ["seam_ratio", "Image seam ratio"], ["mask_seam_ratio", "Mask seam ratio"]];
  cols.forEach(([, c]) => h.append(el("th", c)));
  const b = t.createTBody();
  S.slides.forEach(s => {
    const r = b.insertRow();
    cols.forEach(([k]) => {
      const v = s[k];
      r.append(el("td", k === "aligned_pairs" ? pct(v) : k.endsWith("seam_ratio") ? v.toFixed(2) : String(v)));
    });
  });
  let current = "10.2";
  const show = s => {
    current = s;
    $("map-img").src = $("map-link").href = `stitching/maps/${s}.png`;
    $("map-img").alt = `Coverage map of slide ${s}`;
    [...$("map-tabs").children].forEach(b => b.setAttribute("aria-pressed", b.dataset.s === s));
  };
  S.slides.forEach(s => { const b = el("button", s.slide); b.dataset.s = s.slide; b.onclick = () => show(s.slide); $("map-tabs").append(b); });
  show(S.slides.some(s => s.slide === current) ? current : S.slides[0].slide);
})();

// per-slide table
(() => {
  const t = $("slides-table"), h = t.createTHead().insertRow();
  ["Slide", "Run 1, each tile alone", best.label, "Change"].forEach(c => h.append(el("th", c)));
  const b = t.createTBody();
  DATA.slides.forEach(s => {
    const r = b.insertRow(), d = best.slides[s] - base.slides[s];
    [s, fmt(base.slides[s]), fmt(best.slides[s]), (d >= 0 ? "+" : "") + d.toFixed(3)].forEach(v => r.append(el("td", v)));
  });
})();

// training curves: one axis, two series, crosshair tooltip
(() => {
  const series = [{ name: "Run 1", key: DATA.old, color: "var(--series-1)" },
                  { name: "New model", key: DATA.new, color: "var(--series-2)" }];
  series.forEach(s => { const sp = el("span", s.name); sp.style.setProperty("--c", s.color); $("legend").append(sp); });
  const maxEpoch = Math.max(...series.map(s => DATA.curves[s.key].length));
  let lastWidth = 0;
  const draw = () => {
  const W = Math.max(320, Math.round($("chart").clientWidth)), H = 300, m = { l: 48, r: 92, t: 14, b: 36 };
  if (W === lastWidth) return;
  lastWidth = W;
  $("chart").querySelector("svg")?.remove();
  const y0 = 0.65, y1 = 0.85;
  const X = e => m.l + (e - 1) / (maxEpoch - 1) * (W - m.l - m.r);
  const Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Validation Dice per epoch for run 1 and the new model");
  const add = (tag, attrs, parent = svg) => { const n = document.createElementNS(ns, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]); parent.append(n); return n; };
  for (let v = y0; v <= y1 + 1e-9; v += 0.05) {
    add("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 });
    add("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "var(--muted)" }).textContent = v.toFixed(2);
  }
  add("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1 });
  const step = W < 520 ? 4 : 2;
  for (let e = 1; e <= maxEpoch; e++) if (e === 1 || e % step === 0)
    add("text", { x: X(e), y: H - m.b + 18, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)" }).textContent = e;
  add("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)" }).textContent = "epoch";
  series.forEach(s => {
    const pts = DATA.curves[s.key];
    add("path", { d: pts.map((p, i) => (i ? "L" : "M") + X(p[0]) + "," + Y(p[1])).join(""),
                  fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" });
    const bi = pts.reduce((a, p, i) => p[1] > pts[a][1] ? i : a, 0);
    add("circle", { cx: X(pts[bi][0]), cy: Y(pts[bi][1]), r: 4.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 });
    const last = pts[pts.length - 1];
    add("text", { x: X(last[0]) + 8, y: Y(last[1]) + 4, "font-size": 12, fill: "var(--ink-2)" }).textContent = s.name;
    s.best = pts[bi];
  });
  const cross = add("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const hit = add("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent", tabindex: 0 });
  $("chart").prepend(svg);
  const tip = $("tip");
  const show = e => {
    cross.setAttribute("x1", X(e)); cross.setAttribute("x2", X(e)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", "Epoch " + e, "name"));
    tip.firstChild.style.cssText = "color:var(--ink-2);margin-bottom:4px";
    tip.firstChild.className = "";
    series.forEach(s => {
      const p = DATA.curves[s.key].find(q => q[0] === e);
      const row = el("div", null, "row"), n = el("span", s.name, "name");
      n.style.setProperty("--c", s.color);
      row.append(n, el("b", p ? p[1].toFixed(3) + (p === s.best ? " best" : "") : "stopped"));
      tip.append(row);
    });
    const box = $("chart").getBoundingClientRect(), px = X(e) / W * box.width;
    tip.style.display = "block";
    tip.style.left = Math.min(px + 12, box.width - tip.offsetWidth - 4) + "px";
    tip.style.top = "8px";
  };
  let focusEpoch = 1;
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
    focusEpoch = Math.max(1, Math.min(maxEpoch, Math.round((x - m.l) / (W - m.l - m.r) * (maxEpoch - 1) + 1)));
    show(focusEpoch);
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  hit.addEventListener("focus", () => show(focusEpoch));
  hit.addEventListener("blur", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  hit.addEventListener("keydown", ev => {
    if (ev.key === "ArrowRight") focusEpoch = Math.min(maxEpoch, focusEpoch + 1);
    else if (ev.key === "ArrowLeft") focusEpoch = Math.max(1, focusEpoch - 1); else return;
    ev.preventDefault(); show(focusEpoch);
  });
  };
  draw();
  new ResizeObserver(draw).observe($("chart"));
  // table view
  const t = $("curve-table"), h = t.createTHead().insertRow();
  ["Epoch", ...series.map(s => s.name)].forEach(c => h.append(el("th", c)));
  const b = t.createTBody();
  for (let e = 1; e <= maxEpoch; e++) {
    const r = b.insertRow(); r.append(el("td", e));
    series.forEach(s => { const p = DATA.curves[s.key].find(q => q[0] === e); r.append(el("td", p ? p[1].toFixed(3) : "–")); });
  }
})();

// slide viewer
(() => {
  const models = [{ label: best.label, cfg: best }, { label: "Run 1, each tile alone", cfg: base }];
  const notes = {
    "10.2": "Improved most: the blocky tile-shaped patches of run 1 are gone. The top and bottom sections still miss part of the upper-right region.",
    "40.2": "Still the weakest slide. The model marks the same areas in the left half as tumour in both serial sections: either tumour that was not annotated or benign glands that look like tumour. A pathologist should check these areas.",
    "43.2": "Like 40.2, part of the false-alarm area (red) repeats in both sections and is worth a pathologist's look.",
  };
  let slide = DATA.slides.includes("10.2") ? "10.2" : DATA.slides[0], model = 0;
  const path = (c, kind) => `runs/${c.run}/eval_test_${c.mode}/${kind}/${slide}.png`;
  const render = () => {
    const c = models[model].cfg;
    $("slide-title").textContent = "Slide " + slide;
    $("slide-scores").textContent = `Dice: run 1 ${fmt(base.slides[slide])} → ${best.label.toLowerCase()} ${fmt(best.slides[slide])}`;
    $("slide-img").src = path(c, "slides"); $("slide-link").href = path(c, "slides");
    $("slide-img").alt = `Slide ${slide}: image, comparison, original mask and predicted mask (${models[model].label})`;
    $("seam-img").src = path(c, "seams"); $("seam-link").href = path(c, "seams");
    $("seam-img").alt = `3 x 3 tiles of slide ${slide} at full resolution with tile borders (${models[model].label})`;
    $("seam-img").onerror = () => { $("seam-box").hidden = true; };
    $("seam-img").onload = () => { $("seam-box").hidden = false; };
    $("tile-img").src = path(c, "tiles"); $("tile-link").href = path(c, "tiles");
    $("tile-img").alt = `Example tiles of slide ${slide} (${models[model].label})`;
    $("slide-note").replaceChildren(...(notes[slide] ? [el("div", notes[slide], "callout")] : []));
    [...$("slide-tabs").children].forEach(b => b.setAttribute("aria-pressed", b.dataset.s === slide));
    [...$("model-toggle").children].forEach((b, i) => b.setAttribute("aria-pressed", i === model));
  };
  DATA.slides.forEach(s => { const b = el("button", s); b.dataset.s = s; b.onclick = () => { slide = s; render(); }; $("slide-tabs").append(b); });
  models.forEach((mo, i) => { const b = el("button", mo.label); b.onclick = () => { model = i; render(); }; $("model-toggle").append(b); });
  render();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
