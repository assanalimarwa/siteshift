"""
Effect of the three fixes on the 5 parity slides: OLD pipeline (square band, small-region rule by detection, inner band = distance to the nearest non-tumour pixel;
code kept as protocol_square_old.py) against the FINAL pipeline (protocol.py), same predictions. Writes band_effect.md / band_effect.csv.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import importlib.util
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
HERE = P.OUT
EXP = P.PRED
SLIDES = [("39.2.bif - Series 0", "fold3", "biopsy"), ("9.1.bif - Series 0", "fold0", "biopsy"), ("23.3.bif - Series 0", "fold2", "biopsy"),
          ("44.2.bif - Series 0", "fold4", "large"), ("44.1.bif - Series 0", "fold4", "large")]
MODELS = {"nnU-Net-style site CV": lambda fold: EXP / "nnunet" / "predictions" / fold,
          "nnU-Net-style all-public": lambda fold: EXP / "nnunet_allpublic" / "predictions" / "all"}
BINS = ["missed", "fill_px", "fn_band", "fp_band", "false"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(args):
    which, path = args
    m = load(P.EVAL / "protocol_square_old.py" if which == "old" else str(P.EVAL / "protocol.py"), f"p_{which}")
    _, s, _ = m.analyse_slide(path, 0.5, True)
    return s


def main():
    jobs = [(w, f(fold) / (re.sub(r"[^\w.-]+", "_", sl) + ".npz")) for _, f in MODELS.items() for sl, fold, _ in SLIDES for w in ("old", "new")]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run, jobs))
    rows, i = [], 0
    for mname in MODELS:
        for sl, fold, typ in SLIDES:
            old, new = res[i], res[i + 1]
            i += 2
            d = 2 * new["tp"] + new["fp"] + new["fn"]
            assert (old["tp"], old["fp"], old["fn"]) == (new["tp"], new["fp"], new["fn"])
            r = {"model": mname, "slide": sl, "type": typ}
            for b in BINS:
                r[b + " old"], r[b + " new"] = int(old[b]), int(new[b])
                r[b + " d/err"] = (new[b] - old[b]) / d
            r["boundary d/err"] = r["fn_band d/err"] + r["fp_band d/err"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.round(6).to_csv(HERE / "band_effect.csv", index=False)
    lines = ["# Effect of the three fixes on the 5 parity slides (old pipeline vs final pipeline)", "",
             "Old: square 50 um band, a sub-2,500 um^2 region counted as found when >= 10% marked, inner band = distance to the nearest non-tumour pixel.",
             "Final: Euclidean 50 um band, a sub-2,500 um^2 component can never be detected (its unmarked pixels are missed), inner band = distance to the outline pixels",
             "(the reference attribution.py). Same predictions; tp, fp, fn and the region counts are identical. Change (final minus old) as a share of 2 tp + fp + fn of the slide,",
             "i.e. in units of (1 - Dice) of the slide, in percentage points.", "",
             "| model | slide | type | missed | interior (fill) | boundary | false tumour |", "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | {r['slide'].split('.bif')[0]} | {r['type']} | {r['missed d/err']*100:+.2f} | {r['fill_px d/err']*100:+.2f} | "
                     f"{r['boundary d/err']*100:+.2f} | {r['false d/err']*100:+.2f} |")
    m = df.groupby("type")[["missed d/err", "fill_px d/err", "boundary d/err", "false d/err"]].mean() * 100
    lines += ["", "Mean over the slides of a type (points of the error):", "", "| type | missed | interior (fill) | boundary | false tumour |", "|---|---|---|---|---|"]
    for t, r in m.iterrows():
        lines.append(f"| {t} | {r['missed d/err']:+.2f} | {r['fill_px d/err']:+.2f} | {r['boundary d/err']:+.2f} | {r['false d/err']:+.2f} |")
    (HERE / "band_effect.md").write_text("\n".join(lines) + "\n")
    print(df[["model", "slide", "missed d/err", "fill_px d/err", "boundary d/err", "false d/err"]].round(4).to_string(index=False))
    print("by type (mean, points):")
    print((df.groupby("type")[["missed d/err", "fill_px d/err", "boundary d/err", "false d/err"]].mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
