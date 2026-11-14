"""
labels_vs_errors.md: each public dataset's label geometry next to its single-source model's error breakdown at our site.

    python ds_link.py

Label geometry from data/summary.csv (ds_report.py; the stroma share was dropped by the user 2026-11-10). Error bins from
the 47-patient protocol (cv47/protocol_by_patient.csv, report_protocol47.py; parity-verified protocol.py, Euclidean band;
switched from the 17-patient cv/protocol.md on 2026-11-12, old version ds_link_17patient_2026-11-12.py): per patient the
contribution to 1 - patient-pooled Dice of missed regions, false tumour, incomplete fill inside found regions and the
+-50 um boundary band (the four add up to 1 - Dice), mean over the 47 patients [95% bootstrap CI]; share = the bin's part
of the mean error. 'isolated / attached' = protocol.measures' split of the false bin (components with no pixel within
50 um of tumour / the rest).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

from pathlib import Path

import numpy as np
import pandas as pd

from ds_lib import DATA, OUT, U

MODELS = {"SICAPv2": "sicap_pretrain", "PANDA-Karolinska": "karolinska_all_unet", "PANDA-Radboud": "radboud_unet"}
SITE = "unet_site"
BINS = ["missed", "false", "fill", "boundary"]


def boot_ci(v, n=10000, seed=0):
    v = v.dropna().values
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return f"{v.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def main():
    bp = pd.read_csv(P.RESULTS / "cv47" / "protocol_by_patient.csv")
    s = pd.read_csv(DATA / "summary.csv", index_col=0)
    geo = ["gap fill at 40 um", "glass inside tumour", "tumour regions per 256-um window", "region roughness P^2/(4 pi A)"]
    out = ["# Label geometry vs error type at our site", "",
           "_paper_figs/src/ds_link.py; errors from cv47/protocol_by_patient.csv (47 patients, parity-verified protocol.py, "
           "report_protocol47.py); label geometry from the per-dataset analysis (median over units [95% CI])._", "",
           "\"All public combined\" (model A) = SICAPv2 + PANDA-Karolinska + PANDA-Radboud: 3,814 + 1,800 + 202 = 5,816; "
           "the rows below are the single-source models.", "",
           "## Label geometry of the training data", "",
           "| dataset | gap fill at 40 um | glass in tumour | regions per window | roughness P^2/(4 pi A) |",
           "|---|---|---|---|---|"]
    for d in ["Ours", "SICAPv2", "PANDA-Karolinska", "PANDA-Radboud"]:
        out.append(f"| {d} | " + " | ".join(str(s.loc[g, d]) for g in geo) + " |")
    share = {}
    for mode in ["tile", "full"]:
        out += ["", f"## Error breakdown at our site, 47 patients, {mode} mode (contribution to 1 - Dice [95% CI]; share of the "
                "mean error in brackets)", "",
                "| model (trained on) | 1 - Dice | missed | false tumour | incomplete fill | boundary | false: isolated / attached |",
                "|---|---|---|---|---|---|---|"]
        for d, m in [("ours (site CV, reference)", SITE)] + [(f"{k}-only U-Net", v) for k, v in MODELS.items()]:
            x = bp[(bp.model == m) & (bp["mode"] == mode)]
            if not len(x):
                out.append(f"| {d} | not run in {mode} mode | | | | | |")
                continue
            tot = sum(x[f"err_{b}"].mean() for b in BINS)
            share[(m, mode)] = {b: x[f"err_{b}"].mean() / tot for b in BINS}
            cells = [f"{boot_ci(x[f'err_{b}'])} ({share[(m, mode)][b]:.0%})" for b in BINS]
            out.append(f"| {d} | {boot_ci(1 - x.dice)} | " + " | ".join(cells) +
                       f" | {x.err_false_isolated.mean():.3f} / {x.err_false_attached.mean():.3f} |")
    sh = lambda m, b, mode="tile": f"{share[(m, mode)][b]:.0%}" if (m, mode) in share else "n/a"
    out += ["", "## Reading (47 patients, tile mode unless stated)", "",
            f"- Gland-level labels do not by themselves predict a fill-dominated error. SICAPv2 (gland-level: gap fill "
            f"{s.loc['gap fill at 40 um', 'SICAPv2']}) transfers with its error spread over missed regions "
            f"({sh(MODELS['SICAPv2'], 'missed')}), false tumour ({sh(MODELS['SICAPv2'], 'false')}), fill "
            f"({sh(MODELS['SICAPv2'], 'fill')}) and boundary ({sh(MODELS['SICAPv2'], 'boundary')}).",
            f"- Coarse region labels that spill onto glass go with over-painting: Karolinska has almost no missed tumour "
            f"({sh(MODELS['PANDA-Karolinska'], 'missed')}) and little fill error ({sh(MODELS['PANDA-Karolinska'], 'fill')}); "
            f"false tumour is {sh(MODELS['PANDA-Karolinska'], 'false')} (tile) / {sh(MODELS['PANDA-Karolinska'], 'false', 'full')} "
            "(full) of its error.",
            f"- Radboud (the most gland-tight labels) is fill-dominated: fill {sh(MODELS['PANDA-Radboud'], 'fill')}, false "
            f"tumour {sh(MODELS['PANDA-Radboud'], 'false')} (radboud_prediction.md, Result section).",
            f"- Ours (site CV) has the most even mix: missed {sh(SITE, 'missed')}, false {sh(SITE, 'false')}, fill "
            f"{sh(SITE, 'fill')}, boundary {sh(SITE, 'boundary')}.", ""]
    (OUT / "labels_vs_errors.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
