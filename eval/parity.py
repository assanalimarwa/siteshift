"""
Step 1 parity: our protocol.py (Euclidean band) against the reference attribution.py (not edited)
on the same real slide predictions. Each slide's tiles are stitched on their grid into one full-slide array (missing tiles = glass), G = annotation,
P = prediction; the reference runs on those arrays, our analyse_slide on the saved tile predictions. Bin counts are compared exactly.

    python parity.py            # 5 slides x (nnU-Net-style site CV, nnU-Net-style all-public)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import attribution as ref  # noqa: E402
import protocol  # noqa: E402
from data import grid_position  # noqa: E402

protocol.INDEX["ours"] = "tiles47.csv"
OUT = P.OUT
EXP = P.PRED
SLIDES = [("39.2.bif - Series 0", "fold3"), ("9.1.bif - Series 0", "fold0"), ("23.3.bif - Series 0", "fold2"),
          ("44.2.bif - Series 0", "fold4"), ("44.1.bif - Series 0", "fold4")]
MODELS = {"nnU-Net-style site CV": lambda fold: EXP / "nnunet" / "predictions" / fold,
          "nnU-Net-style all-public": lambda fold: EXP / "nnunet_allpublic" / "predictions" / "all"}


def stitch(path):
    z = np.load(path)
    S, images = int(z["size"]), z["image"]
    pos = np.array([grid_position(p) for p in images])
    c0, r0 = pos.min(0)
    w, h = pos[:, 0].max() - c0 + 1, pos[:, 1].max() - r0 + 1
    G, P = np.zeros((h * S, w * S), bool), np.zeros((h * S, w * S), bool)
    truth, pred = z["truth"], z["pred"]
    for i, (c, r) in enumerate(pos):
        y, x = (r - r0) * S, (c - c0) * S
        G[y:y + S, x:x + S] = np.unpackbits(truth[i], count=S * S).reshape(S, S).view(bool)
        P[y:y + S, x:x + S] = np.unpackbits(pred[i], count=S * S).reshape(S, S).view(bool)
    return G, P


def main():
    rows = []
    for mname, folder in MODELS.items():
        for slide, fold in SLIDES:
            path = folder(fold) / (re.sub(r"[^\w.-]+", "_", slide) + ".npz")
            t0 = time.time()
            _, ours, _ = protocol.analyse_slide(path, 0.5, True)
            t1 = time.time()
            G, P = stitch(path)
            t2 = time.time()
            a = ref.attribution(G, P)
            b = a["bins"]
            cmp = {"tp": (ours["tp"], a["TP"]), "fp": (ours["fp"], a["FP"]), "fn": (ours["fn"], a["FN"]),
                   "miss": (ours["missed"], b["miss"]), "false": (ours["false"], b["false"]), "fill": (ours["fill_px"], b["fill"]),
                   "boundary": (ours["fn_band"] + ours["fp_band"], b["bnd"]),
                   "regions": (ours["regions"], a["n_reg"]), "regions found": (ours["regions_found"], a["n_det"])}
            r = {"model": mname, "slide": slide, "Mpx": round(G.size / 1e6), "ref identity exact": a["exact"]}
            for k, (o, rf) in cmp.items():
                r[k + " ours"], r[k + " ref"], r[k + " diff"] = int(o), int(rf), int(o) - int(rf)
            r["all equal"] = all(o == rf for o, rf in cmp.values())
            rows.append(r)
            print(f"{mname} | {slide}: {'EXACT' if r['all equal'] else 'MISMATCH'}  (ours {t1 - t0:.0f} s, reference {time.time() - t2:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "parity.csv", index=False)
            del G, P
    df = pd.DataFrame(rows)
    print(df[["model", "slide", "all equal"] + [c for c in df.columns if c.endswith("diff")]].to_string(index=False))


if __name__ == "__main__":
    main()
