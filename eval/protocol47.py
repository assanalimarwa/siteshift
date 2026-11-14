"""
Detection / delineation analysis (protocol.py) of the 47-patient predictions exported by protocol_predict.py, for the RQ6 gaps.

    python protocol47.py nnunet nnunet_allpublic     # analyse these exports (CPU only)
    python protocol47.py --auto                      # every export whose prediction folders are all finished and not yet analysed,
                                                     # then cv47/rq6_table.md is rewritten (run by analysis_watch.sh)

Reads export/<key>/predictions/<fold or all>/ (done.json + one .npz per slide), analyses them with protocol.analyse (same code and
definitions as the local protocol analysis; tile mode, regions stitched on the tile grid) and writes cv47/protocol_<key>_by_patient.csv
(measures of protocol.measures per patient: Dice of the patient's pooled tiles, error bins err_missed / err_false (detection) and
err_fill / err_boundary (delineation) that add up to 1 - Dice, clean-tile AUC, recall, false regions, dice_found, ...).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P
import sys
import time
from pathlib import Path

import pandas as pd

import protocol

protocol.INDEX["ours"] = "tiles47.csv"                      # protocol.py names the 17-patient index; ours is tiles47.csv
from protocol import analyse, measures  # noqa: E402

import export_model as em  # noqa: E402

HERE = P.INDEX
EXPORT, CV = em.EXPORT, em.CV
KEYS = ["unet"] + list(em.MODELS) + list(em.ALLPUBLIC) + ["unet_allpublic"]
EXPECT = {k: (1 if k.endswith("_allpublic") else 5) for k in KEYS}


def folders(key):
    d = EXPORT / key / "predictions"
    return sorted(f.parent for f in d.glob("*/done.json")) if d.exists() else []


def csv_path(key):
    return CV / f"protocol_{key}_by_patient.csv"


def ready(key):
    return len(folders(key)) >= EXPECT[key]


def stale(key):
    out = csv_path(key)
    return not out.exists() or any(out.stat().st_mtime < (f / "done.json").stat().st_mtime for f in folders(key))


def run(key, workers=6):
    fs = folders(key)
    if len(fs) < EXPECT[key]:
        print(f"{key}: {len(fs)} of {EXPECT[key]} prediction folders finished, skipped")
        return False
    tis = em.tissue()
    sl, ti = [], []
    for f in fs:
        t0 = time.time()
        s, t = analyse(f, False, workers)
        sl.append(s)
        ti.append(t)
        print(f"{key} {f.name}: {len(s)} slides, {time.time() - t0:.0f} s", flush=True)
    m = measures(pd.concat(sl), pd.concat(ti), "ours", tis)
    m.index.name = "patient"
    m.round(5).to_csv(csv_path(key))
    print(f"{key}: written {csv_path(key)}", flush=True)
    return True


def main():
    args = sys.argv[1:]
    if "--auto" in args:
        keys = [k for k in KEYS if ready(k) and stale(k)]
    else:
        keys = [k for k in args if not k.startswith("--")]
    for k in keys:
        run(k)
    if "--auto" in args or keys:
        import rq6_report
        rq6_report.main()


if __name__ == "__main__":
    main()
