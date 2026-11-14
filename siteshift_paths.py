"""Central configuration of every location the SiteShift scripts read or write.

Set these environment variables (see setup_env.sh) or accept the defaults, which are folders inside this repository:

    SITESHIFT_DATA     folder that holds the datasets:  <DATA>/dataset47/<slide>/{images,masks}/ (our patients, obtained as described in
                       README.md), <DATA>/sicapv2/, <DATA>/karolinska_all/, <DATA>/radboud/ (public data, converted by data/prepare_*.py)
    SITESHIFT_RAW      folder with the raw public downloads: <RAW>/SICAPv2_Kaggle, <RAW>/prostate-cancer-grade-assessment/{Panda,Karolinska}
    SITESHIFT_RUNS     training / evaluation outputs (checkpoints, logs, per-slide CSVs); default <repo>/runs
    SITESHIFT_PRED     protocol-format prediction folders: <PRED>/<model>/predictions/<fold>/<slide>.npz; default <repo>/predictions
    SITESHIFT_RESULTS  tables written by the report scripts; default <repo>/results
    SITESHIFT_OUT      figure / analysis outputs; default <repo>/results/analysis
    SITESHIFT_INDEX    folder with the tile indexes and splits (tiles47.csv, cv47/fold<k>.json, ...); default <repo>/data
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_DIRS = [ROOT / d for d in ("data", "train", "eval", "figures")]


def _env(name, default):
    return Path(os.environ.get(name) or default)


DATA_ROOT = _env("SITESHIFT_DATA", ROOT / "datasets")
RAW = _env("SITESHIFT_RAW", ROOT / "raw_downloads")
RUNS = _env("SITESHIFT_RUNS", ROOT / "runs")
PRED = _env("SITESHIFT_PRED", ROOT / "predictions")
RESULTS = _env("SITESHIFT_RESULTS", ROOT / "results")
OUT = _env("SITESHIFT_OUT", ROOT / "results" / "analysis")
INDEX = _env("SITESHIFT_INDEX", ROOT / "data")
EVAL = ROOT / "eval"

DATASET47 = DATA_ROOT / "dataset47"
SICAP_RAW = RAW / "SICAPv2_Kaggle"
PANDA_RAW = RAW / "prostate-cancer-grade-assessment"


def script(name):
    """Full path of a script of this repository by file name (the run_*.py drivers start train.py, evaluate.py, ... as sub-processes)."""
    for d in CODE_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)
