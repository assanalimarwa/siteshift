"""
protocol.py's stitched analysis (the parity-verified one; no per-tile sensitivity variant) of prediction folders, cached
next to the predictions as protocol_*_euclid.csv (4090, 2026-11-10):

    python protocol_cache47.py <prediction folder> ...
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import sys
import time

import protocol

protocol.INDEX["ours"] = "tiles47.csv"                      # as eval/protocol47.py: our 47 patients

if __name__ == "__main__":
    for f in sys.argv[1:]:
        if not protocol.is_cached(f, False):
            t0 = time.time()
            s, _ = protocol.analyse(f)
            print(f"{f}: {len(s)} slides, {time.time() - t0:.0f} s", flush=True)
