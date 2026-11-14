"""
Smoke test on synthetic masks; needs no patient data and runs in well under a minute:

    python tests/smoke_test.py

1. The reference attribution (eval/attribution.py) on the ten demo scenarios of eval/scenarios.py: the identity
       missed + false + fill + boundary = FP + FN
   holds exactly in integer pixels in every scenario, and each scenario lands in the bin the construction implies.
2. If torch is installed: the pipeline code of the paper (eval/protocol.py, tiles stitched on a grid) on a synthetic
   slide, written in the protocol format: the same identity, FP = false + fp_band, and the same bins as the reference
   attribution computed on the whole synthetic slide.
Exit code 0 = all checks passed.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in ("eval", "train", "data", "."):
    sys.path.insert(0, str(ROOT / d))

import attribution as A  # noqa: E402
import scenarios as SC  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1. reference attribution on the demo scenarios ---------------------------------------------------------------
G = SC.G
res = {n: A.attribution(G, P) for n, P in SC.scenarios().items()}
for n, a in res.items():
    check(a["exact"] and sum(a["bins"].values()) == a["FP"] + a["FN"], f"identity missed+false+fill+boundary = FP+FN: {n}")
check(res["identical"]["dice"] == 1.0 and sum(res["identical"]["bins"].values()) == 0, "identical masks: Dice 1, every bin 0")
m = res["miss one region"]
check(m["bins"]["miss"] > 0 and m["n_det"] < m["n_reg"] and m["s"]["false"] == 0, "deleting a region is a detection error (missed bin)")
f = res["far false blob"]
check(f["bins"]["false"] > 0 and f["bins"]["miss"] == 0 and f["bins"]["fill"] == 0, "a blob far from the tumour is a detection error (false bin)")
for n in ("erode 25 um", "dilate 25 um"):
    check(res[n]["dele"] > res[n]["det"], f"a shrunk or grown outline (within the 50 um band) is mostly delineation: {n}")
d120 = res["dilate 120 um"]
check(d120["bins"]["false"] > 0 and d120["bins"]["bnd"] > 0, "growing the outline by 120 um: the first 50 um is boundary, the ring beyond the band is false tumour (detection)")
e120 = res["erode 120 um"]
check(e120["dice"] == 0.0 and e120["bins"]["miss"] == e120["FN"], "shrinking by 120 um removes every region: all of it is missed")
check(res["shift R1 by 150 um"]["det"] > 0 and res["shift R1 by 150 um"]["dele"] > 0, "a region shifted by 150 um costs both kinds of error")

# ---- 2. the paper's pipeline on a synthetic stitched slide --------------------------------------------------------
try:
    import torch  # noqa: F401
    import protocol
except Exception as e:      # protocol.py imports the data module, which needs torch
    print(f"skip  pipeline check ({type(e).__name__}: {e}); install the requirements to run it")
else:
    S, UM = 256, 0.5                     # 2 x 2 tiles of 256 px = a 256 x 256 um slide
    N = 2 * S

    def disk(cy, cx, r):
        y, x = np.ogrid[:N, :N]
        return (y - cy) ** 2 + (x - cx) ** 2 <= r * r

    T = disk(250, 260, 90) | disk(400, 120, 12)             # one region across the tile corner, one small region (< 2,500 um^2)
    preds = {"eroded": A.erode_um(T, 15), "dilated": A.dilate_um(T, 15), "with false blob": T | disk(60, 60, 25), "shifted": np.roll(T, 40, axis=1)}

    with tempfile.TemporaryDirectory() as tmp:
        for name, Pm in preds.items():
            imgs, pk = [], {"pred": [], "truth": [], "tissue": []}
            for r in range(2):
                for c in range(2):
                    sl = (slice(r * S, (r + 1) * S), slice(c * S, (c + 1) * S))
                    imgs.append(f"synthetic/images/synthetic [d=1,x={c * S},y={r * S},w={S},h={S}].png")
                    for k, a_ in (("pred", Pm[sl]), ("truth", T[sl]), ("tissue", np.ones((S, S), bool))):
                        pk[k].append(np.packbits(a_.reshape(-1)))
            f_ = Path(tmp) / f"{name.replace(' ', '_')}.npz"
            np.savez_compressed(f_, slide="synthetic", size=S, image=np.array(imgs), **{k: np.stack(v) for k, v in pk.items()})
            _, s, _ = protocol.analyse_slide(f_, UM, True)
            ref = A.attribution(T, Pm)
            bnd = s["fn_band"] + s["fp_band"]
            check(s["missed"] + s["fill_px"] + s["fn_band"] == s["fn"] and s["false"] + s["fp_band"] == s["fp"],
                  f"pipeline identity (fn and fp fully binned): {name}")
            got = dict(miss=s["missed"], false=s["false"], fill=s["fill_px"], bnd=bnd)
            check(got == ref["bins"], f"pipeline bins equal the reference attribution: {name}  {got} vs {ref['bins']}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
