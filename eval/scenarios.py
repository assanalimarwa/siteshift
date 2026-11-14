"""Demo scenarios of the reference attribution (copied from the reference scenarios.py; the construction of G and of the ten predicted masks is unchanged,
wrapped so that importing it has no side effects and tests/smoke_test.py can reuse it).

    python eval/scenarios.py        # prints the table of the ten scenarios, writes scenarios.csv
"""
import numpy as np
from scipy import ndimage as ndi
from attribution import *

rng = np.random.default_rng(0)
H = W = 1600  # 800 um x 800 um

def blob(cy, cx, r_px, shape=(H, W)):
    m = np.zeros(shape, bool)
    y, x = np.ogrid[:shape[0], :shape[1]]
    m |= (y-cy)**2 + (x-cx)**2 <= r_px**2
    return m

# Ground truth: two big regions + one tiny (below 2,500 um^2)
R1 = blob(500, 450, 220)            # ~38,000 um^2*... area pi*110um^2 = 38,013 um2
R2 = blob(1100, 1050, 150)
R3 = blob(300, 1300, 25)            # ~490 um^2 -> ignored for counting
G = R1 | R2 | R3

def glands_inside(X, r_um=22, pitch_um=60):
    """gland-level style: keep only a lattice of small disks inside X (stroma removed)"""
    r = int(r_um/UM); p = int(pitch_um/UM)
    lat = np.zeros_like(X)
    for cy in range(r, X.shape[0], p):
        for cx in range(r, X.shape[1], p):
            if X[cy, cx]:
                lat |= blob(cy, cx, r, X.shape)
    return lat & X



def scenarios():
    scen = {}
    scen['identical']            = G.copy()
    scen['miss one region']      = R1 | R3                       # R2 deleted
    scen['far false blob']       = G | blob(250, 250, 90)        # ~300 um from R1
    scen['near false blob']      = G | blob(500, 450+220+int(30/UM)+6, 40)  # ~15-30 um from R1 edge... adjust below
    scen['erode 25 um']          = erode_um(G, 25)
    scen['erode 120 um']         = erode_um(G, 120)
    scen['dilate 25 um']         = dilate_um(G, 25)
    scen['dilate 120 um']        = dilate_um(G, 120)
    scen['gland-style (holes)']  = glands_inside(G)
    scen['shift R1 by 150 um']   = blob(500+int(150/UM), 450, 220) | R2 | R3
    return scen


if __name__ == "__main__":
    import pandas as pd
    rows = []
    for name, P in scenarios().items():
        a = attribution(G, P)
        rows.append(dict(scenario=name, dice=a['dice'], miss=a['s']['miss'], false=a['s']['false'],
                         fill=a['s']['fill'], bnd=a['s']['bnd'], det=a['det'], dele=a['dele'],
                         exact=a['exact'], regions=f"{a['n_det']}/{a['n_reg']}",
                         bias_um=signed_bias_um(G, P)))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(df.to_string(index=False))
    assert df.exact.all(), "identity violated"
    print("\nIdentity 1-Dice = miss+false+fill+bnd holds EXACTLY (integer pixel counts) in all scenarios.")
    df.to_csv('scenarios.csv', index=False)
