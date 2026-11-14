import numpy as np
from scipy import ndimage as ndi

UM = 0.5  # um per pixel
MIN_REGION_UM2 = 2500.0
BAND_UM = 50.0
DET_FRAC = 0.10
S8 = np.ones((3,3), bool)

def within(mask, radius_um):
    """pixels within radius_um of mask (including mask)"""
    if not mask.any(): return np.zeros_like(mask)
    d = ndi.distance_transform_edt(~mask) * UM
    return d <= radius_um

def outline(G):
    er = ndi.binary_erosion(G, S8, border_value=0)
    return G & ~er

def attribution(G, P):
    G = G.astype(bool); P = P.astype(bool)
    TP = int((G & P).sum()); FP_m = P & ~G; FN_m = G & ~P
    FP = int(FP_m.sum()); FN = int(FN_m.sum())
    denom = 2*TP + FP + FN
    lab, n = ndi.label(G, structure=S8)
    areas = ndi.sum_array = ndi.sum(np.ones_like(lab), lab, index=range(1, n+1))
    marked = ndi.sum(P.astype(float), lab, index=range(1, n+1))
    D = np.zeros_like(G)
    n_det = n_reg = 0
    for i in range(1, n+1):
        big = areas[i-1] * UM*UM >= MIN_REGION_UM2
        det = marked[i-1] >= DET_FRAC * areas[i-1]
        if big:
            n_reg += 1; n_det += det
        if det and big:
            D |= (lab == i)
    M = G & ~D
    bG = within(G, BAND_UM)
    bdG = within(outline(G), BAND_UM)
    B_miss = FN_m & M
    B_false = FP_m & ~bG
    B_fill = (FN_m & D) & ~bdG
    E = FP_m | FN_m
    B_bnd = E & ~(B_miss | B_false | B_fill)
    bins = {k: int(v.sum()) for k, v in
            dict(miss=B_miss, false=B_false, fill=B_fill, bnd=B_bnd).items()}
    dice = 2*TP/denom if denom else 1.0
    s = {k: v/denom if denom else 0.0 for k, v in bins.items()}
    exact = (bins['miss']+bins['false']+bins['fill']+bins['bnd']) == (FP+FN)
    return dict(dice=dice, s=s, bins=bins, FP=FP, FN=FN, TP=TP, denom=denom,
                exact=exact, det=s['miss']+s['false'], dele=s['fill']+s['bnd'],
                n_reg=n_reg, n_det=n_det)

def disk(r_px):
    y, x = np.ogrid[-int(r_px):int(r_px)+1, -int(r_px):int(r_px)+1]
    return x*x + y*y <= r_px*r_px

def dilate_um(X, r_um):
    if r_um <= 0: return X.copy()
    return ndi.distance_transform_edt(~X)*UM <= r_um

def erode_um(X, r_um):
    if r_um <= 0: return X.copy()
    return ndi.distance_transform_edt(X)*UM > r_um

def close_um(X, r_um):
    return erode_um(dilate_um(X, r_um), r_um)

def signed_bias_um(G, P):
    """median signed distance from predicted boundary to annotation boundary; + outside G"""
    pb = outline(P)
    if not pb.any() or not G.any(): return np.nan
    d_out = ndi.distance_transform_edt(~G)*UM   # >0 outside G
    d_in  = ndi.distance_transform_edt(G)*UM    # >0 inside G
    sd = np.where(G, -d_in, d_out)
    return float(np.median(sd[pb]))
