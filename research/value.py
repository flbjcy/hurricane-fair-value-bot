"""Exact settlement formulas recovered from the archive (residual sd < $0.01 except closure-driven terms)."""
import numpy as np
from damage import SW, SC, NS, site_u
SIG_W, THR_W = 0.67, 0.30      # wind kernel / hurricane-condition threshold (refined below)
LOSS_B = np.array([9.0, 6.0, 2.5, 7.0, 1.2])          # $M insured loss per damage-% point
def value(cum, clo, L):
    """cum[...,5] cumulative damage %, clo[...,5] site-weighted closed seconds, L season loss $M -> [...,10]"""
    P, I, A, G, F = [cum[..., k] for k in range(5)]
    cP, cG, cF = clo[..., 0], clo[..., 3], clo[..., 4]
    out = np.stack([
        100 - 0.55*P - 0.02*cP,
        60 - 0.38*I,
        35 - 0.20*A,
        50 - 0.24*G - 0.05*cG,
        25 - 0.05*F - 0.06*cF,
        40 + 24*(1 - np.exp(-(0.009*P + 0.007*G + 0.0012*F))),
        28 + 14*(1 - np.exp(-(0.0075*P + 0.005*I + 0.0021*A + 0.0058*G + 0.001*F))),
        32 + 18*(1 - np.exp(-0.0082*cG)),
        75 - 0.06*np.clip(L - 400, 0, 900),
        90 - 0.035*np.clip(L - 1300, 0, 700),
    ], -1)
    return out
B1 = np.array([1, 1, 2, 1, 2, 0, 0, 0, 1, 0.]); B2 = np.array([0, 0, 0, 0, 0, 2, 2, 1, 0, 0.])
def with_etfs(v): return np.concatenate([v, (v@B1)[..., None], (v@B2)[..., None]], -1)
def closure_seconds(S, sig=SIG_W, thr=THR_W):
    """per-site closed seconds per storm from observations: S[...,241,6] -> [...,NS] and cumulative over time"""
    u, I = site_u(S)
    w = I[..., None]*np.exp(-.5*(u/sig)**2)
    return np.nan_to_num(w > thr)*2.0
def company_closure(site_secs):
    return np.stack([(site_secs*SW)[..., SC == j].sum(-1) for j in range(5)], -1)
