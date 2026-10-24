"""Causal nowcast features on a 10-second decision grid. Everything at decision time T (real seconds
since round open) uses only messages with timestamp <= T: storm obs age <= T-60, advisories age <= T-60,
damage reports time <= T, prices rows <= T/2."""
import numpy as np
from damage import *
from value import *
a = np.load('arch.npz'); S = a['S']; ADV = a['ADV']; CONE = a['CONE']; WIRE = a['WIRE']; IMP = a['IMP']
DMG = a['DMG']; DT = a['DT']; P = a['P']
p = np.load('damage_params.npy'); sig, thr, pw, k, fmax = unpack(p)
TICKS = np.arange(70, 741, 10)                      # decision times (real s), 68 per round
NT = len(TICKS)
def proj_damage(xs, ys, R, I, mask):
    """xs,ys [...,H] path; R,I [...] ; returns exposure increment per site [...,NS]"""
    d = np.hypot(xs[..., None]-SX, ys[..., None]-SY)/R[..., None, None]
    w = I[..., None, None]*np.exp(-.5*(d/sig)**2)
    wc = I[..., None, None]*np.exp(-.5*(d/0.70)**2)
    return np.concatenate([(np.maximum(w-thr, 0)**pw*mask[..., None]).sum(-2)*2, ((wc > 0.34)*mask[..., None]).sum(-2)*2.0], -1)
def season_feats(s):
    out = {}
    St = S[s]                                                   # [3,241,6]
    u, I = site_u(St)
    w = I[..., None]*np.exp(-.5*(u/sig)**2)
    Ecum = np.cumsum(np.nan_to_num(np.maximum(w-thr, 0)**pw), 1)*2     # [3,241,NS] exposure up to age
    Ccum = np.cumsum(np.nan_to_num(w > 0.34), 1)*2.0                   # closure secs (0.70 kernel approx)
    w2 = I[..., None]*np.exp(-.5*(u/0.70)**2); Ccum = np.cumsum(np.nan_to_num(w2 > 0.34), 1)*2.0
    age = np.clip(TICKS-60, 0, 480)
    ai = age//2
    E_now = Ecum[:, ai]; C_now = Ccum[:, ai]                            # [3,NT,NS]
    st = St[:, ai]                                                      # [3,NT,6]
    x, y, vx, vy, R, In = [st[..., q] for q in range(6)]
    # constant-velocity projection to end of storm (age 480), and 60s
    tau = np.arange(2, 482, 2.)
    remain = (age[None, :, None]+tau[None, None, :]) <= 480
    remain = np.broadcast_to(remain, (3, NT, len(tau)))
    xs = x[..., None]+vx[..., None]*tau; ys = y[..., None]+vy[..., None]*tau
    E_cv = proj_damage(xs, ys, R, In, remain.astype(float))
    E_cv60 = proj_damage(xs, ys, R, In, (remain & (tau <= 60)).astype(float))
    # advisory cone path (latest advisory available)
    n_adv = np.clip(((age-30)//60).astype(int), -1, 7)                  # index of latest advisory
    E_cone = np.zeros(E_now.shape[:-1]+(2*NS,)); hc = np.zeros((3, NT, 4)); outl = np.zeros((3, NT, 6))
    for r in range(3):
        for i in range(NT):
            n = n_adv[i]
            if n < 0: continue
            hc[r, i] = ADV[s, r, n, 7:11]; outl[r, i] = ADV[s, r, n, 1:7]
            cp = CONE[s, r, n]; cp = cp[~np.isnan(cp[:, 0])]
            ag = np.r_[age[i], cp[:, 0]]; px = np.r_[x[r, i], cp[:, 1]]; py = np.r_[y[r, i], cp[:, 2]]; sp = np.r_[0, cp[:, 3]]
            if len(ag) < 2: continue
            fut = np.arange(age[i]+2, 481, 2.)
            if len(fut) == 0: continue
            # extrapolate beyond the last cone point with the last segment's velocity
            vxl = (px[-1]-px[-2])/(ag[-1]-ag[-2]); vyl = (py[-1]-py[-2])/(ag[-1]-ag[-2])
            fx = np.where(fut <= ag[-1], np.interp(fut, ag, px), px[-1]+vxl*(fut-ag[-1]))
            fy = np.where(fut <= ag[-1], np.interp(fut, ag, py), py[-1]+vyl*(fut-ag[-1]))
            fs = np.interp(fut, ag, sp)
            acc = 0
            for off in (-1, 0, 1):
                acc = acc + proj_damage(fx+off*fs, fy, np.array(R[r, i]), np.array(In[r, i]), np.ones(len(fut)))
            E_cone[r, i] = acc/3
    # wire: latest rumor outlook by T
    wire = np.zeros((3, NT, 6))
    for q in range(40):
        if np.isnan(WIRE[s, q, 0]): break
        r = int(WIRE[s, q, 0]); ag_ = WIRE[s, q, 1]
        for i in range(NT):
            if age[i] >= ag_: wire[r, i] = 0; wire[r, i, int(WIRE[s, q, 2])] = 1
    imp = np.nan_to_num(IMP[s][:, None, :] <= age[None, :, None]).astype(float)   # [3,NT,4]
    rep = np.stack([(DT[s, :, 0][:, None] <= TICKS[None]), (DT[s, :, 1][:, None] <= TICKS[None])], -1).astype(float)
    out = dict(E_now=E_now, C_now=C_now, E_cv=E_cv, E_cv60=E_cv60, E_cone=E_cone, st=st, hc=hc, outl=outl,
               wire=wire, imp=imp, rep=rep)
    return out
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(2) as pool: res = pool.map(season_feats, range(600))
    np.savez_compressed('feats_raw.npz', **{k: np.stack([r[k] for r in res]) for k in res[0]})
    print('saved')
