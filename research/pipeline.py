"""Honest cross-fitted fair values. Dev = seasons 1-450 (5 folds of 90, out-of-fold predictions),
Test = seasons 451-600 predicted by models fitted on all of 1-450 only."""
import numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
import nowcast as N
from fv import ev_batch
from damage import NS, SW, SC
from value import LOSS_B, company_closure
from feats import NT
fr = N.f; Fr = N.Fr; CLO = N.CLO
Cfin = company_closure(CLO)
X_all = [N.design(j) for j in range(5)]
def nowcast(train_idx, pred_idx):
    Dh = np.zeros((len(pred_idx), 3, NT, 5)); Ch = np.zeros_like(Dh)
    for j in range(5):
        X = X_all[j]
        for tgt, out in [(N.D[..., j], Dh), (Cfin[..., j], Ch)]:
            y = np.broadcast_to(tgt[:, :, None], X.shape[:-1])
            m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            m.fit(X[train_idx].reshape(-1, X.shape[-1]), y[train_idx].ravel())
            out[..., j] = np.maximum(m.predict(X[pred_idx].reshape(-1, X.shape[-1])).reshape(len(pred_idx), 3, NT), 0)
    fin = N.TICKS >= 740
    Dh[:, :, fin] = N.D[pred_idx][:, :, None]
    return Dh, Ch
def fair_value(train_idx, pred_idx, Dh, Ch, M=600, seed=7):
    Flib = Fr[train_idx].reshape(-1, NS); Clib = CLO[train_idx].reshape(-1, NS)
    rng = np.random.default_rng(seed); pi = rng.integers(0, len(Flib), M); pj = rng.integers(0, len(Flib), M)
    H0, CUM0, CLO0, L0 = N.H0[pred_idx], N.CUM0[pred_idx], N.CLO0[pred_idx], N.L0[pred_idx]
    Ecn = fr['E_now'][pred_idx] + fr['E_cone'][pred_idx][..., :NS]
    fs = N.fmax[SC]*(1-np.exp(-N.k[SC]*Ecn))
    modc = 100*np.stack([np.sum((SW*H0[:, :, None]*fs)[..., SC == j], -1) for j in range(5)], -1)
    fs = np.clip(fs*np.where(modc > 1e-6, Dh/np.maximum(modc, 1e-9), 0)[..., SC], 0, 1)
    n = len(pred_idx); FV = np.zeros((n, 3, NT, 10)); SD = np.zeros_like(FV)
    for r in range(3):
        hh = (H0[:, r, None]*(1-fs[:, r])).reshape(-1, NS)
        cum = (CUM0[:, r, None]+Dh[:, r]).reshape(-1, 5); clo = (CLO0[:, r, None]+Ch[:, r]).reshape(-1, 5)
        Ls = (L0[:, r, None]+Dh[:, r]@LOSS_B).reshape(-1)
        ms, ss = [], []
        for c0 in range(0, len(hh), 2000):
            sl = slice(c0, c0+2000); m_, s_ = ev_batch(hh[sl], cum[sl], clo[sl], Ls[sl], 2-r, Flib, Clib, pi, pj); ms.append(m_); ss.append(s_)
        FV[:, r] = np.concatenate(ms).reshape(n, NT, 10); SD[:, r] = np.concatenate(ss).reshape(n, NT, 10)
    return FV, SD
if __name__ == '__main__':
    FV = np.zeros((600, 3, NT, 10)); SD = np.zeros_like(FV); DH = np.zeros((600, 3, NT, 5)); CH = np.zeros_like(DH)
    dev = np.arange(450)
    splits = [(np.setdiff1d(dev, np.arange(k*90, (k+1)*90)), np.arange(k*90, (k+1)*90)) for k in range(5)]
    splits.append((dev, np.arange(450, 600)))
    for tr, pr in splits:
        t0 = time.time(); Dh, Ch = nowcast(tr, pr); FV[pr], SD[pr] = fair_value(tr, pr, Dh, Ch); DH[pr], CH[pr] = Dh, Ch
        print('fold', pr[0], 'done', round(time.time()-t0), 's', flush=True)
    np.savez('fv_cf.npz', FV=FV, SD=SD, DH=DH, CH=CH)
