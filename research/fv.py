"""Fair value = expected settlement given nowcast state, Monte Carlo over training-storm library."""
import numpy as np
from damage import *
from value import *
from feats import TICKS, NT
def ev_batch(h, cum, clo, L, nfut, Flib, Clib, pi, pj):
    """vectorised over leading dim N. h[N,NS], cum/clo[N,5], L[N]. returns mean value [N,10] and sd [N,10]"""
    if nfut == 0:
        v = value(cum, clo, L); return v, np.zeros_like(v)
    if nfut == 1:
        F1 = Flib[pi]; C1 = company_closure(Clib[pi])           # [M,NS],[M,5]
        add = h[:, None, :]*F1[None]
        d = 100*np.stack([np.sum((SW*add)[..., SC == j], -1) for j in range(5)], -1)
        vs = value(cum[:, None]+d, clo[:, None]+C1[None], L[:, None]+d@LOSS_B)
    else:
        F1 = Flib[pi]; F2 = Flib[pj]; C12 = company_closure(Clib[pi]+Clib[pj])
        add1 = h[:, None, :]*F1[None]; add2 = (h[:, None, :]-add1)*F2[None]
        d = 100*np.stack([np.sum((SW*(add1+add2))[..., SC == j], -1) for j in range(5)], -1)
        vs = value(cum[:, None]+d, clo[:, None]+C12[None], L[:, None]+d@LOSS_B)
    return vs.mean(1), vs.std(1)
if __name__ == '__main__':
    a = np.load('arch.npz'); lib = np.load('library.npz'); nc = np.load('nowcast.npz'); fr = np.load('feats_raw.npz')
    Fr = lib['Fr']; CLO = lib['CLO']; F = lib['F']
    p = np.load('damage_params.npy'); sig, thr, pw, k, fmax = unpack(p)
    Dhat, Chat, H0, CUM0, CLO0, L0 = [nc[x] for x in ['Dhat', 'Chat', 'H0', 'CUM0', 'CLO0', 'L0']]
    Flib = Fr[:350].reshape(-1, NS); Clib = CLO[:350].reshape(-1, NS)
    rng = np.random.default_rng(7); M = 600
    pi = rng.integers(0, len(Flib), M); pj = rng.integers(0, len(Flib), M)
    # site fractions of this storm: cone-variant shape, rescaled to the nowcast company damage
    Ecn = fr['E_now'] + fr['E_cone'][..., :NS]
    fsite = fmax[SC]*(1-np.exp(-k[SC]*Ecn))
    modc = 100*np.stack([np.sum((SW*H0[:, :, None]*fsite)[..., SC == j], -1) for j in range(5)], -1)
    scale = np.where(modc > 1e-6, Dhat/np.maximum(modc, 1e-9), 0)
    fsite = np.clip(fsite*scale[..., SC], 0, 1)
    FV = np.zeros(Dhat.shape[:-1]+(10,)); SD = np.zeros_like(FV)
    for r in range(3):
        hh = (H0[:, r, None]*(1-fsite[:, r])).reshape(-1, NS)
        cum = (CUM0[:, r, None]+Dhat[:, r]).reshape(-1, 5); clo = (CLO0[:, r, None]+Chat[:, r]).reshape(-1, 5)
        Ls = (L0[:, r, None]+Dhat[:, r]@LOSS_B).reshape(-1)
        out_m = []; out_s = []
        for c0 in range(0, len(hh), 2000):
            sl = slice(c0, c0+2000)
            m_, s_ = ev_batch(hh[sl], cum[sl], clo[sl], Ls[sl], 2-r, Flib, Clib, pi, pj)
            out_m.append(m_); out_s.append(s_)
        FV[:, r] = np.concatenate(out_m).reshape(600, NT, 10); SD[:, r] = np.concatenate(out_s).reshape(600, NT, 10)
        print('round', r+1, 'done', flush=True)
    np.savez('fv.npz', FV=FV, SD=SD)
