"""Storm-outcome nowcast (this storm's final damage/closure per company) + fair value (expected settlement)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from damage import *
from value import *
from feats import TICKS, NT
a = np.load('arch.npz'); D = a['DMG'][:, :, 1, :5]; Dp = a['DMG'][:, :, 0, :5]; L = a['DMG'][:, :, 1, 5]; Lp = a['DMG'][:, :, 0, 5]
P = a['P']; DT = a['DT']
f = np.load('feats_raw.npz'); lib = np.load('library.npz'); Fr = lib['Fr']; CLO = lib['CLO']
p = np.load('damage_params.npy'); sig, thr, pw, k, fmax = unpack(p)
TRAIN = slice(0, 350)
# health / cumulative state at the start of each round, from earlier final reports (causal)
H0 = np.ones((600, 3, NS)); CUM0 = np.zeros((600, 3, 5)); CLO0 = np.zeros((600, 3, 5)); L0 = np.zeros((600, 3))
for r in range(1, 3):
    H0[:, r] = H0[:, r-1]*(1-Fr[:, r-1]); CUM0[:, r] = CUM0[:, r-1]+D[:, r-1]
    CLO0[:, r] = CLO0[:, r-1]+company_closure(CLO[:, r-1]); L0[:, r] = L0[:, r-1]+L[:, r-1]
def comp(Fsite, h):  # site storm fractions -> company damage %
    return 100*np.stack([np.sum((SW*h*Fsite)[..., SC == j], -1) for j in range(5)], -1)
def frac(E): return fmax[SC]*(1-np.exp(-k[SC]*E))
h = H0[:, :, None, :]
En = f['E_now']; Cn = f['C_now']
V = {}
V['now'] = comp(frac(En), h); V['cv'] = comp(frac(En+f['E_cv'][..., :NS]), h)
V['cv60'] = comp(frac(En+f['E_cv60'][..., :NS]), h); V['cone'] = comp(frac(En+f['E_cone'][..., :NS]), h)
C = {kk: company_closure(Cn + (0 if kk == 'now' else f[{'cv': 'E_cv', 'cv60': 'E_cv60', 'cone': 'E_cone'}[kk]][..., NS:])) for kk in ['now', 'cv', 'cv60', 'cone']}
age = np.clip(TICKS-60, 0, 480)
prelim_av = (TICKS >= 570)
def design(j):
    cols = [V[kk][..., j] for kk in V] + [C[kk][..., j] for kk in C]
    cols += [np.broadcast_to(age, V['now'].shape[:-1]).astype(float)]
    cols += [f['hc'][..., q] for q in range(4)] + [f['outl'][..., q] for q in range(6)] + [f['wire'][..., q] for q in range(6)]
    cols += [f['st'][..., q] for q in range(6)] + [f['imp'][..., q] for q in range(4)]
    pre = np.where(prelim_av[None, None, :], Dp[:, :, None, j], np.nan)
    cols += [pre, np.broadcast_to(H0[:, :, None, SC == j].mean(-1), pre.shape)]
    return np.stack(cols, -1)
def fit_predict():
    Dhat = np.zeros(V['now'].shape); Chat = np.zeros(V['now'].shape)
    Cfin = company_closure(CLO)
    for j in range(5):
        X = design(j)
        for tgt, out in [(D[..., j], Dhat), (Cfin[..., j], Chat)]:
            y = np.broadcast_to(tgt[:, :, None], X.shape[:-1])
            m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
            m.fit(X[TRAIN].reshape(-1, X.shape[-1]), y[TRAIN].ravel())
            out[..., j] = np.maximum(m.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1]), 0)
    # after the final report the damage is known exactly
    fin = TICKS >= 740
    Dhat[:, :, fin] = D[:, :, None]
    return Dhat, Chat
if __name__ == '__main__':
    Dhat, Chat = fit_predict()
    np.savez('nowcast.npz', Dhat=Dhat, Chat=Chat, H0=H0, CUM0=CUM0, CLO0=CLO0, L0=L0)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    Cfin = company_closure(CLO)
    for nm, sl in [('val', slice(350, 450)), ('test', slice(450, 600))]:
        for i in [0, 5, 11, 17, 23, 29, 35, 41, 47, 50, 53, 60]:
            e = np.abs(Dhat[sl, :, i]-D[sl]).mean((0, 1)); e0 = np.abs(V['now'][sl, :, i]-D[sl]).mean((0, 1))
            ec = np.abs(Chat[sl, :, i]-Cfin[sl]).mean((0, 1))
            print(nm, 'T', TICKS[i], 'MAE dmg nowcast', e, ' so-far', e0, ' closure MAE', ec)
