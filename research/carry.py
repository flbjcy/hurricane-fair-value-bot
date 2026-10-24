"""Strategy B: between-round fundamental carry. After each final damage report (rounds 1-2), compute the
full distribution of settlement values by Monte Carlo over future storms, then hold the mean-variance
portfolio (exact model covariance -> natural hedges) until settlement."""
import numpy as np
from ev import EV
from value import company_closure
from damage import NS
a = np.load('arch.npz'); P = a['P']; SET = a['SET']; D = a['DMG'][:, :, 1, :5]; L = a['DMG'][:, :, 1, 5]
lib = np.load('library.npz'); Fr = lib['Fr']; CLO = lib['CLO']
LIM = np.array([200, 300, 500, 400, 800, 500, 700, 600, 250, 200], float)
def states(s):
    h = np.ones(NS); cum = np.zeros(5); clo = np.zeros(5); Ls = 0.0; out = []
    for r in range(2):
        h = h*(1-Fr[s, r]); cum = cum+D[s, r]; clo = clo+company_closure(CLO[s, r]); Ls += L[s, r]
        out.append((h.copy(), cum.copy(), clo.copy(), Ls))
    return out
def run(seasons, ev, lam=0.05, cost=0.10, entry_row=377, rounds=(0, 1), bias=None):
    res = []
    for s in seasons:
        st = states(s); pnl = 0.0; pos = np.zeros(10)
        for r in rounds:
            smp = ev.samples(*st[r], 2-r)                    # [M,10] settlement scenarios
            mu = smp.mean(0); C = np.cov(smp.T) + np.diag(0.05**2*np.ones(10))
            px = P[s, r, entry_row, :10]
            e = mu - px
            if bias is not None: e = e - bias[r]
            e = np.sign(e)*np.maximum(np.abs(e)-cost, 0)
            q = np.clip(np.linalg.solve(C, e)/lam, -LIM, LIM).round()
            dq = q-pos; pnl -= dq@px + np.abs(dq).sum()*cost; pos = q
        pnl += pos@SET[s, :10]
        res.append(pnl)
    return np.array(res)
if __name__ == '__main__':
    ev = EV(Fr[:350], CLO[:350], n_pairs=1500)
    for lam in [0.002, 0.005, 0.01, 0.03]:
        for rounds in [(0, 1), (1,)]:
            r = run(range(350, 450), ev, lam, rounds=rounds)
            print(f'VAL lam={lam} rounds={rounds}: mean {r.mean():8.1f} median {np.median(r):8.1f} sd {r.std():8.1f} worst {r.min():8.1f} win {(r>0).mean():.2f}')
