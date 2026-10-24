"""Strategy A - Storm Fair-Value Convergence (FVC), laddered checkpoints, risk-budgeted, pre-committed exits."""
import numpy as np
from feats import TICKS, NT
a = np.load('arch.npz'); P = a['P']; SET = a['SET']
LIM = np.array([200, 300, 500, 400, 800, 500, 700, 600, 250, 200], float)
SYM = ['PORT', 'ISLE', 'AQUA', 'GRID', 'FERRY', 'RECON', 'MAT', 'GEN', 'HAVEN', 'REIN']
TI = TICKS//2
def calibrate(FV, dev, exit_row=272, delay=4):
    """Per checkpoint & stock: expected move to exit = beta*gap; residual covariance. Fitted on dev seasons only."""
    dr = delay//2
    G = FV - P[:, :, TI, :10]
    y = P[:, :, exit_row, :10][:, :, None] - P[:, :, np.minimum(TI+dr, 389), :10]
    beta = np.zeros((NT, 10)); Sig = np.zeros((NT, 10, 10))
    for i in range(NT):
        x = G[dev, :, i].reshape(-1, 10); yy = y[dev, :, i].reshape(-1, 10)
        beta[i] = (x*yy).sum(0)/np.maximum((x*x).sum(0), 1e-9)
    k = np.ones(5)/5
    beta = np.clip(np.array([np.convolve(np.pad(beta[:, j], 2, mode='edge'), k, 'valid') for j in range(10)]).T, 0, 1.2)
    for i in range(NT):
        R = (y[dev, :, i]-beta[i]*G[dev, :, i]).reshape(-1, 10); C = np.cov(R.T)
        Sig[i] = 0.7*C+0.3*np.diag(np.diag(C))
    gsd = G[dev].std((0, 1))            # typical gap size per tick/stock (for the sanity guard)
    return dict(beta=beta, Sig=Sig, gsd=gsd)
DEFAULT = dict(checkpoints=(190, 230, 270, 310, 350), frac=0.2, z=1.0, band=0.25, cost=0.10, delay=4,
               exit='storm_end', risk_budget=None, guard_sigma=5.0, size=1.0, stocks=tuple(range(10)), flag=None, cut=1.0, Pover=None, skip_prob=0.0, adverse=0.0, seed=0, breach=None, widen_stop=None, round_stop=None)
def simulate(FV, cal, seasons, **kw):
    o = dict(DEFAULT); o.update(kw)
    beta, Sig, gsd = cal['beta'], cal['Sig'], cal['gsd']
    dr = o['delay']//2; c = o['cost']; ck = [int(np.searchsorted(TICKS, t)) for t in o['checkpoints']]
    mask = np.zeros(10); mask[list(o['stocks'])] = 1
    PP = P if o['Pover'] is None else o['Pover']
    rng = np.random.default_rng(o['seed'])
    rows = []; attrib = np.zeros((len(seasons), 10)); nlong = nshort = 0
    nbreach = 0
    for n, s in enumerate(seasons):
        cash = 0.0; eq = [0.0]; per = np.zeros(10); alive = mask.copy()
        for r in range(3):
            pos = np.zeros(10); g0 = None; frozen = np.zeros(10, bool); cash_round0 = cash
            for i in ck:
                px_obs = PP[s, r, TI[i], :10]; g = FV[s, r, i]-px_obs
                if rng.random() < o['skip_prob']:
                    continue                      # stale data / rejected entry: no action this checkpoint
                if g0 is None: g0 = g.copy()
                if o['widen_stop'] is not None:
                    # gap has widened against the first signal by > k typical gaps: stop adding to that stock
                    frozen |= (np.sign(g0)*(g-g0) > o['widen_stop']*gsd[i]) & (np.abs(g0) > 0)
                e = beta[i]*g
                ok = (np.abs(e) > 2*c+o['band']) & (np.abs(g) < o['guard_sigma']*gsd[i]) & (alive > 0) & ~frozen
                sd = np.sqrt(np.diag(Sig[i]))
                tr = np.where(ok, np.clip(e/(o['z']*sd), -1, 1), 0)*LIM*o['frac']*o['size']
                if o['flag'] is not None and o['flag'][s, r, i]:
                    tr = 0*tr; pos_t = np.round(pos*o['cut'])
                else: pos_t = pos
                new = np.clip(pos_t+tr, -LIM, LIM)
                if o['risk_budget'] is not None:
                    risk = np.sqrt(new@Sig[i]@new)
                    if risk > o['risk_budget']:
                        # shrink the new tranche so the total stays within the risk budget
                        lo, hi = 0.0, 1.0
                        for _ in range(20):
                            m = (lo+hi)/2; cand = pos+m*(new-pos)
                            lo, hi = (m, hi) if np.sqrt(cand@Sig[i]@cand) <= o['risk_budget'] else (lo, m)
                        new = pos+lo*(new-pos)
                new = np.round(new); dq = new-pos
                if np.any(dq):
                    fill = PP[s, r, min(TI[i]+dr, 389), :10] + o['adverse']*np.abs(g)*np.sign(dq)
                    cash -= dq@fill + np.abs(dq).sum()*c; per -= dq*fill + np.abs(dq)*c
                    nlong += int((dq > 0).sum()); nshort += int((dq < 0).sum())
                pos = new
                eq.append(cash+pos@PP[s, r, min(TI[i]+dr, 389), :10])
            # optional hold-period stop: round mark-to-market loss beyond round_stop -> exit early
            stopped = False
            if o['round_stop'] is not None and np.any(pos):
                start_cash = cash_round0
                for T in range(400, 550, 10):
                    mtm = cash - start_cash + pos@PP[s, r, T//2, :10]
                    if mtm < -o['round_stop']:
                        ex = PP[s, r, min((T+o['delay'])//2, 389), :10]
                        cash += pos@ex - np.abs(pos).sum()*c; per += pos*ex - np.abs(pos)*c; pos = np.zeros(10); stopped = True; break
            # pre-committed exit
            if stopped: pass
            elif o['exit'] == 'storm_end' or r < 2:
                ex = PP[s, r, min((550+o['delay'])//2, 284) if o['exit'] == 'storm_end' else 376, :10]
                cash += pos@ex - np.abs(pos).sum()*c; per += pos*ex - np.abs(pos)*c
            else:
                cash += pos@SET[s, :10]; per += pos*SET[s, :10]
            if o['breach'] is not None:
                # model-breach rule: after the storm, our fair value should agree with the house; if a stock's
                # gap is still abnormally large, our model of that company is presumed broken -> disable it.
                iend = int(np.searchsorted(TICKS, 540))
                zb = np.abs(FV[s, r, iend]-PP[s, r, TI[iend], :10])/gsd[iend]
                newly = (zb > o['breach']) & (alive > 0); nbreach += int(newly.sum()); alive[newly] = 0
            eq.append(cash)
        e = np.array(eq); dd = np.max(np.maximum.accumulate(e)-e)
        rows.append((cash, dd)); attrib[n] = per
    return np.array(rows), attrib, (nlong, nshort, nbreach)
def summary(res, name=''):
    p = res[:, 0]
    return (f"{name:34s} mean {p.mean():8.0f}  median {np.median(p):8.0f}  worst {p.min():8.0f}  "
            f"5%ile {np.percentile(p,5):8.0f}  win {100*(p>0).mean():4.0f}%  sd {p.std():7.0f}  maxDD {res[:,1].max():7.0f}")
