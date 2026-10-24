"""ETF lag strategy: ETF house mark ~= basket mark 6 s earlier. When the live basket has moved away from the
ETF, take an ETF position in the basket's direction; exit after a fixed hold. Every 2 s decision."""
import numpy as np
a = np.load('arch.npz'); P = a['P']
b1 = np.array([1, 1, 2, 1, 2, 0, 0, 0, 1, 0.]); b2 = np.array([0, 0, 0, 0, 0, 2, 2, 1, 0, 0.])
def run_etf(seasons, thr=1.0, delay=4, hold=8, cost=0.10, lim=(50, 100), one_at_a_time=True):
    dr = delay//2; hr = hold//2; out = []
    for s in seasons:
        tot = 0.0; n = 0
        for e, (b, L) in enumerate([(b1, lim[0]), (b2, lim[1])]):
            X = P[s, :, :, 10+e]; B = P[s, :, :, :10]@b
            for r in range(3):
                t = 0
                while t < 390-dr-hr:
                    g = B[r, t]-X[r, t]
                    if abs(g) > thr:
                        q = np.sign(g)*L; ent = X[r, t+dr]; ex = X[r, t+dr+hr]
                        tot += q*(ex-ent) - 2*cost*L; n += 1
                        t += dr+hr if one_at_a_time else 1
                    else: t += 1
        out.append((tot, n))
    return np.array(out)
if __name__ == '__main__':
    for delay in [2, 4, 6]:
        for thr in [0.75, 1.0, 1.5, 2.5]:
            for hold in [6, 10]:
                r = run_etf(range(0, 450), thr, delay, hold)
                print(f'delay {delay}s thr {thr} hold {hold}s: mean/season {r[:,0].mean():8.1f} sd {r[:,0].std():7.1f} worst {r[:,0].min():8.1f} trades/season {r[:,1].mean():.1f}')
