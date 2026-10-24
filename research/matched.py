"""Strategy A squeezed into the previous candidate's exact risk box: 10 shares/entry, <=2 open stocks,
$3,000 gross cap, $0.10/side, same seasons 451-600. Isolates signal quality from position size."""
import numpy as np
from fvc import *
FV = np.load('fv_cf.npz')['FV']; dev = np.arange(450); test = np.arange(450, 600)
cal = calibrate(FV, dev, exit_row=272, delay=4)
beta, Sig = cal['beta'], cal['Sig']
def run(seasons, cost=0.10, delay=4, band=0.25, cks=(150, 190, 230, 270, 310, 350, 390)):
    dr = delay//2; out = []
    for s in seasons:
        pnl = 0.0
        for r in range(3):
            held = {}
            for t in cks:
                i = int(np.searchsorted(TICKS, t)); g = FV[s, r, i]-P[s, r, TI[i], :10]; e = beta[i]*g
                sc = np.abs(e)/np.sqrt(np.diag(Sig[i]))
                for j in np.argsort(-sc):
                    if len(held) >= 2: break
                    if j in held or abs(e[j]) <= 2*cost+band: continue
                    fill = P[s, r, TI[i]+dr, j]
                    if sum(10*P[s, r, TI[i]+dr, k] for k in held)+10*fill > 3000: continue
                    held[j] = (np.sign(e[j])*10, fill)
            for j, (q, f0) in held.items():
                pnl += q*(P[s, r, 272, j]-f0) - 2*cost*abs(q)
        out.append(pnl)
    return np.array(out)
for nm, kw in [('base', {}), ('cost .25', dict(cost=0.25)), ('delay 10s', dict(delay=10))]:
    p = run(test, **kw)
    print(f'TEST matched-size A {nm:10s}: mean {p.mean():7.2f} median {np.median(p):7.2f} worst {p.min():8.2f} win {100*(p>0).mean():.1f}%')
