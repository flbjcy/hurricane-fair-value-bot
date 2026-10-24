"""Frozen Strategy A evaluated ONCE on test seasons 451-600 + planned stresses. Calibration uses dev 1-450 only."""
import numpy as np, json
from fvc import *
FV = np.load('fv_cf.npz')['FV']; dev = np.arange(450); test = np.arange(450, 600)
FROZEN = dict(checkpoints=(150, 190, 230, 270, 310, 350, 390), frac=1/7, z=1.0, band=0.25, exit='storm_end',
              risk_budget=1500, guard_sigma=5.0, cost=0.10, delay=4, breach=6.0, widen_stop=3.0)
cal = calibrate(FV, dev, exit_row=272, delay=4)
out = {}
def rec(name, res, extra=None):
    p = res[:, 0]; out[name] = dict(mean=float(p.mean()), median=float(np.median(p)), worst=float(p.min()),
        p5=float(np.percentile(p, 5)), win=float((p > 0).mean()), sd=float(p.std()), maxdd=float(res[:, 1].max()))
    if extra: out[name].update(extra)
    print(summary(res, name))
res, att, ls = simulate(FV, cal, test, **FROZEN)
rec('TEST frozen A (base)', res, dict(per_stock=dict(zip(SYM, att.mean(0).round(1).tolist())), long_trades=ls[0], short_trades=ls[1]))
np.savetxt('test_A_seasons.csv', np.c_[test+1, res], delimiter=',', header='season,pnl,maxdd', fmt='%.2f')
for nm, kw in [('cost $0.25/share/side', dict(cost=0.25)), ('cost $0.50/share/side', dict(cost=0.50)),
               ('delay 10s', dict(delay=10)), ('delay 20s', dict(delay=20)),
               ('half size', dict(size=0.5)), ('delay 2s', dict(delay=2)), ('no safety layer', dict(breach=None, widen_stop=None)), ('exit after final report', dict(exit='after_final'))]:
    k = dict(FROZEN); k.update(kw); rec('TEST '+nm, simulate(FV, cal, test, **k)[0])
# long-only vs short-only attribution
for side in ['long', 'short']:
    F2 = FV.copy(); G = FV-P[:, :, TI, :10]
    F2 = np.where((G > 0) if side == 'short' else (G < 0), P[:, :, TI, :10], FV)   # suppress the other side
    rec(f'TEST {side}-only', simulate(F2, cal, test, **FROZEN)[0])
json.dump(out, open('results_A.json', 'w'), indent=1)
