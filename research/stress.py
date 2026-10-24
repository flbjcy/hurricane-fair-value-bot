"""Planned stresses on TEST (451-600) for frozen Strategy A, with and without the safety layer. All reported."""
import numpy as np, json
from fvc import *
FV = np.load('fv_cf.npz')['FV']; dev = np.arange(450); test = np.arange(450, 600)
cal = calibrate(FV, dev, exit_row=272, delay=4)
FROZEN = dict(checkpoints=(150, 190, 230, 270, 310, 350, 390), frac=1/7, z=1.0, band=0.25, exit='storm_end',
              risk_budget=1500, guard_sigma=5.0, cost=0.10, delay=4)
SAFE = dict(breach=6.0, widen_stop=3.0)
t = np.arange(390)*2.0
def drift(rate, start=400, cols=slice(0, 10)):
    Q = P.copy(); Q[:, :, :, cols] += rate*np.maximum(t-start, 0)[None, None, :, None]; return Q
def jump(pct, at=300):
    Q = P.copy(); Q[:, :, t >= at, :10] *= (1+pct); return Q
def port_regime(amp=0.5):
    Q = P.copy(); base = P[:, 1:2, :1, 0]; Q[:, 1:, :, 0] = P[:, 1:, :, 0] + amp*(P[:, 1:, :, 0]-base); return Q
rng = np.random.default_rng(1)
noisy = P.copy(); noisy[..., :10] *= 1+rng.normal(0, 0.003, noisy[..., :10].shape)
scen = [('none', {}), ('up drift +$0.02/s after 400s', dict(Pover=drift(0.02))), ('down drift -$0.02/s after 400s', dict(Pover=drift(-0.02))),
        ('all stocks +3% jump at 300s', dict(Pover=jump(0.03))), ('all stocks -3% jump at 300s', dict(Pover=jump(-0.03))),
        ('insurers drift -$0.03/s after 300s', dict(Pover=drift(-0.03, 300, slice(8, 10)))),
        ('PORT economics change from round 2 (unmodelled)', dict(Pover=port_regime(0.5))),
        ('quote noise 0.3%', dict(Pover=noisy)), ('30% checkpoints stale/rejected', dict(skip_prob=0.3)),
        ('adverse fills (20% of gap)', dict(adverse=0.2)), ('cost $0.25 + delay 10s', dict(cost=0.25, delay=10))]
out = {}
for nm, kw in scen:
    for lab, extra in [('plain', {}), ('safety', SAFE)]:
        k = dict(FROZEN); k.update(kw); k.update(extra)
        r = simulate(FV, cal, test, **k)
        print(summary(r[0], f'{nm[:26]} [{lab}]'), 'breaches', r[2][2])
        p = r[0][:, 0]; out[f'{nm} [{lab}]'] = dict(mean=float(p.mean()), median=float(np.median(p)), worst=float(p.min()), win=float((p > 0).mean()), breaches=int(r[2][2]))
json.dump(out, open('stress_A.json', 'w'), indent=1)
