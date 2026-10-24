"""Real-time 'storm off-script' detector: official outlook says erratic, or the storm is far from where the
latest official cone said it would be now (in units of the cone spread)."""
import numpy as np
from feats import TICKS, NT
a = np.load('arch.npz'); S = a['S']; ADV = a['ADV']; CONE = a['CONE']
def offscript():
    age = np.clip(TICKS-60, 0, 480); n_adv = np.clip(((age-30)//60).astype(int), -1, 7)
    z = np.zeros((600, 3, NT)); err_flag = np.zeros((600, 3, NT))
    for i in range(NT):
        n = n_adv[i]
        if n < 0: continue
        ob = S[:, :, age[i]//2, :2]
        # cone of the PREVIOUS advisory evaluated at the current age (an out-of-sample check of the forecast)
        for nn in [n-1, n]:
            if nn < 0: continue
            cp = CONE[:, :, nn]                             # [600,3,4,4]
            ag0 = ADV[:, :, nn, 0]
            # interpolate between (adv age, obs pos at adv age) and cone points
            for s in range(600):
                for r in range(3):
                    c = cp[s, r]; c = c[~np.isnan(c[:, 0])]
                    ags = np.r_[ag0[s, r], c[:, 0]]
                    if age[i] < ags[0] or age[i] > ags[-1] or len(ags) < 2: continue
                    p0 = S[s, r, int(ags[0])//2, :2]
                    xs = np.interp(age[i], ags, np.r_[p0[0], c[:, 1]]); ys = np.interp(age[i], ags, np.r_[p0[1], c[:, 2]])
                    sp = np.interp(age[i], ags, np.r_[5.0, c[:, 3]])
                    z[s, r, i] = max(z[s, r, i], np.hypot(ob[s, r, 0]-xs, ob[s, r, 1]-ys)/max(sp, 5.0))
            break
        err_flag[:, :, i] = ADV[:, :, n, 6]
    return z, err_flag
if __name__ == '__main__':
    z, ef = offscript(); np.savez('offscript.npz', z=z, ef=ef)
    ERR = a['ERR']; turned = ~np.isnan(ERR)
    for i in [10, 15, 20, 25, 30, 35]:
        print('T', TICKS[i], 'z pctl normal', np.percentile(z[~turned][:, i], [50, 90, 99]).round(2),
              ' turned', np.percentile(z[turned][:, i], [50, 90]).round(2), ' erratic outlook frac normal/turned', ef[~turned][:, i].mean().round(3), ef[turned][:, i].mean().round(3))
