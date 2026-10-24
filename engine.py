"""Streaming Hurricane Season engine: fair value + pre-committed playbook.

Everything here works message by message, with bounded state, so the same code runs in historical replay
and (through bot_adapter.py) against the exchange. Nothing looks at future data: each method only uses
what has been delivered so far.

Feed events  ->  StormTracker (per-site exposure / closure so far, latest advisory, wire, impacts, reports)
             ->  FairValue (nowcast of this storm's final damage + expected settlement over future storms)
Clock events ->  Playbook (Strategy A checkpoints, exits, safety rules; optional D and C modules)
             ->  target positions  ->  orders (adapter decides how to work them)
"""
from __future__ import annotations
import json, math, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = ['PORT', 'ISLE', 'AQUA', 'GRID', 'FERRY', 'RECON', 'MAT', 'GEN', 'HAVEN', 'REIN', 'ARCH', 'REBLD']
STOCKS = SYMS[:10]
PHYS = SYMS[:5]
ISL = ['cascadia_prime', 'coral_key', 'azure_isle', 'palm_atoll']
OUT = ['direct', 'chain', 'recurve', 'stall', 'eyewall', 'erratic']
LIM = np.array([200, 300, 500, 400, 800, 500, 700, 600, 250, 200], float)
LOSS_B = np.array([9.0, 6.0, 2.5, 7.0, 1.2])            # $M insured loss per damage-% point (recovered exactly)
B1 = np.array([1, 1, 2, 1, 2, 0, 0, 0, 1, 0.]); B2 = np.array([0, 0, 0, 0, 0, 2, 2, 1, 0, 0.])
CLOSE_SIG, CLOSE_THR = 0.70, 0.34                        # wind kernel / closure threshold (recovered)


def market_seconds(ts: str) -> float:
    """feed/market timestamp -> real seconds since the round opened (09:30 market = 0)."""
    h, m, s = map(int, ts.split('T')[1][:8].split(':'))
    return (h*3600 + m*60 + s - 34200)/30.0


# ----------------------------------------------------------------------------------------------- valuation
def value(cum, clo, L):
    """Settlement value of the 10 stocks from cumulative damage %, site-weighted closed seconds, season loss $M.
    Recovered from the 600 research seasons (residual sd <= $0.04)."""
    P, I, A, G, F = [cum[..., k] for k in range(5)]
    cP, cG, cF = clo[..., 0], clo[..., 3], clo[..., 4]
    return np.stack([
        100 - 0.55*P - 0.02*cP,
        60 - 0.38*I,
        35 - 0.20*A,
        50 - 0.24*G - 0.05*cG,
        25 - 0.05*F - 0.06*cF,
        40 + 24*(1 - np.exp(-(0.009*P + 0.007*G + 0.0012*F))),
        28 + 14*(1 - np.exp(-(0.0075*P + 0.005*I + 0.0021*A + 0.0058*G + 0.001*F))),
        32 + 18*(1 - np.exp(-0.0082*cG)),
        75 - 0.06*np.clip(L - 400, 0, 900),
        90 - 0.035*np.clip(L - 1300, 0, 700),
    ], -1)


class Geometry:
    def __init__(self, case: dict):
        sx, sy, sw, sc = [], [], [], []
        for j, c in enumerate(PHYS):
            for s in case['companies'][c]['sites']:
                sx.append(s['x']); sy.append(s['y']); sw.append(s['weight']); sc.append(j)
        self.SX, self.SY, self.SW, self.SC = map(np.array, (sx, sy, sw, sc))
        self.NS = len(sx)

    def company(self, site_vals):       # weighted sum of site values per company
        return np.stack([(site_vals*self.SW)[..., self.SC == j].sum(-1) for j in range(5)], -1)


# ------------------------------------------------------------------------------------------- storm state
class StormTracker:
    """Per-round storm state. Integrates damage exposure and closure time at every site as observations
    arrive (gaps are filled by linear interpolation between observations, as the 'track' message allows)."""

    def __init__(self, geo: Geometry, dmg_params):
        self.g = geo; p = dmg_params
        self.sig, self.thr, self.pw = p[:3]; self.k = p[3:8]; self.fmax = p[8:13]
        self.reset(0)

    def reset(self, rnd):
        ns = self.g.NS
        self.round = rnd; self.E = np.zeros(ns); self.C = np.zeros(ns); self.last = None   # last obs dict
        self.adv = None; self.wire = None; self.impacts = set(); self.prelim = None; self.final = None
        self.dissipated = False; self.obs_ages = []

    def _accumulate(self, m):
        u = np.hypot(m['x']-self.g.SX, m['y']-self.g.SY)/m['radius']
        w = m['intensity']*np.exp(-.5*(u/self.sig)**2)
        self.E += np.maximum(w-self.thr, 0)**self.pw*2
        wc = m['intensity']*np.exp(-.5*(u/CLOSE_SIG)**2)
        self.C += (wc > CLOSE_THR)*2.0

    def on_storm(self, m):
        a = int(m['age'])
        if self.last is not None and a <= self.last['age']:
            return                                          # duplicate / out of order
        if self.last is not None and a - self.last['age'] > 2:   # fill a gap by interpolation
            l = self.last
            for aa in range(l['age']+2, a, 2):
                f = (aa-l['age'])/(a-l['age'])
                self._accumulate({k: l[k]+f*(m[k]-l[k]) for k in ('x', 'y', 'radius', 'intensity')})
        self._accumulate(m)
        self.last = {k: m[k] for k in ('age', 'x', 'y', 'vx', 'vy', 'radius', 'intensity')}

    def on_message(self, m):
        t = m.get('type')
        if t == 'storm': self.on_storm(m)
        elif t == 'advisory' and m.get('source') == 'official': self.adv = m
        elif t == 'advisory': self.wire = m
        elif t == 'impact': self.impacts.add(m['island'])
        elif t == 'damage_report':
            if m['stage'] == 'preliminary': self.prelim = m
            else: self.final = m
        elif t == 'storm_dissipated': self.dissipated = True

    # ---- projections (identical maths to research/feats.py)
    def _proj(self, xs, ys, R, I):
        d = np.hypot(xs[:, None]-self.g.SX, ys[:, None]-self.g.SY)/R
        w = I*np.exp(-.5*(d/self.sig)**2); wc = I*np.exp(-.5*(d/CLOSE_SIG)**2)
        return np.maximum(w-self.thr, 0).__pow__(self.pw).sum(0)*2, (wc > CLOSE_THR).sum(0)*2.0

    def projections(self):
        """exposure/closure increments under: constant velocity (to end of storm, and 60 s), official cone."""
        ns = self.g.NS; z = (np.zeros(ns), np.zeros(ns))
        if self.last is None: return z, z, z
        l = self.last; age = l['age']
        tau = np.arange(2, 482, 2.); keep = age+tau <= 480
        xs = l['x']+l['vx']*tau; ys = l['y']+l['vy']*tau
        cv = self._proj(xs[keep], ys[keep], l['radius'], l['intensity']) if keep.any() else z
        k60 = keep & (tau <= 60)
        cv60 = self._proj(xs[k60], ys[k60], l['radius'], l['intensity']) if k60.any() else z
        cone = z
        a = self.adv
        if a is not None and a['age'] <= age:
            cp = [c for c in a['cone']][:4]
            ag = np.r_[age, [c['age'] for c in cp]]; px = np.r_[l['x'], [c['x'] for c in cp]]
            py = np.r_[l['y'], [c['y'] for c in cp]]; sp = np.r_[0, [c['spread'] for c in cp]]
            fut = np.arange(age+2, 481, 2.)
            if len(ag) >= 2 and len(fut):
                vxl = (px[-1]-px[-2])/(ag[-1]-ag[-2]); vyl = (py[-1]-py[-2])/(ag[-1]-ag[-2])
                fx = np.where(fut <= ag[-1], np.interp(fut, ag, px), px[-1]+vxl*(fut-ag[-1]))
                fy = np.where(fut <= ag[-1], np.interp(fut, ag, py), py[-1]+vyl*(fut-ag[-1]))
                fs = np.interp(fut, ag, sp)
                e = np.zeros(ns); c = np.zeros(ns)
                for off in (-1, 0, 1):
                    e1, c1 = self._proj(fx+off*fs, fy, l['radius'], l['intensity']); e += e1; c += c1
                cone = (e/3, c/3)
        return cv, cv60, cone

    def frac(self, E):
        return self.fmax[self.g.SC]*(1-np.exp(-self.k[self.g.SC]*E))


# ---------------------------------------------------------------------------------------- season memory
class SeasonState:
    """What carries between rounds: site health, cumulative damage, closure, season loss (all from the
    final damage reports, which are exact)."""

    def __init__(self, geo):
        self.g = geo; self.h = np.ones(geo.NS); self.cum = np.zeros(5); self.clo = np.zeros(5); self.L = 0.0

    def close_round(self, tracker: StormTracker):
        f = tracker.final
        D = np.array([f['damage_pct'][k] for k in PHYS]); Lr = f['industry_loss_musd']
        F = tracker.frac(tracker.E)
        mod = 100*self.g.company(self.h*F)
        ratio = np.where(mod > 1e-6, D/np.maximum(mod, 1e-9), 0.0)
        Fr = np.clip(F*ratio[self.g.SC], 0, 1)                    # reconcile site fractions with the report
        self.h = self.h*(1-Fr); self.cum = self.cum+D; self.clo = self.clo+self.g.company(tracker.C); self.L += Lr


# ------------------------------------------------------------------------------------------ fair value
class FairValue:
    def __init__(self, geo, model_dir=os.path.join(HERE, 'model'), M=600, seed=7):
        self.g = geo
        self.models = pickle.load(open(os.path.join(model_dir, 'nowcast_models.pkl'), 'rb'))
        lib = np.load(os.path.join(model_dir, 'library.npz')); self.Flib = lib['Flib']; self.Clib = lib['Clib']
        rng = np.random.default_rng(seed)
        self.pi = rng.integers(0, len(self.Flib), M); self.pj = rng.integers(0, len(self.Flib), M)

    def design(self, tr: StormTracker, ss: SeasonState, T: float):
        """feature rows (one per company) exactly as research/nowcast.design"""
        cv, cv60, cone = tr.projections()
        comp = lambda E: 100*self.g.company(ss.h*tr.frac(E))
        Vs = [comp(tr.E), comp(tr.E+cv[0]), comp(tr.E+cv60[0]), comp(tr.E+cone[0])]
        Cs = [self.g.company(tr.C), self.g.company(tr.C+cv[1]), self.g.company(tr.C+cv60[1]), self.g.company(tr.C+cone[1])]
        age = float(np.clip(T-60, 0, 480))
        a = tr.adv if (tr.adv is not None and tr.adv['age'] <= age) else None
        hc = [a['hurricane_conditions'][k] for k in ISL] if a else [0.0]*4
        outl = [1.0 if (a and a['outlook'] == o) else 0.0 for o in OUT]
        w = tr.wire if (tr.wire is not None and tr.wire['age'] <= age) else None
        wire = [1.0 if (w and w['outlook'] == o) else 0.0 for o in OUT]
        l = tr.last or {}
        st = [l.get(k, np.nan) for k in ('x', 'y', 'vx', 'vy', 'radius', 'intensity')]
        imp = [1.0 if i in tr.impacts else 0.0 for i in ISL]
        rows = []
        for j in range(5):
            pre = tr.prelim['damage_pct'][PHYS[j]] if tr.prelim is not None else np.nan
            rows.append([v[j] for v in Vs] + [c[j] for c in Cs] + [age] + hc + outl + wire + st + imp +
                        [pre, ss.h[self.g.SC == j].mean()])
        return np.array(rows, float), cone

    def nowcast(self, tr, ss, T):
        X, cone = self.design(tr, ss, T)
        Dh = np.array([max(self.models[('D', j)].predict(X[j:j+1])[0], 0) for j in range(5)])
        Ch = np.array([max(self.models[('C', j)].predict(X[j:j+1])[0], 0) for j in range(5)])
        if tr.final is not None:
            Dh = np.array([tr.final['damage_pct'][k] for k in PHYS])
        return Dh, Ch, cone

    def fair_value(self, tr, ss, T, rnd):
        """expected settlement of the 10 stocks given everything seen so far"""
        Dh, Ch, cone = self.nowcast(tr, ss, T)
        fs = tr.frac(tr.E+cone[0])
        modc = 100*self.g.company(ss.h*fs)
        fs = np.clip(fs*np.where(modc > 1e-6, Dh/np.maximum(modc, 1e-9), 0)[self.g.SC], 0, 1)
        h = ss.h*(1-fs); cum = ss.cum+Dh; clo = ss.clo+Ch; L = ss.L+Dh@LOSS_B
        nf = 2-rnd
        if nf == 0: return value(cum, clo, L), Dh
        F1 = self.Flib[self.pi]; add1 = h*F1
        if nf == 1:
            d = 100*self.g.company(add1); c = self.g.company(self.Clib[self.pi])
        else:
            add2 = (h-add1)*self.Flib[self.pj]
            d = 100*self.g.company(add1+add2); c = self.g.company(self.Clib[self.pi]+self.Clib[self.pj])
        return value(cum+d, clo+c, L+d@LOSS_B).mean(0), Dh


# -------------------------------------------------------------------------------------------- playbook
FROZEN = dict(checkpoints=(150, 190, 230, 270, 310, 350, 390), frac=1/7, z=1.0, band=0.25, cost=0.10,
              risk_budget=1500.0, guard_sigma=5.0, exit_time=550, breach=6.0, widen_stop=3.0, size=1.0,
              snap=False, snap_margin=0.10, etf=False, etf_thr=1.0, etf_hold=6)


class Playbook:
    """The rules are fixed before the round starts. At each checkpoint the engine only plugs the new fair
    value and the current price into them; there is nothing to decide on the spot.

    Strategy A (on): at checkpoints 150..390 s after the open (storm age 90-330 s), compare fair value with
    the house price. expected_move = beta[t,stock]*gap (beta learned on 450 past seasons). If it beats
    2*cost+band, add a tranche of up to 1/7 of the position limit, scaled by expected_move / typical error.
    Total risk is capped (risk_budget, $ one-sd). All positions are closed at 550 s (10 s after the storm
    dissipates, after the house has repriced for it), before the damage reports. Safety: a stock whose gap keeps widening against us stops
    receiving tranches for the round; a stock whose fair value still disagrees with the house by > 6 typical
    gaps after the storm is presumed mis-modelled and disabled for the rest of the season; an announcement
    naming a stock disables it immediately and closes it.
    Strategy D (optional, latency-gated): round-3 final report -> settlement is known exactly; post limit
    orders that only fill if the house quote is still stale.
    Strategy C (optional, latency-gated): ETF marks trail their basket by ~6 s; trade the ETF toward it.
    """

    def __init__(self, model_dir=os.path.join(HERE, 'model'), **kw):
        self.o = dict(FROZEN); self.o.update(kw)
        cal = np.load(os.path.join(model_dir, 'calibration.npz'))
        self.beta, self.Sig, self.gsd = cal['beta'], cal['Sig'], cal['gsd']
        self.ticks = np.arange(70, 741, 10)
        self.alive = np.ones(10, bool)

    def idx(self, T): return int(np.searchsorted(self.ticks, T))

    def new_round(self):
        self.pos = np.zeros(10); self.g0 = None; self.frozen = np.zeros(10, bool)

    def checkpoint(self, T, fv, px):
        """returns new target position (shares) for the 10 stocks"""
        o = self.o; i = self.idx(T); g = fv-px; e = self.beta[i]*g; c = o['cost']
        if self.g0 is None: self.g0 = g.copy()
        self.frozen |= (np.sign(self.g0)*(g-self.g0) > o['widen_stop']*self.gsd[i]) & (np.abs(self.g0) > 0)
        ok = (np.abs(e) > 2*c+o['band']) & (np.abs(g) < o['guard_sigma']*self.gsd[i]) & self.alive & ~self.frozen
        sd = np.sqrt(np.diag(self.Sig[i]))
        tr = np.where(ok, np.clip(e/(o['z']*sd), -1, 1), 0)*LIM*o['frac']*o['size']
        new = np.clip(self.pos+tr, -LIM, LIM)
        if o['risk_budget'] is not None and np.sqrt(new@self.Sig[i]@new) > o['risk_budget']:
            lo, hi = 0.0, 1.0
            for _ in range(20):
                m = (lo+hi)/2; cand = self.pos+m*(new-self.pos)
                lo, hi = (m, hi) if np.sqrt(cand@self.Sig[i]@cand) <= o['risk_budget'] else (lo, m)
            new = self.pos+lo*(new-self.pos)
        self.pos = np.round(new)
        return self.pos.copy(), dict(gap=g, expected=e, ok=ok)

    def ladder(self, T, fv):
        """The decision table for the next checkpoint, known before any new price arrives: for each stock,
        the price at or below which we buy, at or above which we sell, and the tranche size at a given price."""
        o = self.o; i = self.idx(T); b = np.maximum(self.beta[i], 1e-6); need = (2*o['cost']+o['band'])/b
        sd = np.sqrt(np.diag(self.Sig[i]))
        def size(price):
            e = b*(fv-price); ok = (np.abs(e) > 2*o['cost']+o['band']) & self.alive & ~self.frozen
            return np.where(ok, np.clip(e/(o['z']*sd), -1, 1), 0)*LIM*o['frac']*o['size']
        return dict(buy_below=np.where(self.alive, fv-need, np.nan), sell_above=np.where(self.alive, fv+need, np.nan), size=size)

    def after_storm(self, fv, px):
        """model-breach check (run at 540 s)"""
        i = self.idx(540); z = np.abs(fv-px)/self.gsd[i]
        newly = (z > self.o['breach']) & self.alive; self.alive &= ~newly
        return [STOCKS[j] for j in np.where(newly)[0]]

    def on_announcement(self, symbols):
        for s in symbols:
            if s in STOCKS: self.alive[STOCKS.index(s)] = False
        hit = np.array([s in symbols for s in STOCKS]); self.pos = np.where(hit, 0, self.pos)
        return self.pos.copy()


class Engine:
    """Glue: feed messages + clock + prices -> target positions / order intents."""

    def __init__(self, case: dict, model_dir=os.path.join(HERE, 'model'), **kw):
        self.geo = Geometry(case)
        self.tr = StormTracker(self.geo, np.load(os.path.join(model_dir, 'damage_params.npy')))
        self.ss = SeasonState(self.geo); self.fvm = FairValue(self.geo, model_dir)
        self.pb = Playbook(model_dir, **kw); self.round = -1; self.seen = set(); self.log = []
        self.disabled_reason = {}

    def on_message(self, m):
        if 'seq' in m:
            if m['seq'] in self.seen: return
            self.seen.add(m['seq'])
        r = int(m.get('round', 1))-1
        if r != self.round: self.start_round(r)
        if m.get('type') == 'announcement':
            for s in m.get('symbols', []): self.disabled_reason[s] = 'announcement: '+m.get('headline', '')
            self.log.append(('announcement', m.get('symbols')))
            return ('target', self.pb.on_announcement(m.get('symbols', [])))
        self.tr.on_message(m)
        if m.get('type') == 'damage_report' and m.get('stage') == 'final':
            return ('final_report', None)

    def start_round(self, r):
        if self.round >= 0 and self.tr.final is not None: self.ss.close_round(self.tr)
        self.round = r; self.tr.reset(r); self.pb.new_round()

    def fair_value(self, T):
        fv, Dh = self.fvm.fair_value(self.tr, self.ss, T, self.round); return fv

    def on_clock(self, T, px):
        """call every time the clock passes a whole 10 s (T real seconds since open); px = current marks/mids"""
        o = self.pb.o
        if T in o['checkpoints']:
            data_ok = self.tr.last is not None and (T-60)-self.tr.last['age'] <= 6     # storm data fresh?
            if not data_ok:
                self.log.append(('stale_data_skip', T)); return None
            fv = self.fair_value(T); q, info = self.pb.checkpoint(T, fv, px[:10])
            self.log.append(('checkpoint', T, q.copy())); return ('target', q)
        if T == 540:
            fv = self.fair_value(T); hit = self.pb.after_storm(fv, px[:10])
            for s in hit: self.disabled_reason[s] = 'model breach after storm %d' % (self.round+1)
        if T == o['exit_time']:
            self.pb.pos = np.zeros(10); return ('target', self.pb.pos.copy())
        return None

    def settlement_snap(self):
        """round 3 final report: exact settlement values (Strategy D)."""
        if self.round != 2 or self.tr.final is None: return None
        f = self.tr.final; D = np.array([f['damage_pct'][k] for k in PHYS])
        v = value(self.ss.cum+D, self.ss.clo+self.geo.company(self.tr.C), self.ss.L+f['industry_loss_musd'])
        for s in self.disabled_reason: v[STOCKS.index(s)] = np.nan if s in STOCKS else 0
        return v
