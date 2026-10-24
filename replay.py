"""Historical paper replay of the streaming engine, message by message.

    python replay.py --data /path/to/hurricane-season-research --seasons 451-600 [--delay 4] [--cost 0.10]
                     [--snap] [--etf] [--out replay.json]

Fills are hypothetical: an order decided at real second T fills at the house mark of T+delay, plus `cost`
per share. The research data has no order book, so depth, queue position and quote skew are NOT modelled.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from engine import Engine, market_seconds, SYMS, STOCKS, LIM, B1, B2


def load_season(data, s):
    d = os.path.join(data, 'history', f'season-{s:03d}')
    msgs = [json.loads(l) for l in open(os.path.join(d, 'feed.jsonl'))]
    for m in msgs: m['_T'] = market_seconds(m['time'])
    P = np.zeros((3, 390, 12))
    for line in open(os.path.join(d, 'market.csv')).read().splitlines()[1:]:
        f = line.split(','); P[int(f[1])-1, int(round(market_seconds(f[0])/2))] = [float(v) for v in f[2:]]
    st = json.load(open(os.path.join(d, 'settlement.json')))
    return msgs, P, np.array([st[k] for k in SYMS])


def replay_season(case, data, s, delay=4, cost=0.10, snap=False, etf=False, etf_thr=1.0, etf_hold=6, verbose=False, **kw):
    msgs, P, SET = load_season(data, s)
    eng = Engine(case, cost=cost, **kw)
    posA = np.zeros(10); pos = np.zeros(12); cash = 0.0; pending = []; trades = []; mi = 0
    etf_open = [None, None]                  # (exit_T, qty) per ETF
    per_strategy = {'A': 0.0, 'D': 0.0, 'C': 0.0}
    def fill(T, px, dq, tag):
        nonlocal cash
        cash -= dq@px + np.abs(dq).sum()*cost; trades.append((T, tag, dq.copy()))
        per_strategy[tag] -= dq@px + np.abs(dq).sum()*cost
    for r in range(3):
        for t in range(390):
            T = 2*t; px = P[r, t]
            while mi < len(msgs) and (msgs[mi]['round']-1 < r or (msgs[mi]['round']-1 == r and msgs[mi]['_T'] <= T)):
                ev = eng.on_message(msgs[mi]); mi += 1
                if ev and ev[0] == 'final_report' and snap and r == 2:
                    v = eng.settlement_snap()
                    for j in range(10):          # limit orders at V -/+ (cost+margin): only fill if still stale
                        if np.isfinite(v[j]): pending.append((T+delay, 'D', j, v[j]))
                if ev and ev[0] == 'target':    # announcement: flatten named stocks now
                    tgt = np.r_[ev[1], pos[10:]]; pending.append((T+delay, 'A', None, tgt))
            # fills due now
            keep = []
            for due, tag, j, x in pending:
                if due > T: keep.append((due, tag, j, x)); continue
                if tag == 'A':
                    dq = np.zeros(12); dq[:10] = x[:10]-posA;
                    if np.any(dq): fill(T, px, dq, 'A'); pos += dq; posA[:] = x[:10]
                elif tag == 'D':
                    v = x; m = px[j]; edge = v-m
                    if abs(edge) > cost+eng.pb.o['snap_margin']:
                        room = LIM[j]-np.sign(edge)*pos[j]
                        q = np.sign(edge)*max(room, 0)
                        if q: dq = np.zeros(12); dq[j] = q; fill(T, px, dq, 'D'); pos += dq
                elif tag == 'C':
                    dq = np.zeros(12); dq[j] = x; fill(T, px, dq, 'C'); pos += dq
            pending = keep
            if T % 10 == 0:
                ev = eng.on_clock(T, px)
                if ev and ev[0] == 'target':
                    pending.append((T+delay, 'A', None, ev[1]))
            if etf:
                for e, b, lim in [(0, B1, 50), (1, B2, 100)]:
                    if etf_open[e] is not None and T >= etf_open[e][0]:
                        pending.append((T+delay, 'C', 10+e, -etf_open[e][1])); etf_open[e] = None
                    elif etf_open[e] is None and T <= 760-delay-etf_hold:
                        gap = px[:10]@b - px[10+e]
                        if abs(gap) > etf_thr:
                            q = np.sign(gap)*lim; pending.append((T+delay, 'C', 10+e, q)); etf_open[e] = (T+delay+etf_hold, q)
        # end of round: anything still pending is dropped (market closed)
        pending = [p for p in pending if p[1] == 'D'] if r < 2 else []
        if any(e is not None for e in etf_open):      # an ETF trade still open at the close: flatten at the last mark
            for e in range(2):
                if etf_open[e] is not None:
                    dq = np.zeros(12); dq[10+e] = -etf_open[e][1]; fill(T, px, dq, 'C'); pos += dq; etf_open[e] = None
    final = cash + pos@SET
    # strategy attribution: settle residual positions (A is flat; D/C may hold to settlement)
    by = {k: float(v) for k, v in per_strategy.items()}
    # positions still open at the close settle at the settlement values
    for tag in ('D', 'C'):
        q = np.zeros(12)
        for T_, tg, dq in trades:
            if tg == tag: q += dq
        by[tag] += float(q@SET)
    return dict(season=s, pnl=float(final), by_strategy=by, trades=len(trades), log=eng.log,
                disabled=eng.disabled_reason, residual=pos.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--seasons', default='451-600')
    ap.add_argument('--delay', type=int, default=4); ap.add_argument('--cost', type=float, default=0.10)
    ap.add_argument('--snap', action='store_true'); ap.add_argument('--etf', action='store_true')
    ap.add_argument('--out', default='replay.json')
    a = ap.parse_args()
    lo, hi = map(int, a.seasons.split('-')); case = json.load(open(os.path.join(a.data, 'case.json')))
    rows = []; t0 = time.time()
    for s in range(lo, hi+1):
        r = replay_season(case, a.data, s, delay=a.delay, cost=a.cost, snap=a.snap, etf=a.etf)
        rows.append(r)
        if (s-lo) % 25 == 0: print(f'season {s}: pnl {r["pnl"]:.1f}  ({time.time()-t0:.0f}s)', flush=True)
    p = np.array([r['pnl'] for r in rows])
    summ = dict(seasons=a.seasons, delay=a.delay, cost=a.cost, snap=a.snap, etf=a.etf, mean=float(p.mean()),
                median=float(np.median(p)), worst=float(p.min()), win_rate=float((p > 0).mean()))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, seasons=[{k: v for k, v in r.items() if k != 'log'} for r in rows]), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
