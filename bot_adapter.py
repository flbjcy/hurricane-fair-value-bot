"""Live adapter (UNTESTED against the exchange - run on the testnet with --dry-run first).

    pip install "websockets>=14" numpy scikit-learn
    python bot_adapter.py --url https://hurricane.cornelltradingcompetition.org --key $BOT_KEY --dry-run

What it does: feeds every news message to the Engine, keeps the latest quotes, and every whole 10 s of the
round clock asks the Engine for target positions. A change of target becomes ONE limit order per symbol
whose price is fixed by the rule, not by the moment:
    buy  limit = min(ask, fair_value - 2*cost - band)      sell limit = max(bid, fair_value + 2*cost + band)
so an order never pays more than the edge the playbook requires. Orders still open after `--ttl` seconds
are cancelled (and re-decided at the next checkpoint). Exits (target 0 at 540 s) cross at bid/ask.
Never paste your key into chat; pass it with --key or the BOT_KEY environment variable.
"""
from __future__ import annotations
import argparse, asyncio, json, os, time, urllib.request
import numpy as np
import websockets
from engine import Engine, market_seconds, SYMS, STOCKS

HEADERS = {"User-Agent": "hurricane-fv/1.0"}


class Live:
    def __init__(self, url, key, dry, ttl):
        self.url, self.key, self.dry, self.ttl = url.rstrip('/'), key, dry, ttl
        self.case = self.get('/cases/hurricane.json', api=False)
        self.eng = Engine(self.case); self.quotes = {}; self.last_tick = None; self.open_orders = []

    def req(self, path, method='GET', body=None, api=True):
        r = urllib.request.Request(self.url+('/api/v1' if api else '')+path, method=method,
                                   data=None if body is None else json.dumps(body).encode(),
                                   headers={**HEADERS, 'X-API-Key': self.key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=5) as resp: return json.load(resp)
    def get(self, p, api=True): return self.req(p, api=api)

    def positions(self):
        try:
            data = self.get('/positions'); out = np.zeros(12)
            rows = data if isinstance(data, list) else data.get('positions', [])
            for row in rows:
                if row.get('symbol') in SYMS: out[SYMS.index(row['symbol'])] = row.get('quantity', row.get('position', 0))
            return out
        except Exception as e:
            print('positions unavailable:', e); return None

    def marks(self):
        px = np.full(12, np.nan)
        for i, s in enumerate(SYMS):
            q = self.quotes.get(s) or {}
            if q.get('bid') is not None and q.get('ask') is not None: px[i] = (q['bid']+q['ask'])/2
        return px

    def send(self, sym, qty, fv):
        o = self.eng.pb.o; edge = 2*o['cost']+o['band']; q = self.quotes.get(sym) or {}
        side = 'buy' if qty > 0 else 'sell'
        if fv is None:      # exit / flatten: cross the spread
            price = q.get('ask') if side == 'buy' else q.get('bid')
        else:
            price = min(q.get('ask', np.inf), fv-edge) if side == 'buy' else max(q.get('bid', -np.inf), fv+edge)
        if price is None or not np.isfinite(price): print('no quote for', sym); return
        order = {'symbol': sym, 'side': side, 'order_type': 'limit', 'quantity': int(abs(qty)), 'price': round(float(price), 2)}
        if self.dry: print('DRY', order); return
        try:
            resp = self.req('/orders', 'POST', order); self.open_orders.append((time.time(), resp.get('order_id') or resp.get('id')))
        except Exception as e: print('order rejected', order, e)

    def cancel_stale(self):
        now = time.time(); keep = []
        for t0, oid in self.open_orders:
            if now-t0 > self.ttl and oid is not None and not self.dry:
                try: self.req(f'/orders/{oid}', 'DELETE')
                except Exception: pass
            else: keep.append((t0, oid))
        self.open_orders = keep

    def on_clock(self, T):
        px = self.marks()
        if np.isnan(px[:10]).any(): print('quotes incomplete; skip', T); return
        ev = self.eng.on_clock(T, px)
        if not ev or ev[0] != 'target': return
        tgt = ev[1]; cur = self.positions()
        if cur is None: return                          # do not trade blind
        fv = self.eng.fair_value(T) if T != self.eng.pb.o['exit_time'] else None
        for j, s in enumerate(STOCKS):
            dq = tgt[j]-cur[j]
            if abs(dq) >= 1: self.send(s, dq, None if fv is None else fv[j])

    async def run(self):
        sub = {'action': 'subscribe', 'symbols': SYMS, 'channels': ['news', 'quotes']}
        ws_url = self.url.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1)+'/ws/v1/market-data'
        while True:
            try:
                async with websockets.connect(ws_url, additional_headers={'X-API-Key': self.key}) as ws:
                    await ws.send(json.dumps(sub))
                    async for frame in ws:
                        d = json.loads(frame)
                        if d.get('type') == 'quote': self.quotes[d['symbol']] = d; continue
                        m = d.get('message') if d.get('type') == 'news' else None
                        if not isinstance(m, dict): continue
                        if m.get('type') == 'snapshot': continue
                        self.eng.on_message(m)
                        T = int(market_seconds(m['time'])) if 'time' in m else None
                        if T is not None:
                            tick = T - T % 10
                            if tick != self.last_tick:
                                self.last_tick = tick; await asyncio.to_thread(self.on_clock, tick); self.cancel_stale()
            except (OSError, websockets.WebSocketException) as e:
                print('disconnected', e, '- reconnecting'); await asyncio.sleep(2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True); ap.add_argument('--key', default=os.environ.get('BOT_KEY'))
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--ttl', type=float, default=8.0)
    a = ap.parse_args()
    asyncio.run(Live(a.url, a.key, a.dry_run, a.ttl).run())
