"""Causality and robustness checks for the streaming engine.   python -m pytest -q tests  (or python tests/test_engine.py)
Set HS_DATA to the research folder."""
import copy, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine, market_seconds
from replay import load_season, replay_season
DATA = os.environ.get('HS_DATA', '/home/claude/data')
CASE = json.load(open(os.path.join(DATA, 'case.json')))

def fair_values(msgs, P, cut=None):
    """run the engine through a season, recording the fair value at every checkpoint"""
    eng = Engine(CASE); out = {}; mi = 0
    for r in range(3):
        for t in range(390):
            T = 2*t
            while mi < len(msgs) and (msgs[mi]['round']-1 < r or (msgs[mi]['round']-1 == r and msgs[mi]['_T'] <= T)):
                eng.on_message(msgs[mi]); mi += 1
            if T in eng.pb.o['checkpoints']:
                out[(r, T)] = eng.fair_value(T)
                eng.on_clock(T, P[r, t])
            if cut is not None and (r, T) == cut: return out
    return out

def test_future_mutation_does_not_change_past():
    """Corrupt everything after a cut point (storm path, advisories, reports, prices): earlier values identical."""
    msgs, P, SET = load_season(DATA, 457)
    base = fair_values(msgs, P)
    cut = (1, 270)
    bad = copy.deepcopy(msgs)
    for m in bad:
        late = (m['round']-1 > cut[0]) or (m['round']-1 == cut[0] and m['_T'] > cut[1])
        if late and m['type'] == 'storm': m['x'] += 150; m['intensity'] *= 1.5
        if late and m['type'] == 'damage_report': m['damage_pct'] = {k: 99.0 for k in m['damage_pct']}
        if late and m['type'] == 'advisory' and m.get('source') == 'official': m['outlook'] = 'erratic'
    P2 = P.copy(); P2[cut[0], cut[1]//2+1:] *= 0.5; P2[cut[0]+1:] *= 0.5
    mut = fair_values(bad, P2, cut)
    for k, v in mut.items():
        assert np.allclose(v, base[k]), k

def test_duplicates_ignored():
    msgs, P, SET = load_season(DATA, 460)
    dup = []
    for m in msgs: dup += [m, dict(m)] if m['type'] in ('storm', 'advisory', 'damage_report') else [m]
    a = fair_values(msgs, P); b = fair_values(dup, P)
    for k in a: assert np.allclose(a[k], b[k]), k

def test_dropped_observations_degrade_gracefully():
    """drop 25% of storm observations (gaps are interpolated): fair values move only a little"""
    msgs, P, SET = load_season(DATA, 463)
    rng = np.random.default_rng(0)
    thin = [m for m in msgs if not (m['type'] == 'storm' and m['age'] % 10 != 0 and rng.random() < 0.25)]
    a = fair_values(msgs, P); b = fair_values(thin, P)
    diff = max(np.abs(a[k]-b[k]).max() for k in a)
    assert diff < 1.0, diff

def test_announcement_disables_symbol():
    msgs, P, SET = load_season(DATA, 470)
    eng = Engine(CASE)
    eng.on_message({'type': 'announcement', 'round': 1, 'storm': 'x', 'time': msgs[0]['time'], 'headline': 'test', 'symbols': ['PORT'], 'seq': -5})
    assert not eng.pb.alive[0] and eng.pb.alive[1:].all()

def test_season_isolation():
    """a fresh engine per season: results do not depend on what ran before"""
    r1 = replay_season(CASE, DATA, 455)['pnl']
    replay_season(CASE, DATA, 456)
    r2 = replay_season(CASE, DATA, 455)['pnl']
    assert r1 == r2

if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('PASS', n)
