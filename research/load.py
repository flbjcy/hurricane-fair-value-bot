"""Parse the research archive into compact numpy arrays (one-time cache)."""
import json, sys, os, numpy as np
from pathlib import Path
from multiprocessing import Pool
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('HS_DATA', '../data'))
SYMS = ['PORT','ISLE','AQUA','GRID','FERRY','RECON','MAT','GEN','HAVEN','REIN','ARCH','REBLD']
PHYS = SYMS[:5]
ISL = ['cascadia_prime','coral_key','azure_isle','palm_atoll']
OUT = ['direct','chain','recurve','stall','eyewall','erratic']
def sec(ts):
    h, m, s = map(int, ts.split('T')[1][:8].split(':'))
    return (h*3600 + m*60 + s - 34200) / 30.0   # real seconds since open
def one(i):
    d = DATA/'history'/f'season-{i+1:03d}'
    P = np.full((3, 390, 12), np.nan)
    for k, line in enumerate(open(d/'market.csv').read().splitlines()[1:]):
        f = line.split(','); r = int(f[1]) - 1; t = int(round(sec(f[0]) / 2))
        P[r, t] = [float(v) for v in f[2:]]
    S = np.full((3, 241, 6), np.nan)                 # x,y,vx,vy,radius,intensity by age/2
    ADV = np.full((3, 8, 1 + 6 + 4), np.nan)         # age, outlook onehot, hc probs
    CONE = np.full((3, 8, 4, 4), np.nan)             # per advisory: 4 cone points (age,x,y,spread)
    WIRE = []                                        # (round, age, outlook idx)
    IMP = np.full((3, 4), np.nan)                    # impact age per island
    DMG = np.full((3, 2, 6), np.nan)                 # prelim/final: 5 dmg pct + loss
    DT = np.full((3, 2), np.nan)                     # real time of reports
    END = np.full(3, np.nan)
    for line in open(d/'feed.jsonl'):
        m = json.loads(line); r = m['round'] - 1; ty = m['type']
        if ty == 'storm':
            S[r, int(m['age'])//2] = [m['x'], m['y'], m['vx'], m['vy'], m['radius'], m['intensity']]
        elif ty == 'advisory' and m['source'] == 'official':
            n = m['number'] - 1
            ADV[r, n, 0] = m['age']; ADV[r, n, 1:7] = 0; ADV[r, n, 1 + OUT.index(m['outlook'])] = 1
            ADV[r, n, 7:11] = [m['hurricane_conditions'][k] for k in ISL]
            for j, c in enumerate(m['cone'][:4]): CONE[r, n, j] = [c['age'], c['x'], c['y'], c['spread']]
        elif ty == 'advisory':
            WIRE.append((r, m['age'], OUT.index(m['outlook'])))
        elif ty == 'impact':
            IMP[r, ISL.index(m['island'])] = m['age']
        elif ty == 'damage_report':
            s = 0 if m['stage'] == 'preliminary' else 1
            DMG[r, s, :5] = [m['damage_pct'][k] for k in PHYS]; DMG[r, s, 5] = m['industry_loss_musd']
            DT[r, s] = sec(m['time'])
        elif ty == 'storm_dissipated':
            END[r] = sec(m['time'])
    st = json.load(open(d/'settlement.json'))
    return P, S, ADV, CONE, np.array(WIRE, float), IMP, DMG, DT, END, np.array([st[k] for k in SYMS])
if __name__ == '__main__':
    with Pool(2) as p: res = p.map(one, range(600))
    man = json.load(open(DATA/'history/manifest.json'))['seasons']
    pat = np.array([[OUT.index(s['pattern']) for s in x['storms']] for x in man])
    err = np.array([[np.nan if s['turned_erratic_at'] is None else s['turned_erratic_at'] for s in x['storms']] for x in man], float)
    W = np.zeros((600, 40, 4)); W[:] = np.nan
    for i, r in enumerate(res):
        w = r[4][:40]; W[i, :len(w), :3] = w
    np.savez_compressed('arch.npz',
        P=np.stack([r[0] for r in res]), S=np.stack([r[1] for r in res]), ADV=np.stack([r[2] for r in res]),
        CONE=np.stack([r[3] for r in res]), WIRE=W, IMP=np.stack([r[5] for r in res]), DMG=np.stack([r[6] for r in res]),
        DT=np.stack([r[7] for r in res]), END=np.stack([r[8] for r in res]), SET=np.stack([r[9] for r in res]),
        PAT=pat, ERR=err, SLOSS=np.array([x['season_loss_musd'] for x in man]))
    print('ok')
