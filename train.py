"""Build the frozen model artifacts from the research archive (seasons 1-450 only; 451-600 stay held out).

    python train.py --data /path/to/hurricane-season-research
Writes model/*.npz|pkl. Re-running reproduces the frozen model exactly (fixed seeds)."""
import argparse, os, sys, pickle, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'research')
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); args = ap.parse_args()
os.chdir(R); sys.path.insert(0, R)
def step(cmd):
    print('>>', cmd, flush=True); subprocess.run([sys.executable] + cmd, check=True)
os.environ['HS_DATA'] = os.path.abspath(args.data); step(['load.py', args.data])          # arch.npz
# damage params: fitted on seasons 1-350 (a subset of dev)
import damage
a = np.load('arch.npz'); p = damage.fit(a['S'][:350], a['DMG'][:350, :, 1, :5]); np.save('damage_params.npy', p)
step(['library.py']); step(['feats.py']); step(['pipeline.py'])       # out-of-fold fair values for calibration
import nowcast as N, fvc
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from value import company_closure
dev = np.arange(450)
models = {}
Cfin = company_closure(N.CLO)
for j in range(5):
    X = N.design(j)
    for nm, tgt in [('D', N.D[..., j]), ('C', Cfin[..., j])]:
        y = np.broadcast_to(tgt[:, :, None], X.shape[:-1])
        m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        m.fit(X[dev].reshape(-1, X.shape[-1]), y[dev].ravel()); models[(nm, j)] = m
M = os.path.join(HERE, 'model'); os.makedirs(M, exist_ok=True)
pickle.dump(models, open(os.path.join(M, 'nowcast_models.pkl'), 'wb'))
lib = np.load('library.npz')
np.savez(os.path.join(M, 'library.npz'), Flib=lib['Fr'][dev].reshape(-1, 17), Clib=lib['CLO'][dev].reshape(-1, 17))
np.save(os.path.join(M, 'damage_params.npy'), p)
FV = np.load('fv_cf.npz')['FV']
cal = fvc.calibrate(FV, dev, exit_row=272, delay=4)
np.savez(os.path.join(M, 'calibration.npz'), **cal)
print('frozen artifacts written to', M)
