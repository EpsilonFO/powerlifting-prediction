"""stack(37 + modèles avec variables candidats) -> correction candidats recalculée -> doublons."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from powerlifting import siblings as sb
from powerlifting.cv import prepare
from powerlifting.config import TARGET
from powerlifting.ensemble import load_predictions, stacking
from powerlifting.postprocess import row_keys, replace_duplicates
from powerlifting.recipes import ENSEMBLE
S, extra = sys.argv[1], sys.argv[2:]
train, test, sample, F_tr, F_te, folds = prepare()
y = train[TARGET].to_numpy(float); ntr = len(train); keys = row_keys(train)
O, T = load_predictions(list(ENSEMBLE))
for e in extra:
    O = np.column_stack([O, np.load(f"{S}/{e}_oof.npy")]); T = np.column_stack([T, np.load(f"{S}/{e}_pred.npy")])
names = list(ENSEMBLE) + extra
honest, pred, coefs = stacking(names, y, folds, O, T, F_tr, F_te)
print("coefficients des ajouts :", {e: coefs[e] for e in extra})
def dd(p):
    p = p.copy()
    for k in range(5):
        va = folds == k; p[va], _ = replace_duplicates(p[va], keys[va], keys[~va], y[~va])
    return p
def show(name, p):
    q = dd(p); f = [round(float(np.sqrt(((y-q)[folds==k]**2).mean())), 3) for k in range(5)]
    print(f"{name:38s} {np.sqrt(((y-p)**2).mean()):.3f} | + doublons {np.mean(f):.3f}", f, flush=True)
show("stacking " + str(len(names)), honest)
al = sb.stack_frame(train, test, folds, honest, pred)
P = sb.pair_table(al, sb.candidate_pairs(al)); w = sb.fit_pair_model(P); R = sb.row_features(al, P, w)
c_tr, c_te = sb.fit_row_model(R, al, seeds=2)
show("+ correction candidats", honest + c_tr)
