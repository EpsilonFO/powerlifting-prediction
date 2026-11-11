"""Étape 29 : modèle au niveau de la paire + candidats élargis. Usage : uv run python archive/analyses_1111/pairs_eval.py <cache.pkl>"""
import sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from powerlifting import siblings as sb
from powerlifting.config import OOF_DIR, PRED_DIR, TARGET
from powerlifting.data import load_data
from powerlifting.postprocess import row_keys, replace_duplicates

train, test, sample = load_data()
folds = pd.read_csv("data/folds.csv").fold.to_numpy()
y = train[TARGET].to_numpy(float); ntr = len(train)
oof, pred = np.load(OOF_DIR / "ensemble_stack.npy"), np.load(PRED_DIR / "ensemble_stack.npy")
t0 = time.time()
al = sb.stack_frame(train, test, folds, oof, pred)
pairs = sb.candidate_pairs(al); print("paires orientées (train + test) :", len(pairs), f"{time.time()-t0:.0f}s", flush=True)
P = sb.pair_table(al, pairs); del pairs
print("paires de référence :", len(P), "| lignes avec candidat : train", P.i_a[P.fold_a>=0].nunique()/ntr, "test", P.i_a[P.fold_a<0].nunique()/len(test), f"{time.time()-t0:.0f}s", flush=True)
w = sb.fit_pair_model(P); print(f"modèle de paires : {time.time()-t0:.0f}s ; w quantiles", np.quantile(w, [0.01,0.5,0.9,0.99,0.999]).round(3), flush=True)
R = sb.row_features(al, P, w)
pd.to_pickle({"R": R, "w": w, "P": P[["i_a","i_b","z","fold_a","fold_b"]]}, sys.argv[1])
keys = row_keys(train)
def dd(p):
    p = p.copy()
    for k in range(5):
        va = folds == k
        p[va], _ = replace_duplicates(p[va], keys[va], keys[~va], y[~va])
    return p
def show(name, p):
    q = dd(p)
    print(f"{name:34s} {np.sqrt(((y-p)**2).mean()):.3f} | + doublons {np.mean([np.sqrt(((y-q)[folds==k]**2).mean()) for k in range(5)]):.3f}",
          [round(float(np.sqrt(((y-q)[folds==k]**2).mean())), 3) for k in range(5)], flush=True)
show("stacking", oof)
s = R.sib_sum.to_numpy()[:ntr]
show("+ somme brute des w·z", oof + s)
show("+ moyenne rétrécie", oof + R.sib_wmean.to_numpy()[:ntr])
c_tr, c_te = sb.fit_row_model(R, al)
show("+ modèle par ligne", oof + c_tr)
np.save(sys.argv[1] + ".corr_tr.npy", c_tr); np.save(sys.argv[1] + ".corr_te.npy", c_te)
print("correction : sd train", c_tr.std(), "sd test", c_te.std(), "moy test", c_te.mean(), f"| total {time.time()-t0:.0f}s")
