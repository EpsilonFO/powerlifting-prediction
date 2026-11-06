"""Étape 25.6 : ablation de l'ensemble (leave-one-out et sélection gloutonne), stacking Ridge en validation imbriquée.
Usage : uv run python logs/analyse_0111/ablation_ensemble.py [liste de modèles, défaut : logs/ensemble_k10_v2_models.txt]"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D + "train.csv"); y = tr.TotalKg.to_numpy()
folds = tr[["id"]].merge(pd.read_csv("data/folds.csv"), on="id", how="left").fold.to_numpy()
names = open(sys.argv[1] if len(sys.argv) > 1 else "logs/ensemble_k10_v2_models.txt").read().split()
P = {n: np.load(f"outputs/oof/{n}.npy") for n in names}


def stack(sel):
    X = np.column_stack([P[n] for n in sel]); oof = np.zeros(len(y))
    for k in range(5):
        a, b = folds != k, folds == k
        oof[b] = Ridge(alpha=10.0).fit(X[a], y[a]).predict(X[b])
    return np.sqrt(np.mean((y - oof) ** 2))


full = stack(names); print(f"{len(names)} modèles : {full:.3f}")
loo = sorted(((stack([m for m in names if m != n]) - full, n) for n in names), reverse=True)
print("leave-one-out (hausse de RMSE au retrait) :")
for d, n in loo: print(f"  {n:22s} {d:+.3f}")
sel, rest = [], list(names)
print("sélection gloutonne :")
while rest:
    best = min(rest, key=lambda n: stack(sel + [n])); sel.append(best); rest.remove(best)
    print(f"  {len(sel):2d} modèles : {stack(sel):.3f}  (+{best})")
