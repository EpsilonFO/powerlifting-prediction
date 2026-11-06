"""Assemble les prédictions TabICL des 5 plis (cible C) -> results/tabicl_C_{oof,test}.npy, puis évaluation standard."""
import numpy as np
from common import load, save, evaluate

_, _, _, F_te, folds, y = load()
oof, test = np.full(len(y), np.nan), np.zeros(len(F_te))
for k in range(5):
    src = "out_gpu1" if k == 0 else "out_gpu2"
    p = np.load(f"tabicl/{src}/tabicl_C_fold{k}.npz")
    oof[p["va"]] = p["pred_va"]
    test += p["pred_te"] / 5
save("tabicl_C", oof, test)
print(evaluate(oof))
