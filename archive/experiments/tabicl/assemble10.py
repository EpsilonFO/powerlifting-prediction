"""Assemble les 10 sous-plis TabICL (cible C) -> results/tabicl_C_k10_{oof,test}.npy, puis évaluation standard."""
import numpy as np
from common import load, save, evaluate

_, _, _, F_te, folds, y = load()
oof, test = np.full(len(y), np.nan), np.zeros(len(F_te))
for j in range(10):
    p = np.load(f"tabicl/out_k10/tabicl10_C_sub{j}.npz")
    assert np.isnan(oof[p["va"]]).all()
    oof[p["va"]] = p["pred_va"]
    test += p["pred_te"] / 10
assert not np.isnan(oof).any()
save("tabicl_C_k10", oof, test)
print(evaluate(oof))
