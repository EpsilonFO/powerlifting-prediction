"""Signal précoce sur le pli 0 seul : corrélation des résidus et stacking Ridge en CV interne (5 sous-plis du pli 0)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from common import load, ensemble_members, OOF_DIR, rmse
from ensemble import _meta_features

_, _, F_tr, _, folds, y = load()
va = np.where(folds == 0)[0]
names = ensemble_members()
O = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in names])[va]
Z = _meta_features(O, F_tr.iloc[va])
for tgt in "CA":
    p = np.load(f"tabicl/out_gpu1/tabicl_{tgt}_fold0.npz")
    assert (p["va"] == va).all()
    t = p["pred_va"]
    r = y[va] - t
    print(tgt, "rmse", round(rmse(y[va], t), 3), "corr", {n: round(float(np.corrcoef(r, y[va] - O[:, names.index(n)])[0, 1]), 3)
          for n in ["lgbm_C_k10", "mlp_C_plr_te_k10", "cat_C_meet_k10", "xgb_C_k10"]})
    for lab, M in [("base", Z), ("+tabicl", np.hstack([Z, t[:, None]]))]:
        h = np.zeros(len(va))
        for a, b in KFold(5, shuffle=True, random_state=0).split(M):
            h[b] = Ridge(alpha=10).fit(M[a], y[va][a]).predict(M[b])
        print("   stack", lab, round(rmse(y[va], h), 4))
