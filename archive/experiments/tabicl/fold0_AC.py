import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from common import load, ensemble_members, OOF_DIR, rmse
from ensemble import _meta_features
_, _, F_tr, _, folds, y = load()
va = np.where(folds == 0)[0]
O = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in ensemble_members()])[va]
Z = _meta_features(O, F_tr.iloc[va])
C = np.load("tabicl/out_gpu1/tabicl_C_fold0.npz")["pred_va"][:, None]
A = np.load("tabicl/out_gpu1/tabicl_A_fold0.npz")["pred_va"][:, None]
for lab, M in [("base", Z), ("+C", np.hstack([Z, C])), ("+C+A", np.hstack([Z, C, A]))]:
    h = np.zeros(len(va))
    for a, b in KFold(5, shuffle=True, random_state=0).split(M):
        h[b] = Ridge(alpha=10).fit(M[a], y[va][a]).predict(M[b])
    print(lab, round(rmse(y[va], h), 4))
