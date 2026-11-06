import time, numpy as np, torch
from tabicl import TabICLRegressor
d = np.load("tabicl/data/base.npz", allow_pickle=True); te = np.load("tabicl/data/te.npy"); m = np.load("tabicl/data/meta.npz")
X = np.hstack([d["X"], te[0]]); folds, yC, n_tr = m["folds"], m["yC"], int(m["n_tr"])
rng = np.random.default_rng(0)
tr = rng.choice(np.where(folds != 0)[0], 10000, replace=False); va = np.where(folds == 0)[0][:3000]
for dev in ["mps", "cpu"]:
    t = time.time()
    r = TabICLRegressor(device=dev, random_state=0)
    r.fit(X[tr], yC[tr]); p = r.predict(X[va])
    kg = p * m["dots_den"][va] / 500
    print(dev, f"{time.time()-t:.1f}s", "rmse kg", np.sqrt(np.mean((kg - m["y"][va])**2)))
