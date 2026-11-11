import sys, os, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from powerlifting import siblings as sb
from powerlifting.config import OOF_DIR, PRED_DIR, TARGET
from powerlifting.data import load_data
cache = sys.argv[1]
if not os.path.exists(cache):
    train, test, sample = load_data(); folds = pd.read_csv("data/folds.csv").fold.to_numpy()
    al = sb.stack_frame(train, test, folds, np.load(OOF_DIR/"ensemble_stack.npy"), np.load(PRED_DIR/"ensemble_stack.npy"))
    P = sb.pair_table(al, sb.candidate_pairs(al)); pd.to_pickle((al, P), cache)
al, P = pd.read_pickle(cache)
feats = [c for c in P.columns if c not in sb.NOT_FEATURES]
z = P.z.to_numpy(float); zc = np.where(np.abs(z) < 1, np.where(z < 0, -1.0, 1.0), z)
r = P.res_a.to_numpy(float); t = np.clip(r/zc, -50, 50); wgt = zc**2/np.mean(zc**2)
fa, fb = P.fold_a.to_numpy(), P.fold_b.to_numpy()
k = 0; tr = (fa >= 0) & (fa != k) & (fb != k); va = fa == k
X = P[feats]
base = np.mean(r[va]**2)
configs = {
 "A leaves15 mcs1000 lr.03": dict(num_leaves=15, min_child_samples=1000, learning_rate=0.03, reg_lambda=100, colsample_bytree=0.5, subsample=0.7, subsample_freq=1),
 "B leaves7 mcs3000 lr.03":  dict(num_leaves=7, min_child_samples=3000, learning_rate=0.03, reg_lambda=100, colsample_bytree=0.5, subsample=0.7, subsample_freq=1),
 "C leaves31 mcs3000 lr.02": dict(num_leaves=31, min_child_samples=3000, learning_rate=0.02, reg_lambda=300, colsample_bytree=0.5, subsample=0.7, subsample_freq=1),
}
for name, p in configs.items():
    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=800, verbose=-1, n_jobs=8, **p).fit(X[tr], t[tr], sample_weight=wgt[tr])
    out = []
    for n in [25, 50, 100, 200, 400, 800]:
        w = m.predict(X[va], num_iteration=n)
        out.append(f"{n}:{base - np.mean((r[va]-w*z[va])**2):+.2f}")
    print(name, "gain MSE paires (positif = mieux)", " ".join(out), f"{time.time()-t0:.0f}s", flush=True)
imp = pd.Series(m.booster_.feature_importance("gain"), feats).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(20).to_dict())
