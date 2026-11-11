import sys, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from powerlifting import siblings as sb
from powerlifting.postprocess import row_keys, replace_duplicates
from powerlifting.data import load_data
from powerlifting.config import TARGET
S = sys.argv[1]
al, P = pd.read_pickle(S + "/P_v1.pkl"); w = pd.read_pickle(S + "/pairs_v2.pkl")["w"]
train, test, _ = load_data(); y = train[TARGET].to_numpy(float); ntr = len(train); folds = al.fold.to_numpy()[:ntr]
oof = al.pred.to_numpy()[:ntr]; keys = row_keys(train)
def dd(p):
    p = p.copy()
    for k in range(5):
        va = folds == k; p[va], _ = replace_duplicates(p[va], keys[va], keys[~va], y[~va])
    return p
def ev(name, R, seeds=2):
    X = R.iloc[:ntr]; res = y - oof; c = np.zeros(ntr)
    for k in range(5):
        va = folds == k
        for s in range(seeds): c[va] += lgb.LGBMRegressor(**sb.ROW_PARAMS, random_state=s).fit(X[~va], res[~va]).predict(X[va])/seeds
    q = dd(oof + c)
    print(f"{name:40s} {np.mean([np.sqrt(((y-q)[folds==k]**2).mean()) for k in range(5)]):.3f}", [round(float(np.sqrt(((y-q)[folds==k]**2).mean())),3) for k in range(5)], flush=True)
z = P.z.to_numpy(float); zc = np.where(np.abs(z) < 1, np.where(z < 0, -1.0, 1.0), z); r = P.res_a.to_numpy(float)
t = np.clip(r/zc, -50, 50); fa, fb = P.fold_a.to_numpy(), P.fold_b.to_numpy()
wc = np.zeros(len(P))
for k in range(5):
    tr = (fa >= 0) & (fa != k) & (fb != k)
    iso = IsotonicRegression(out_of_bounds="clip").fit(w[tr], t[tr], sample_weight=zc[tr]**2)
    wc[fa == k] = iso.predict(w[fa == k]); wc[fa == -1] += iso.predict(w[fa == -1])/5
    if k == 0: print("calibration (w -> w') :", {q: round(float(iso.predict([q])[0]), 3) for q in [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]})
va = fa >= 0
print("MSE paires : brut", np.mean(r[va]**2).round(2), "w", np.mean((r-w*z)[va]**2).round(2), "w calibré", np.mean((r-wc*z)[va]**2).round(2))
ev("ligne, w brut", sb.row_features(al, P, w))
Rc = sb.row_features(al, P, wc)
ev("ligne, w calibré", Rc)
# somme de type « noisy-or » : espérance du résidu si au plus un vrai candidat par valeur
R3 = Rc.copy(); R0 = sb.row_features(al, P, w)
for c in ["sib_sum", "sib_w_sum", "sib_w_max", "sib_wmean", "sib_top_z", "sib_top_wz"]: R3[c + "_raw"] = R0[c]
ev("ligne, w brut + calibré", R3)
pd.to_pickle((wc, R3), S + "/calib.pkl")
