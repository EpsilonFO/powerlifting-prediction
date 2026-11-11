import numpy as np, pandas as pd, lightgbm as lgb, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from powerlifting.postprocess import row_keys, replace_duplicates
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); ntr = len(tr)
F = pd.read_pickle(sys.argv[1])
oof = np.load("outputs/oof/ensemble_stack.npy"); y = tr.TotalKg.to_numpy(float); fold = pd.read_csv("data/folds.csv").fold.to_numpy()
X = F.iloc[:ntr].copy()
for c in ["Age","BodyweightKg","MeetYear"]: X[c] = tr[c]
X["sex"] = (tr.Sex=="M").astype(int)
res = y - oof; corr = np.zeros(ntr)
for k in range(5):
    va = fold==k
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, verbose=-1).fit(X[~va], res[~va])
    corr[va] = m.predict(X[va])
new = oof + corr
keys = row_keys(tr)
def dd(p):
    p = p.copy()
    for k in range(5):
        va = fold==k
        p[va], _ = replace_duplicates(p[va], keys[va], keys[~va], y[~va])
    return p
r = lambda p: np.sqrt(((y-p)**2).mean())
rf = lambda p: [round(float(np.sqrt(((y-p)[fold==k]**2).mean())),3) for k in range(5)]
print("stack           ", r(oof), rf(oof))
print("stack+dedup     ", r(dd(oof)), rf(dd(oof)))
print("stack+sib       ", r(new), rf(new))
print("stack+sib+dedup ", r(dd(new)), rf(dd(new)))
has = (F.iloc[:ntr].filter(like="_nref").sum(axis=1) > 0).to_numpy()
print("rows with >=1 candidate: train", has.mean(), "test", (F.iloc[ntr:].filter(like="_nref").sum(axis=1) > 0).mean())
print("RMSE on rows with candidate:", r(oof[has]) if False else np.sqrt(((y-oof)[has]**2).mean()), "->", np.sqrt(((y-new)[has]**2).mean()), "| without:", np.sqrt(((y-oof)[~has]**2).mean()), "->", np.sqrt(((y-new)[~has]**2).mean()))
print("sd of correction:", corr.std(), "share |corr|>20:", (abs(corr)>20).mean())
