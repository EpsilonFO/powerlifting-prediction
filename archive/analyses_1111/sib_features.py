import numpy as np, pandas as pd, lightgbm as lgb, sys
pd.set_option("display.width", 250)
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv")
RAW = [c for c in te.columns if c != "id"]
ntr = len(tr)
tr["res"] = tr.TotalKg - np.load("outputs/oof/ensemble_stack.npy")
tr["fold"] = pd.read_csv("data/folds.csv").fold.to_numpy()
te["res"] = np.nan; te["fold"] = -1
al = pd.concat([tr, te], ignore_index=True)
al["rk"] = pd.util.hash_pandas_object(al[RAW], index=False).to_numpy()
al["i"] = np.arange(len(al))
h = (al.Age % 1) == 0.5
E = pd.concat([al[h].assign(by=(al.MeetYear - al.Age - 0.5)[h]), al[~h].assign(by=(al.MeetYear - al.Age)[~h]), al[~h].assign(by=(al.MeetYear - al.Age - 1)[~h])])
KEYS = {
  "st":   (["Sex","Country","State"], lambda e: e.State != "Unknown"),
  "stfed":(["Sex","Country","State","Federation"], lambda e: e.State != "Unknown"),
  "fedms":(["Sex","Country","Federation","MeetState"], lambda e: e.State == "Unknown"),
  "fed":  (["Sex","Country","Federation"], lambda e: e.State == "Unknown"),
}
keep = ["i","rk","BodyweightKg","MeetYear","Equipment","res","fold","Age","MeetState","Federation","Tested","WeightClassKg"]
F = pd.DataFrame(index=np.arange(len(al)))
for name,(key,mask) in KEYS.items():
    e = E[mask(E)]
    k2 = [c for c in keep if c not in key]
    m = e[key+["by"]+k2].merge(e[key+["by"]+k2], on=key+["by"], suffixes=("_a","_b"))
    m = m[(m.i_a != m.i_b) & (m.rk_a != m.rk_b)].drop_duplicates(["i_a","i_b"])
    m["dbw"] = (m.BodyweightKg_a - m.BodyweightKg_b).abs()
    m["dy"] = (m.MeetYear_a - m.MeetYear_b).abs()
    # plausibility: bodyweight drift allowed grows with years
    m = m[m.dbw <= 4 + 1.5*m.dy.clip(upper=4)]
    F[f"{name}_npool"] = m.groupby("i_a").size().reindex(F.index).fillna(0)
    r = m[(m.fold_b >= 0) & (m.fold_b != m.fold_a)].copy()   # refs: train rows of other folds (all train for test rows)
    r["sameeq"] = (r.Equipment_a == r.Equipment_b).astype(float) if "Equipment_a" in r else 1.0
    r["w"] = 1/(1+r.dbw)
    r["wres"] = r.w*r.res_b
    g = r.groupby("i_a")
    F[f"{name}_nref"] = g.size().reindex(F.index).fillna(0)
    F[f"{name}_mres"] = g.res_b.mean().reindex(F.index)
    F[f"{name}_wres"] = (g.wres.sum()/g.w.sum()).reindex(F.index)
    F[f"{name}_sdres"] = g.res_b.std().reindex(F.index)
    nn = r.sort_values(["i_a","dbw","dy"]).drop_duplicates("i_a").set_index("i_a")
    for c in ["res_b","dbw","dy","sameeq"]:
        F[f"{name}_nn_{c}"] = nn[c].reindex(F.index)
    rs = r[r.sameeq == 1].groupby("i_a")
    F[f"{name}_mres_eq"] = rs.res_b.mean().reindex(F.index); F[f"{name}_nref_eq"] = rs.size().reindex(F.index).fillna(0)
    print(name, "pairs", len(m), "rows with ref", (F[f"{name}_nref"]>0).mean(), flush=True)
    # diagnostic on train: corr of res with nn res by npool
    d = pd.DataFrame({"res": al.res, "nn": F[f"{name}_nn_res_b"], "np": F[f"{name}_npool"]}).iloc[:ntr].dropna()
    print(d.groupby(pd.cut(d.np,[0,1,2,3,5,10,30,1e9])).apply(lambda x: pd.Series({"n":len(x),"rho":np.corrcoef(x.res,x.nn)[0,1]})))
F.to_pickle(sys.argv[1])
# residual model
X = F.iloc[:ntr].copy(); y = tr.res.to_numpy(); fold = tr.fold.to_numpy()
for extra in ["Age","BodyweightKg","MeetYear"]: X[extra] = tr[extra]
X["sex"] = (tr.Sex=="M").astype(int)
base = np.sqrt((y**2).mean()); print("base rmse", base)
corr = np.zeros(ntr)
for k in range(5):
    va = fold==k
    mdl = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, verbose=-1)
    mdl.fit(X[~va], y[~va], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    corr[va] = mdl.predict(X[va])
    print(k, "iters", mdl.best_iteration_, "rmse", np.sqrt((y[va]**2).mean()), "->", np.sqrt(((y[va]-corr[va])**2).mean()), flush=True)
print("TOTAL", base, "->", np.sqrt(((y-corr)**2).mean()))
imp = pd.Series(mdl.booster_.feature_importance("gain"), X.columns).sort_values(ascending=False); print(imp.head(15))
np.save(sys.argv[1]+".corr.npy", corr)
