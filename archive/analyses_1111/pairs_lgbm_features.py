import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from powerlifting.cv import prepare, run_cv, load_params
S, name, use = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
train, test, sample, F_tr, F_te, folds = prepare()
if use:
    R = pd.read_pickle(S + "/R2.pkl"); al, _ = pd.read_pickle(S + "/P_v1.pkl")
    R = R.drop(columns=["Age", "BodyweightKg", "MeetYear", "male"])
    R = R.astype(np.float32); ntr = len(train)
    F_tr = pd.concat([F_tr, R.iloc[:ntr].set_index(F_tr.index)], axis=1); F_te = pd.concat([F_te, R.iloc[ntr:].set_index(F_te.index)], axis=1)
p = load_params("lgbm"); p.update(tabprep=True, n_jobs=4)
r = run_cv(name, "lgbm", train, test, F_tr, F_te, folds, target="C", seeds=1, params=p, save=False)
np.save(f"{S}/{name}_oof.npy", r["oof"]); np.save(f"{S}/{name}_pred.npy", r["pred"])
print("FOLDS", r["fold_rmse"], r["cv_rmse"])
