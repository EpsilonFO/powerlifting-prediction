"""Soumission prototype : stacking + correction « candidats même athlète » + doublons (étape 28)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from powerlifting.postprocess import row_keys, replace_duplicates
D = "data/competitions/performances-en-powerlifting/"
tr = pd.read_csv(D+"train.csv"); te = pd.read_csv(D+"test.csv"); ntr = len(tr)
F = pd.read_pickle(sys.argv[1])
y = tr.TotalKg.to_numpy(float)
def X_of(F, d):
    X = F.reset_index(drop=True).copy()
    for c in ["Age","BodyweightKg","MeetYear"]: X[c] = d[c].to_numpy()
    X["sex"] = (d.Sex=="M").astype(int).to_numpy()
    return X
Xtr, Xte = X_of(F.iloc[:ntr], tr), X_of(F.iloc[ntr:], te)
res = y - np.load("outputs/oof/ensemble_stack.npy")
corr = np.zeros(len(te))
for seed in range(5):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, random_state=seed, verbose=-1).fit(Xtr, res)
    corr += m.predict(Xte)/5
base = np.load("outputs/preds/ensemble_stack.npy")
pred, hit = replace_duplicates(base + corr, row_keys(te), row_keys(tr), y)
old = pd.read_csv("outputs/submissions/ensemble_stack_dedup.csv")
assert (old.id.to_numpy() == te.id.to_numpy()).all()
out = pd.DataFrame({"id": te.id, "TotalKg": pred}); out.to_csv(sys.argv[2], index=False)
print("lignes", len(out), "NaN", out.TotalKg.isna().sum(), "doublons remplacés", hit.sum())
print("correction test : sd", corr.std(), "moyenne", corr.mean(), "| part |corr|>20 :", (abs(corr)>20).mean())
print("écart RMS avec la soumission actuelle :", np.sqrt(((out.TotalKg-old.TotalKg)**2).mean()))
