"""Moyenne de 4 modèles de sous-plis (75 % des données) vs un seul modèle sur 100 % (arbres fixés), jugés sur un pli
mis de côté. Mesure directement le gain attendu d'un réentraînement sur tout le train."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
from cv import prepare, load_params, rmse
import models
from features import TargetTransform
from config import SEED, TARGET
train, test, sample, F_tr, F_te, folds = prepare()
p = {**models.MODELS["lgbm"]["params"], **load_params("lgbm")}
tt = TargetTransform("C"); y = train[TARGET].to_numpy(float)
for k in (0, 1):
    te_idx = np.where(folds == k)[0]; Xte = F_tr.iloc[te_idx].reset_index(drop=True)
    sub = [f for f in range(5) if f != k]
    pte_avg, best = np.zeros(len(te_idx)), []
    for f in sub:
        tr = np.where((folds != k) & (folds != f))[0]; va = np.where(folds == f)[0]
        Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
        ytr, yva = tt.forward(y[tr], Xtr), tt.forward(y[va], Xva); wtr, wva = tt.weight(Xtr), tt.weight(Xva)
        Xtr2, Xva2, Xte2 = models._with_te(Xtr, ytr, Xva, Xte)
        m = lgb.LGBMRegressor(n_estimators=5000, random_state=SEED, n_jobs=-1, verbose=-1, **p)
        m.fit(Xtr2, ytr, sample_weight=wtr, eval_X=Xva2, eval_y=yva, eval_sample_weight=[wva], callbacks=[lgb.early_stopping(200, verbose=False)])
        best.append(m.best_iteration_); pte_avg += tt.inverse(m.predict(Xte2), Xte) / len(sub)
        print(f"  fold{k} sub{f} best_iter={m.best_iteration_} rmse_sub={rmse(y[va], tt.inverse(m.predict(Xva2), Xva)):.3f}", flush=True)
    r_avg = rmse(y[te_idx], pte_avg)
    tr = np.where(folds != k)[0]; Xtr = F_tr.iloc[tr].reset_index(drop=True)
    ytr = tt.forward(y[tr], Xtr); wtr = tt.weight(Xtr)
    Xtr2, _, Xte2 = models._with_te(Xtr, ytr, Xte, Xte)
    for mult in (1.0, 1.25):
        n = int(np.mean(best) * mult)
        m = lgb.LGBMRegressor(n_estimators=n, random_state=SEED, n_jobs=-1, verbose=-1, **p).fit(Xtr2, ytr, sample_weight=wtr)
        print(f"RESULT fold{k}: moyenne 4 sous-plis (63k lignes) = {r_avg:.3f} | 1 modèle sur 84k, {n} arbres (x{mult}) = {rmse(y[te_idx], tt.inverse(m.predict(Xte2), Xte)):.3f}", flush=True)
