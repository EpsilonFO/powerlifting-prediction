"""Courbe d'apprentissage : LightGBM tuné (1 seed, cible C) sur des fractions du train du pli, validé sur le pli complet."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
from cv import prepare, run_cv, load_params
train, test, sample, F_tr, F_te, folds = prepare()
p = load_params("lgbm"); rng = np.random.default_rng(0)
for k in (0, 1):
    for frac in (0.5, 0.75, 1.0):
        keep = (folds == k) | (rng.random(len(folds)) < frac)
        r = run_cv("lc", "lgbm", train[keep].reset_index(drop=True), test, F_tr[keep].reset_index(drop=True), F_te,
                   folds[keep], target="C", seeds=1, params=dict(p), only_folds=[k], save=False, verbose=False)
        print(f"RESULT lc fold{k} frac_of_fold_train={frac:.2f} n_train={int(((folds!=k)&keep).sum())} rmse={r['cv_rmse']:.3f}", flush=True)
