"""Gradient boosting : M4 LightGBM, M5 CatBoost, M6 XGBoost. Nombre d'arbres fixé par early stopping sur le pli de
validation ; la prédiction est la moyenne de `seeds` modèles."""

import pandas as pd

from powerlifting.config import SEED
from powerlifting.features import CAT_COLS, MEET
from powerlifting.models.common import as_str, pop_common, with_te


def lgbm_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    import lightgbm as lgb

    p, seeds, n_est, patience = pop_common(p)
    n_jobs, tabprep = p.pop("n_jobs", -1), p.pop("tabprep", False)
    new = None
    if tabprep:  # variables relatives TabPrep (journal, étape 26) : --set tabprep=true
        from powerlifting.tabprep import tabprep_features
        new = tabprep_features(Xtr, Xva, Xte)
    Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
    if new is not None:
        Xtr, Xva, Xte = (pd.concat([X, n], axis=1) for X, n in zip((Xtr, Xva, Xte), new))
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = lgb.LGBMRegressor(n_estimators=n_est, random_state=SEED + s, n_jobs=n_jobs, verbose=-1, **p)
        m.fit(
            Xtr, ytr, sample_weight=wtr, eval_X=Xva, eval_y=yva,
            eval_sample_weight=None if wva is None else [wva],
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
        pva, pte = pva + m.predict(Xva) / seeds, pte + m.predict(Xte) / seeds
    return pva, pte


def cat_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    from catboost import CatBoostRegressor, Pool

    p, seeds, n_est, patience = pop_common(p)
    meet_cats = p.pop("meet_cats", False)  # clé de compétition en catégorielles natives (journal, étape 25.5)
    cat_cols = CAT_COLS + (["Meet", "Meet_Div"] if meet_cats else [])

    def as_pool(X, y=None, w=None):
        X = X.copy()
        if meet_cats:
            X["Meet"] = as_str(X[MEET]).agg("|".join, axis=1)
            X["Meet_Div"] = X["Meet"] + "|" + as_str(X["Sex"]) + "|" + as_str(X["Equipment"])
        for c in cat_cols:  # encodage ordonné natif de CatBoost : pas de TE manuel
            X[c] = as_str(X[c])
        return Pool(X, y, cat_features=cat_cols, weight=w)

    ptr, pva_pool, pte_pool = as_pool(Xtr, ytr, wtr), as_pool(Xva, yva, wva), as_pool(Xte)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = CatBoostRegressor(
            iterations=n_est, loss_function="RMSE", od_type="Iter", od_wait=patience, random_seed=SEED + s,
            thread_count=-1, verbose=0, allow_writing_files=False, **p,
        )
        m.fit(ptr, eval_set=pva_pool, use_best_model=True)
        pva, pte = pva + m.predict(pva_pool) / seeds, pte + m.predict(pte_pool) / seeds
    return pva, pte


def xgb_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    import xgboost as xgb

    p, seeds, n_est, patience = pop_common(p)
    Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = xgb.XGBRegressor(
            n_estimators=n_est, tree_method="hist", enable_categorical=True, early_stopping_rounds=patience,
            random_state=SEED + s, n_jobs=-1, **p,
        )
        m.fit(
            Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)],
            sample_weight_eval_set=None if wva is None else [wva], verbose=False,
        )
        pva, pte = pva + m.predict(Xva) / seeds, pte + m.predict(Xte) / seeds
    return pva, pte
