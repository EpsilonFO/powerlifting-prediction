"""Phase 4 : les modèles. Tous partagent la même signature :

    fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, params) -> (prédiction validation, prédiction test)

Xtr / Xva / Xte sont les features de `features.build_features` (catégories en dtype `category`),
ytr / yva la cible déjà transformée, wtr / wva des poids (ou None). `Xva` sert à l'early stopping
des GBM / MLP, comme prévu dans le plan (nombre d'arbres fixé par early stopping sur chaque pli).
"""

import numpy as np
import pandas as pd

from config import SEED
from features import CAT_COLS, MEET, RAW_CATS, RAW_FEATURES, TE_COLS, ResidualTargetEncoder


def _pop_common(p: dict) -> tuple[dict, int, int, int]:
    p = dict(p)
    return p, p.pop("seeds", 1), p.pop("n_estimators", 5000), p.pop("patience", 200)


def _with_te(Xtr, ytr, Xva, Xte):
    """Ajoute les features de target encoding (apprises sur le pli d'entraînement uniquement)."""
    te = ResidualTargetEncoder(TE_COLS)
    return (
        pd.concat([Xtr, te.fit_transform(Xtr, ytr)], axis=1),
        pd.concat([Xva, te.transform(Xva)], axis=1),
        pd.concat([Xte, te.transform(Xte)], axis=1),
    )


def _weighted_mean(y, w):
    return float(np.average(y, weights=w))


# ---------------------------------------------------------------- références

def const_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    m = _weighted_mean(ytr, wtr)
    return np.full(len(Xva), m), np.full(len(Xte), m)


def _collapse(Xs, cols, top_k):
    """Garde les `top_k` modalités les plus fréquentes du 1er jeu, regroupe le reste en RARE."""
    outs = [X.copy() for X in Xs]
    for c in cols:
        keep = list(Xs[0][c].value_counts().index[:top_k])
        for X in outs:
            s = X[c].astype(object)
            X[c] = pd.Categorical(s.where(s.isin(keep) | s.isna(), "RARE"), categories=keep + ["RARE"])
    return outs


def hgb_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """HistGradientBoosting « brut » sur les 16 variables d'origine (référence ≈ 79 du plan)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xs = _collapse([X[RAW_FEATURES] for X in (Xtr, Xva, Xte)], RAW_CATS, 200)
    m = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=500, random_state=SEED)
    m.fit(Xs[0], ytr, sample_weight=wtr)
    return m.predict(Xs[1]), m.predict(Xs[2])


# ---------------------------------------------------------------- M1 : Ridge + splines + interactions

def ridge_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """Ridge sur one-hot + splines (poids de corps, âge) ; tous les termes sont dupliqués par sexe (interaction
    Sex × *) et les splines sont croisées avec Sex × Equipment."""
    import scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    spl_cols = ["BodyweightKg", "Age"]
    lin_cols = [c for c in Xtr.columns if c not in CAT_COLS and c not in spl_cols]
    num_cols = spl_cols + lin_cols
    lo, hi = Xtr[num_cols].quantile(0.005), Xtr[num_cols].quantile(0.995)  # winsorisation
    med = Xtr[lin_cols].median()

    def numeric(X):
        A = X[num_cols].astype(float).clip(lo, hi, axis=1)
        A[lin_cols] = A[lin_cols].fillna(med)
        return A

    def as_str(X, cols):
        return X[cols].astype(object).fillna("NA").astype(str)

    A = numeric(Xtr)
    spline = SplineTransformer(n_knots=10, degree=3, knots="quantile", extrapolation="linear").fit(A[spl_cols])
    scaler = StandardScaler().fit(A[lin_cols])
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25).fit(as_str(Xtr, CAT_COLS))
    ohe_se = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200).fit(as_str(Xtr, ["Sex_Equip"]))

    def design(X):
        A = numeric(X)
        spl = sp.csr_matrix(spline.transform(A[spl_cols]))
        base = sp.hstack([spl, sp.csr_matrix(scaler.transform(A[lin_cols])), ohe.transform(as_str(X, CAT_COLS))]).tocsr()
        female = X["is_female"].to_numpy(dtype=float)[:, None]
        blocks = [base.multiply(1 - female), base.multiply(female)]  # Sex × tout
        se = ohe_se.transform(as_str(X, ["Sex_Equip"])).tocsc()
        blocks += [spl.multiply(se[:, j].toarray()) for j in range(se.shape[1])]  # (Sex × Equipment) × splines
        return sp.hstack(blocks).tocsr()

    m = Ridge(alpha=p.get("alpha", 10.0)).fit(design(Xtr), ytr, sample_weight=wtr)
    return m.predict(design(Xva)), m.predict(design(Xte))


# ---------------------------------------------------------------- M3 : Random Forest / ExtraTrees

def _numeric_view(X, max_cats=20):
    """Pour les forêts sklearn : catégories peu nombreuses -> codes, les autres sont portées par TE / fréquences."""
    out = X.copy()
    for c in CAT_COLS:
        if len(X[c].cat.categories) <= max_cats:
            out[c] = X[c].cat.codes.astype(float).where(X[c].notna())
        else:
            out = out.drop(columns=c)
    return out


def _forest(cls, Xtr, ytr, wtr, Xva, Xte, p):
    Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
    Xtr, Xva, Xte = _numeric_view(Xtr), _numeric_view(Xva), _numeric_view(Xte)
    m = cls(random_state=SEED, n_jobs=-1, **{k: v for k, v in p.items() if k != "seeds"})
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xva), m.predict(Xte)


def rf_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    from sklearn.ensemble import RandomForestRegressor
    return _forest(RandomForestRegressor, Xtr, ytr, wtr, Xva, Xte, p)


def et_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    from sklearn.ensemble import ExtraTreesRegressor
    return _forest(ExtraTreesRegressor, Xtr, ytr, wtr, Xva, Xte, p)


# ---------------------------------------------------------------- M4 : LightGBM

def lgbm_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    import lightgbm as lgb

    p, seeds, n_est, patience = _pop_common(p)
    n_jobs, tabprep = p.pop("n_jobs", -1), p.pop("tabprep", False)
    new = None
    if tabprep:  # variables relatives TabPrep (journal, étape 26) : --set tabprep=true
        from tabprep import tabprep_features
        new = tabprep_features(Xtr, Xva, Xte)
    Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
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


# ---------------------------------------------------------------- M5 : CatBoost

def cat_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    from catboost import CatBoostRegressor, Pool

    p, seeds, n_est, patience = _pop_common(p)
    meet_cats = p.pop("meet_cats", False)  # clé de compétition en catégorielles natives (journal, étape 25.5)
    cat_cols = CAT_COLS + (["Meet", "Meet_Div"] if meet_cats else [])

    def as_str(s):
        return s.astype(object).fillna("NA").astype(str)

    def as_pool(X, y=None, w=None):
        X = X.copy()
        if meet_cats:
            X["Meet"] = X[MEET].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
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


# ---------------------------------------------------------------- M6 : XGBoost

def xgb_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    import xgboost as xgb

    p, seeds, n_est, patience = _pop_common(p)
    Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
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


# ---------------------------------------------------------------- M7 : MLP tabulaire avec embeddings

def nn_arrays(Xtr, Xva, Xte, min_count: int = 5):
    """Prétraitement des réseaux, ajusté sur le pli d'entraînement : une matrice float32 par jeu,
    `[codes des catégories | numériques écrêtées (0,5 %-99,5 %) et standardisées | indicatrices de manquants]`.
    Codes : 0 = modalité rare (< min_count) ou inconnue, 1 = manquant. Retourne aussi la cardinalité de chaque
    catégorie et les colonnes numériques continues (> 2 valeurs), candidates au plongement PLR."""
    num_cols = [c for c in Xtr.columns if c not in CAT_COLS]
    nan_cols = [c for c in num_cols if Xtr[c].isna().any()]
    med, lo, hi = Xtr[num_cols].median(), Xtr[num_cols].quantile(0.005), Xtr[num_cols].quantile(0.995)

    def clipped(X):
        return X[num_cols].astype(float).fillna(med).clip(lo, hi, axis=1)

    mu, sd = clipped(Xtr).mean(), clipped(Xtr).std().replace(0, 1)
    vocab = {}
    for c in CAT_COLS:
        vc = Xtr[c].value_counts()
        vocab[c] = {v: i + 2 for i, v in enumerate(vc.index[vc >= min_count])}

    def encode(X):
        codes = [X[c].astype(object).pipe(lambda s: s.map(vocab[c]).where(s.notna(), 1).fillna(0)) for c in CAT_COLS]
        num = ((clipped(X) - mu) / sd).to_numpy()
        return np.hstack([np.stack(codes, 1).astype(float), num, X[nan_cols].isna().to_numpy(float)]).astype(np.float32)

    cards = [len(vocab[c]) + 2 for c in CAT_COLS]
    cont_idx = [i for i, c in enumerate(num_cols) if Xtr[c].nunique() > 2]
    return encode(Xtr), encode(Xva), encode(Xte), cards, cont_idx


def mlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """Embeddings des catégories + MLP (briques dans `tabnn.py`). `plr` (dict k / d / sigma) active le plongement
    périodique des variables continues ; `history` (liste) reçoit les courbes d'apprentissage de la 1re seed."""
    import torch

    from tabnn import TabularInput, TabularMLP, fit, predict

    p = dict(p)
    seeds, hidden, drop = p.pop("seeds", 1), p.pop("hidden", (256, 128, 64)), p.pop("dropout", 0.15)
    plr, history, min_count = p.pop("plr", None), p.pop("history", None), p.pop("min_count", 5)
    torch.set_num_threads(8)

    if p.pop("te", False):  # target encoding en entrée numérique (journal, étape 25.5) : --set te=true
        Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
    Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, Xte, min_count)
    Atr, Ava, Ate = torch.from_numpy(Atr), torch.from_numpy(Ava), torch.from_numpy(Ate)
    n_num = Atr.shape[1] - len(cards)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        torch.manual_seed(SEED + s)
        net = TabularMLP(TabularInput(cards, n_num, cont_idx, plr), hidden, drop)
        fit(net, Atr, ytr, wtr, Ava, yva, wva, history=history if s == 0 else None, **p)
        pva, pte = pva + predict(net, Ava) / seeds, pte + predict(net, Ate) / seeds
    return pva, pte


# ---------------------------------------------------------------- étape 26 : réseaux récents

def _device(name: str) -> str:
    """« auto » -> mps (Mac), sinon cuda, sinon cpu."""
    if name != "auto":
        return name
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def tabm_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """TabM (BatchEnsemble de 32 têtes, plongements linéaires par morceaux), briques dans `tabm_net.py`.
    Défauts retenus dans `experiments/tabm/REPORT.md` : catégories en embeddings, target encoding."""
    from tabm_net import fit_tabm

    p = dict(p)
    seeds = p.pop("seeds", 1)
    p["device"] = _device(p.get("device", "auto"))
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        a, b = fit_tabm(Xtr, ytr, wtr, Xva, yva, wva, Xte, {**p, "seed": SEED + s})
        pva, pte = pva + a / seeds, pte + b / seeds
    return pva, pte


def realmlp_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    """RealMLP (Holzmüller et al., NeurIPS 2024), réglages par défaut « TD » de pytabkit. pytabkit n'accepte pas de
    poids d'échantillon : avec la cible C, la perte est la MSE en points DOTS (meilleure que A quand même, étape 26).
    Le pli de validation choisit la meilleure époque. NaN numériques : médiane du pli + indicatrices `na_*`."""
    import torch
    from pytabkit import RealMLP_TD_Regressor

    p = dict(p)
    seeds, threads = p.pop("seeds", 1), p.pop("threads", 2)
    p["device"] = _device(p.get("device", "auto"))
    torch.set_num_threads(threads)
    if p.pop("te", True):
        Xtr, Xva, Xte = _with_te(Xtr, ytr, Xva, Xte)
    num = [c for c in Xtr.columns if not isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    nan_cols = [c for c in num if Xtr[c].isna().any()]
    med = Xtr[num].median()

    def prep(X):
        X = X.copy()
        for c in nan_cols:
            X[f"na_{c}"] = X[c].isna().astype(np.float32)
        X[num] = X[num].astype(float).fillna(med)
        return X

    Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = RealMLP_TD_Regressor(random_state=SEED + s, n_threads=threads, **p)
        m.fit(Xtr, np.asarray(ytr, float), X_val=Xva, y_val=np.asarray(yva, float))
        pva = pva + np.asarray(m.predict(Xva)).ravel() / seeds
        pte = pte + np.asarray(m.predict(Xte)).ravel() / seeds
    return pva, pte


# ---------------------------------------------------------------- registre

MODELS = {
    "const": {"fn": const_fn},
    "hgb": {"fn": hgb_fn},
    "ridge": {"fn": ridge_fn, "params": {"alpha": 10.0}},
    "rf": {"fn": rf_fn, "params": {"n_estimators": 300, "min_samples_leaf": 5, "max_features": 0.4, "max_samples": 0.8}},
    "et": {"fn": et_fn, "params": {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.6}},
    "lgbm": {"fn": lgbm_fn, "params": {
        "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 30, "colsample_bytree": 0.7,
        "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 5.0, "cat_smooth": 20, "cat_l2": 10,
        "min_data_per_group": 50, "max_cat_threshold": 32,
    }},
    "cat": {"fn": cat_fn, "params": {"learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 5.0}},
    "xgb": {"fn": xgb_fn, "params": {
        "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 5, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_lambda": 5.0, "max_cat_to_onehot": 4,
    }},
    "mlp": {"fn": mlp_fn},
    "tabm": {"fn": tabm_fn, "params": {"cat": "emb", "te": True, "device": "auto"}},
    "realmlp": {"fn": realmlp_fn, "params": {"te": True, "device": "auto"}},
}
