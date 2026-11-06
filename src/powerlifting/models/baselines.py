"""Repères et modèle linéaire : constante, HistGradientBoosting brut (journal, étape 6), Ridge + splines (M1)."""

import numpy as np
import pandas as pd

from powerlifting.config import SEED
from powerlifting.features import CAT_COLS, RAW_CATS, RAW_FEATURES
from powerlifting.models.common import as_str, weighted_mean


def const_fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, p):
    m = weighted_mean(ytr, wtr)
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
    """HistGradientBoosting « brut » sur les 16 variables d'origine (repère ≈ 79)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xs = _collapse([X[RAW_FEATURES] for X in (Xtr, Xva, Xte)], RAW_CATS, 200)
    m = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=500, random_state=SEED)
    m.fit(Xs[0], ytr, sample_weight=wtr)
    return m.predict(Xs[1]), m.predict(Xs[2])


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

    A = numeric(Xtr)
    spline = SplineTransformer(n_knots=10, degree=3, knots="quantile", extrapolation="linear").fit(A[spl_cols])
    scaler = StandardScaler().fit(A[lin_cols])
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25).fit(as_str(Xtr[CAT_COLS]))
    ohe_se = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200).fit(as_str(Xtr[["Sex_Equip"]]))

    def design(X):
        A = numeric(X)
        spl = sp.csr_matrix(spline.transform(A[spl_cols]))
        base = sp.hstack([spl, sp.csr_matrix(scaler.transform(A[lin_cols])), ohe.transform(as_str(X[CAT_COLS]))]).tocsr()
        female = X["is_female"].to_numpy(dtype=float)[:, None]
        blocks = [base.multiply(1 - female), base.multiply(female)]  # Sex × tout
        se = ohe_se.transform(as_str(X[["Sex_Equip"]])).tocsc()
        blocks += [spl.multiply(se[:, j].toarray()) for j in range(se.shape[1])]  # (Sex × Equipment) × splines
        return sp.hstack(blocks).tocsr()

    m = Ridge(alpha=p.get("alpha", 10.0)).fit(design(Xtr), ytr, sample_weight=wtr)
    return m.predict(design(Xva)), m.predict(design(Xte))
