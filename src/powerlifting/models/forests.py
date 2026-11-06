"""M3 : Random Forest et ExtraTrees (scikit-learn)."""

from powerlifting.config import SEED
from powerlifting.features import CAT_COLS
from powerlifting.models.common import with_te


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
    Xtr, Xva, Xte = with_te(Xtr, ytr, Xva, Xte)
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
