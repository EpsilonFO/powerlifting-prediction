"""Briques partagées par les modèles."""

import numpy as np
import pandas as pd

from powerlifting.features import TE_COLS, ResidualTargetEncoder


def pop_common(p: dict) -> tuple[dict, int, int, int]:
    """Sépare des hyperparamètres du modèle : nombre de seeds, nombre maximal d'arbres, patience de l'early stopping."""
    p = dict(p)
    return p, p.pop("seeds", 1), p.pop("n_estimators", 5000), p.pop("patience", 200)


def with_te(Xtr, ytr, Xva, Xte):
    """Ajoute les features de target encoding (apprises sur le pli d'entraînement uniquement)."""
    te = ResidualTargetEncoder(TE_COLS)
    return (
        pd.concat([Xtr, te.fit_transform(Xtr, ytr)], axis=1),
        pd.concat([Xva, te.transform(Xva)], axis=1),
        pd.concat([Xte, te.transform(Xte)], axis=1),
    )


def as_str(x):
    """Catégories en chaînes, manquants compris (« NA ») : pour le one-hot de sklearn et CatBoost."""
    return x.astype(object).fillna("NA").astype(str)


def weighted_mean(y, w):
    return float(np.average(y, weights=w))


def device(name: str) -> str:
    """« auto » -> mps (Mac), sinon cuda, sinon cpu."""
    if name != "auto":
        return name
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
