"""Génération et contrôle des fichiers de soumission (`id,TotalKg`)."""

import numpy as np
import pandas as pd

from powerlifting.config import ID, SUB_DIR, TARGET

MIN_KG, MAX_KG = 20.0, 1400.0


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Contrôles : même nombre de lignes et mêmes ids que sample_submission, aucun NaN,
    prédictions plausibles. Lève AssertionError au premier défaut."""
    assert list(sub.columns) == list(sample.columns), f"colonnes {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} lignes au lieu de {len(sample)}"
    assert sub[ID].is_unique and set(sub[ID]) == set(sample[ID]), "ids différents de sample_submission.csv"
    assert not sub[TARGET].isna().any(), "NaN dans les prédictions"
    assert sub[TARGET].between(MIN_KG, MAX_KG).all(), f"prédictions hors de [{MIN_KG}, {MAX_KG}] kg"


def write_submission(name: str, pred: np.ndarray, test: pd.DataFrame, sample: pd.DataFrame):
    """Écrit outputs/submissions/{name}.csv dans l'ordre de sample_submission, après contrôle."""
    sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: np.clip(pred, MIN_KG, MAX_KG)})
    sub = sub.set_index(ID).loc[sample[ID]].reset_index()
    check_submission(sub, sample)
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path
