"""Modèles entraînés hors du pipeline : DRAGON (environnement séparé, numpy < 2, `external/dragon/`) et TabICLv2
(GPU Kaggle, `external/tabicl/`). Export des entrées des réseaux et journalisation des prédictions rapatriées."""

import numpy as np
import pandas as pd

from powerlifting.config import N_FOLDS, NN_DATA_DIR, OOF_DIR, TARGET
from powerlifting.cv import rmse
from powerlifting.experiment_log import log_experiment, make_result
from powerlifting.features import TargetTransform


def export_nn_data(train: pd.DataFrame, F_tr: pd.DataFrame, F_te: pd.DataFrame, folds: np.ndarray, target: str = "C"):
    """Écrit pour chaque pli outputs/nn_data/fold{k}.npz : les matrices d'entrée des réseaux (`nn_arrays`, ajustées
    sur le pli d'entraînement), la cible transformée, les poids de la perte et `scale_*` tel que prédiction en kg =
    prédiction × scale. Lu par external/dragon/search.py, qui n'a accès ni à pandas ni au projet."""
    from powerlifting.models.neural import nn_arrays

    assert target in ("A", "C"), "cible B non prise en charge : sa retransformation n'est pas linéaire"
    tt = TargetTransform(target)
    y = train[TARGET].to_numpy(dtype=float)
    NN_DATA_DIR.mkdir(exist_ok=True)

    def scale(X):
        return X["dots_den"].to_numpy() / 500 if target == "C" else np.ones(len(X))

    for k in range(N_FOLDS):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
        Atr, Ava, Ate, cards, cont_idx = nn_arrays(Xtr, Xva, F_te)
        np.savez(
            NN_DATA_DIR / f"fold{k}.npz", Xtr=Atr, Xva=Ava, Xte=Ate,
            ytr=tt.forward(y[tr], Xtr), yva=tt.forward(y[va], Xva), wtr=scale(Xtr) ** 2, wva=scale(Xva) ** 2,
            scale_va=scale(Xva), scale_te=scale(F_te), va_idx=va, cards=np.array(cards), cont_idx=np.array(cont_idx),
        )
        print(f"  pli {k} -> {NN_DATA_DIR / f'fold{k}.npz'}  ({Atr.shape[1]} colonnes)", flush=True)


def register_external(name: str, model: str, train: pd.DataFrame, folds: np.ndarray, *, target: str, seeds: int,
                      minutes: float, n_features: int) -> dict:
    """Journalise une expérience dont les prédictions OOF et test (en kg) ont été écrites par un autre processus
    dans outputs/oof/{name}.npy et outputs/preds/{name}.npy."""
    y = train[TARGET].to_numpy(dtype=float)
    oof = np.load(OOF_DIR / f"{name}.npy")
    assert oof.shape == y.shape and not np.isnan(oof).any(), "OOF incomplète"
    scores = [rmse(y[folds == k], oof[folds == k]) for k in range(N_FOLDS)]
    result = make_result(name, model, target, "keep", seeds, n_features, scores, minutes)
    log_experiment(result)
    print(f"  [{name}] CV RMSE = {result['cv_rmse']:.3f} ± {result['cv_std']:.3f}  plis {result['fold_rmse']}")
    return result
