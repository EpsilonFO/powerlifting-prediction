"""Post-traitement des doublons exacts.

Une ligne identique sur les 16 variables d'origine à des lignes du train (très probablement le même athlète à la
même compétition) reçoit la moyenne de leur `TotalKg` à la place de la prédiction du modèle : entre doublons,
l'écart-type de la cible est d'environ 10 kg, contre ~77 d'erreur pour les modèles.

En CV, les correspondances d'une ligne de validation sont cherchées dans le train de son pli uniquement : la
RMSE OOF obtenue est donc honnête et directement comparable aux autres expériences.
"""

import numpy as np
import pandas as pd

from powerlifting.config import N_FOLDS, OOF_DIR, PRED_DIR, TARGET
from powerlifting.cv import rmse
from powerlifting.experiment_log import log_experiment, make_result, read_log
from powerlifting.features import RAW_FEATURES


def row_keys(df: pd.DataFrame) -> np.ndarray:
    """Empreinte (hash 64 bits) des 16 variables d'origine : deux lignes identiques ont la même empreinte."""
    return pd.util.hash_pandas_object(df[RAW_FEATURES], index=False).to_numpy()


def replace_duplicates(pred, keys, ref_keys, ref_y) -> tuple[np.ndarray, np.ndarray]:
    """Remplace pred[i] par la moyenne de ref_y sur les lignes de référence de même empreinte.
    Retourne les prédictions corrigées et le masque des lignes remplacées."""
    means = pd.Series(np.asarray(ref_y, dtype=float)).groupby(ref_keys).mean()
    matched = means.reindex(keys).to_numpy()
    hit = ~np.isnan(matched)
    out = np.array(pred, dtype=float)
    out[hit] = matched[hit]
    return out, hit


def dedup_experiment(name: str, train: pd.DataFrame, test: pd.DataFrame, folds: np.ndarray) -> dict:
    """Applique le post-traitement à l'expérience `name` (OOF dans le pli, test avec tout le train) et la
    journalise sous `{name}_dedup`."""
    y = train[TARGET].to_numpy(dtype=float)
    k_tr, k_te = row_keys(train), row_keys(test)
    oof = np.load(OOF_DIR / f"{name}.npy")
    new_oof, n_oof = oof.copy(), 0
    for k in range(N_FOLDS):
        va = folds == k
        new_oof[va], hit = replace_duplicates(oof[va], k_tr[va], k_tr[~va], y[~va])
        n_oof += hit.sum()
    pred, hit_te = replace_duplicates(np.load(PRED_DIR / f"{name}.npy"), k_te, k_tr, y)

    new = f"{name}_dedup"
    np.save(OOF_DIR / f"{new}.npy", new_oof)
    np.save(PRED_DIR / f"{new}.npy", pred)
    src = read_log().set_index("name").loc[name]
    scores = [rmse(y[folds == k], new_oof[folds == k]) for k in range(N_FOLDS)]
    result = make_result(new, src["model"], src["target"], src["outliers"], src["seeds"], src["n_features"], scores, 0.0)
    log_experiment(result)
    print(f"  [{new}] lignes remplacées : {n_oof} en CV, {hit_te.sum()} / {len(test)} en test")
    print(f"  [{new}] CV RMSE = {result['cv_rmse']:.3f} (avant : {src['cv_rmse']:.3f})  plis {result['fold_rmse']}")
    return result
