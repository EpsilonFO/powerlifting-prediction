"""Phase 5 : moyenne pondérée (poids optimisés sur les OOF) puis stacking Ridge.

Les estimations de RMSE de l'ensemble sont *honnêtes* : les poids (ou le méta-modèle) sont appris sur
4 plis d'OOF et évalués sur le 5e (validation imbriquée), pour ne pas biaiser la mesure.

Le méta-modèle Ridge garde des coefficients libres (négatifs autorisés) : à poids positifs (NNLS) ou par
sélection gloutonne, on retombe au niveau de la moyenne pondérée, moins bon de ~0,09 sur chacun des 5 plis.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from config import N_FOLDS, OOF_DIR, PRED_DIR, TARGET
from cv import _result, log_experiment, rmse
from submit import write_submission


def load_predictions(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof = np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in names])
    test = np.column_stack([np.load(PRED_DIR / f"{n}.npy") for n in names])
    return oof, test


def fit_weights(O: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poids >= 0 de somme 1 minimisant la RMSE."""
    m = O.shape[1]
    res = minimize(
        lambda w: np.mean((O @ w - y) ** 2), np.full(m, 1 / m), method="SLSQP",
        bounds=[(0, 1)] * m, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    return res.x


def _meta_features(O: np.ndarray, F: pd.DataFrame) -> np.ndarray:
    """OOF des modèles + Sex et Equipment (one-hot) pour que le méta-modèle corrige par segment."""
    seg = pd.get_dummies(F[["Sex", "Equipment"]]).to_numpy(float)  # catégories communes train/test -> mêmes colonnes
    return np.hstack([O, seg])


def weighted_average(names, y, folds, test_pred_matrix, oof_matrix):
    honest = np.zeros(len(y))
    for k in range(N_FOLDS):
        va = folds == k
        honest[va] = oof_matrix[va] @ fit_weights(oof_matrix[~va], y[~va])
    w = fit_weights(oof_matrix, y)
    return w, honest, test_pred_matrix @ w


def _honest_stack(Z, y, folds, alpha=10.0) -> np.ndarray:
    honest = np.zeros(len(y))
    for k in range(N_FOLDS):
        va = folds == k
        honest[va] = Ridge(alpha=alpha).fit(Z[~va], y[~va]).predict(Z[va])
    return honest


def stacking(names, y, folds, oof_matrix, test_pred_matrix, F_tr, F_te, alpha=10.0):
    Z, Zte = _meta_features(oof_matrix, F_tr), _meta_features(test_pred_matrix, F_te)
    honest = _honest_stack(Z, y, folds, alpha)
    meta = Ridge(alpha=alpha).fit(Z, y)
    return honest, meta.predict(Zte), {n: round(float(c), 3) for n, c in zip(names, meta.coef_[: len(names)])}


def ambiguity(O: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Décomposition d'ambiguïté (cours 5) d'une moyenne pondérée convexe, en MSE :
    erreur de l'ensemble = erreur moyenne des membres − ambiguïté (dispersion des membres autour de l'ensemble)."""
    fbar = O @ w
    members = float(np.sum(w * np.mean((O - y[:, None]) ** 2, 0)))
    return members, float(np.sum(w * np.mean((O - fbar[:, None]) ** 2, 0)))


def marginal_contributions(names, y, folds, oof_matrix, F_tr) -> pd.DataFrame:
    """Contribution marginale (cours 5) : hausse de la RMSE du stacking (validation imbriquée) quand on retire
    chaque modèle. Un modèle faible seul peut compter beaucoup s'il se trompe autrement que les autres."""
    base = rmse(y, _honest_stack(_meta_features(oof_matrix, F_tr), y, folds))
    rows = []
    for j, n in enumerate(names):
        Z = _meta_features(np.delete(oof_matrix, j, axis=1), F_tr)
        rows.append({"model": n, "rmse_seul": rmse(y, oof_matrix[:, j]),
                     "retrait": rmse(y, _honest_stack(Z, y, folds)) - base})
    return pd.DataFrame(rows).sort_values("retrait", ascending=False)


def save_ensemble(name: str, honest: np.ndarray, pred: np.ndarray, y, folds, n_models: int) -> None:
    """OOF (validation imbriquée) et prédictions test sauvegardées et journalisées comme une expérience : l'ensemble
    peut ensuite être post-traité (`main.py dedup`) et apparaît dans `main.py report`."""
    np.save(OOF_DIR / f"{name}.npy", honest)
    np.save(PRED_DIR / f"{name}.npy", pred)
    scores = [rmse(y[folds == k], honest[folds == k]) for k in range(N_FOLDS)]
    log_experiment(_result(name, "ensemble", "-", "keep", 1, n_models, scores, 0.0))


def run_ensemble(names: list[str], train, test, sample, F_tr, F_te, folds, contrib: bool = False) -> dict:
    y = train[TARGET].to_numpy(float)
    O, T = load_predictions(names)
    print("RMSE OOF individuelle :", {n: round(rmse(y, O[:, i]), 3) for i, n in enumerate(names)})
    print(f"Moyenne simple : {rmse(y, O.mean(1)):.3f}")

    w, honest_w, pred_w = weighted_average(names, y, folds, T, O)
    print(f"Moyenne pondérée : RMSE {rmse(y, honest_w):.3f} (validation imbriquée) ; poids = "
          f"{ {n: round(float(x), 3) for n, x in zip(names, w) if x > 0.001} }")
    members, amb = ambiguity(O, y, w)
    print(f"  ambiguïté : RMSE moyenne des membres {np.sqrt(members):.3f} -> ensemble {np.sqrt(members - amb):.3f}")
    save_ensemble("ensemble_weighted", honest_w, pred_w, y, folds, len(names))
    p1 = write_submission("ensemble_weighted", pred_w, test, sample)

    honest_s, pred_s, coefs = stacking(names, y, folds, O, T, F_tr, F_te)
    print(f"Stacking Ridge : RMSE {rmse(y, honest_s):.3f} (validation imbriquée) ; coefficients = {coefs}")
    save_ensemble("ensemble_stack", honest_s, pred_s, y, folds, len(names))
    p2 = write_submission("ensemble_stack", pred_s, test, sample)
    print("Fichiers :", p1, p2, sep="\n  ")
    if contrib:
        print("Contribution marginale (hausse de la RMSE du stacking quand on retire le modèle) :")
        print(marginal_contributions(names, y, folds, O, F_tr).to_string(index=False, float_format="%.3f"))
    return {"weighted": rmse(y, honest_w), "stack": rmse(y, honest_s)}
