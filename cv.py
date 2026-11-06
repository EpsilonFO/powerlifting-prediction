"""Phase 0 : protocole de validation unique (plis fixes, OOF, journal des expériences)."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import (
    EXPERIMENTS_FILE, FOLDS_FILE, ID, N_FOLDS, NN_DATA_DIR, OOF_DIR, PARAM_DIR, PRED_DIR, SEED, TARGET,
)
from features import TargetTransform, build_features, outlier_mask

LOG_COLUMNS = [
    "name", "model", "target", "outliers", "seeds", "n_features", "cv_rmse", "cv_std",
    "fold_rmse", "public_score", "minutes", "date",
]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def make_folds(train: pd.DataFrame) -> np.ndarray:
    """5 plis stratifiés sur (décile de la cible × sexe), fixés une fois pour toutes dans data/folds.csv
    (à partager avec le binôme). Retourne l'indice de pli de chaque ligne de train."""
    if FOLDS_FILE.exists():
        saved = pd.read_csv(FOLDS_FILE).set_index(ID)["fold"]
        if set(saved.index) == set(train[ID]):
            return saved.loc[train[ID]].to_numpy()
    strata = pd.qcut(train[TARGET], 10, labels=False).astype(str) + "_" + train["Sex"].where(train["Sex"] == "F", "M")
    folds = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, strata)):
        folds[va] = k
    pd.DataFrame({ID: train[ID], "fold": folds}).to_csv(FOLDS_FILE, index=False)
    return folds


def make_subfolds(train: pd.DataFrame, folds: np.ndarray, n_sub: int = 2) -> np.ndarray:
    """Découpage d'entraînement plus fin, emboîté dans les plis de référence : chaque pli k est coupé en `n_sub`
    (même stratification), le sous-pli j de k devient le pli k * n_sub + j. Avec 10 plis, chaque modèle apprend sur
    90 % du train au lieu de 80 % et la prédiction test moyenne 10 modèles (journal, section 21)."""
    strata = pd.qcut(train[TARGET], 10, labels=False).astype(str) + "_" + train["Sex"].where(train["Sex"] == "F", "M")
    sub = np.zeros(len(train), dtype=int)
    for k in range(N_FOLDS):
        idx = np.where(folds == k)[0]
        skf = StratifiedKFold(n_sub, shuffle=True, random_state=SEED + k)
        for j, (_, part) in enumerate(skf.split(idx, strata.iloc[idx])):
            sub[idx[part]] = j
    return folds * n_sub + sub


def load_params(model: str) -> dict:
    f = PARAM_DIR / f"{model}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run_cv(
    name: str, model: str, train: pd.DataFrame, test: pd.DataFrame, F_tr: pd.DataFrame, F_te: pd.DataFrame,
    folds: np.ndarray, *, target: str = "A", outliers: str = "keep", seeds: int = 1, params: dict | None = None,
    only_folds: list[int] | None = None, save: bool = True, verbose: bool = True, split: np.ndarray | None = None,
) -> dict:
    """Entraîne `model` sur chaque pli, garde les prédictions OOF et test (moyenne des modèles de pli).

    - outliers="drop" retire du *train de chaque pli* les lignes aberrantes ; la validation reste complète.
    - only_folds : sous-ensemble de plis (tuning rapide) ; rien n'est sauvegardé dans ce cas.
    - split : découpage d'entraînement plus fin que `folds` (`make_subfolds`) ; les scores restent calculés sur
      les 5 plis de référence pour rester comparables aux autres expériences.
    """
    from models import MODELS  # import tardif : évite le cycle et les imports lourds (torch, catboost)

    fit = MODELS[model]["fn"]
    params = {**MODELS[model].get("params", {}), **(params or {})}
    tt = TargetTransform(target)
    y = train[TARGET].to_numpy(dtype=float)
    keep = ~outlier_mask(train) if outliers == "drop" else np.ones(len(train), bool)

    oof = np.full(len(train), np.nan)
    pred_te = np.zeros(len(test))
    split = folds if split is None else split
    ran = sorted(np.unique(split)) if only_folds is None else list(only_folds)
    fold_scores, t0 = [], time.time()
    for k in ran:
        va = np.where(split == k)[0]
        tr = np.where((split != k) & keep)[0]
        Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
        ytr, yva = tt.forward(y[tr], Xtr), tt.forward(y[va], Xva)
        wtr, wva = tt.weight(Xtr), tt.weight(Xva)
        p_va, p_te = fit(Xtr, ytr, wtr, Xva, yva, wva, F_te, {**params, "seeds": seeds})
        log_var = float(np.var(yva - p_va)) if target == "B" else 0.0
        oof[va] = tt.inverse(p_va, Xva, log_var)
        pred_te += tt.inverse(p_te, F_te, log_var) / len(ran)
        fold_scores.append(rmse(y[va], oof[va]))
        if verbose:
            print(f"  [{name}] pli {k} : RMSE {fold_scores[-1]:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if split is not folds and only_folds is None:
        fold_scores = [rmse(y[folds == k], oof[folds == k]) for k in range(N_FOLDS)]
    result = _result(name, model, target, outliers, seeds, F_tr.shape[1], fold_scores, (time.time() - t0) / 60)
    if verbose:
        print(f"  [{name}] CV RMSE = {result['cv_rmse']:.3f} ± {result['cv_std']:.3f}", flush=True)
    if save and only_folds is None:
        np.save(OOF_DIR / f"{name}.npy", oof)
        np.save(PRED_DIR / f"{name}.npy", pred_te)
        log_experiment(result)
    return result


def _result(name, model, target, outliers, seeds, n_features, fold_scores, minutes) -> dict:
    return {
        "name": name, "model": model, "target": target, "outliers": outliers, "seeds": seeds,
        "n_features": n_features, "cv_rmse": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores)),
        "fold_rmse": json.dumps([round(s, 3) for s in fold_scores]), "public_score": np.nan,
        "minutes": round(minutes, 1), "date": time.strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------- réseaux entraînés dans un autre environnement (DRAGON)

def export_nn_data(train: pd.DataFrame, F_tr: pd.DataFrame, F_te: pd.DataFrame, folds: np.ndarray, target: str = "C"):
    """Écrit pour chaque pli outputs/nn_data/fold{k}.npz : les matrices d'entrée des réseaux (`models.nn_arrays`,
    ajustées sur le pli d'entraînement), la cible transformée, les poids de la perte et `scale_*` tel que
    prédiction en kg = prédiction × scale. Lu par dragon/search.py, qui n'a pas accès à pandas ni au projet."""
    from models import nn_arrays

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
    result = _result(name, model, target, "keep", seeds, n_features, scores, minutes)
    log_experiment(result)
    print(f"  [{name}] CV RMSE = {result['cv_rmse']:.3f} ± {result['cv_std']:.3f}  plis {result['fold_rmse']}")
    return result


# ---------------------------------------------------------------- journal des expériences

def read_log() -> pd.DataFrame:
    if EXPERIMENTS_FILE.exists():
        return pd.read_csv(EXPERIMENTS_FILE)
    return pd.DataFrame(columns=LOG_COLUMNS)


def log_experiment(result: dict) -> None:
    log = read_log()
    old = log.loc[log["name"] == result["name"], "public_score"]
    if len(old) and pd.notna(old.iloc[0]):  # relancée : ses prédictions ont changé, l'ancien score public ne vaut plus
        print(f"  [{result['name']}] score public {old.iloc[0]} effacé (il portait sur la version précédente)")
    log = log[log["name"] != result["name"]]
    pd.concat([log, pd.DataFrame([result])], ignore_index=True)[LOG_COLUMNS].to_csv(EXPERIMENTS_FILE, index=False)


def record_public_score(name: str, score: float) -> None:
    log = read_log()
    assert name in set(log["name"]), f"expérience inconnue : {name}"
    log.loc[log["name"] == name, "public_score"] = score
    log.to_csv(EXPERIMENTS_FILE, index=False)


def prepare():
    """Charge les données, construit les features et les plis (raccourci pour les scripts)."""
    from data import load_data

    train, test, sample = load_data()
    F_tr, F_te = build_features(train, test)
    return train, test, sample, F_tr, F_te, make_folds(train)
