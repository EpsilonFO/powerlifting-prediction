"""Journal partagé des expériences `outputs/experiments.csv` : une ligne par expérience (CV, écart-type, scores par
pli, score public Kaggle). Relancer une expérience remplace sa ligne et efface son score public."""

import json
import time

import numpy as np
import pandas as pd

from powerlifting.config import EXPERIMENTS_FILE

LOG_COLUMNS = [
    "name", "model", "target", "outliers", "seeds", "n_features", "cv_rmse", "cv_std",
    "fold_rmse", "public_score", "minutes", "date",
]


def make_result(name, model, target, outliers, seeds, n_features, fold_scores, minutes) -> dict:
    """Ligne du journal à partir des RMSE par pli."""
    return {
        "name": name, "model": model, "target": target, "outliers": outliers, "seeds": seeds,
        "n_features": n_features, "cv_rmse": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores)),
        "fold_rmse": json.dumps([round(s, 3) for s in fold_scores]), "public_score": np.nan,
        "minutes": round(minutes, 1), "date": time.strftime("%Y-%m-%d %H:%M"),
    }


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
    row = pd.DataFrame([{c: result[c] for c in LOG_COLUMNS}])
    pd.concat([log, row], ignore_index=True)[LOG_COLUMNS].to_csv(EXPERIMENTS_FILE, index=False)


def record_public_score(name: str, score: float) -> None:
    log = read_log()
    assert name in set(log["name"]), f"expérience inconnue : {name}"
    log.loc[log["name"] == name, "public_score"] = score
    log.to_csv(EXPERIMENTS_FILE, index=False)
