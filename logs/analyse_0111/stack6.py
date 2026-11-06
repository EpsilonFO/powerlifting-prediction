"""Soumissions de diagnostic (journal 25.6) : stacking Ridge sur les 6 modèles de la sélection gloutonne,
puis dedup. N'écrase rien de l'ensemble à 33 modèles (noms ensemble_stack6 / ensemble_stack6_dedup)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
import numpy as np
from config import PRED_DIR, OOF_DIR, TARGET
from cv import prepare, rmse
from ensemble import load_predictions, stacking, save_ensemble
from postprocess import dedup_experiment
from submit import write_submission

SIX = ["lgbm_C_k10", "mlp_C_plr_k10", "cat_C_meet_k10", "xgb_C_k10", "ridge_A_k10", "lgbm_B_k10"]
train, test, sample, F_tr, F_te, folds = prepare()
y = train[TARGET].to_numpy(float)
O, T = load_predictions(SIX)
honest, pred, coefs = stacking(SIX, y, folds, O, T, F_tr, F_te)
print(f"Stacking 6 : RMSE {rmse(y, honest):.3f} (validation imbriquée) ; coefficients = {coefs}")
save_ensemble("ensemble_stack6", honest, pred, y, folds, len(SIX))
print("Soumission :", write_submission("ensemble_stack6", pred, test, sample))
dedup_experiment("ensemble_stack6", train, test, folds)
print("Soumission :", write_submission("ensemble_stack6_dedup", np.load(PRED_DIR / "ensemble_stack6_dedup.npy"), test, sample))

# Écarts entre les trois soumissions candidates (test) et entre leurs OOF : ordre de grandeur de ce que
# le public peut distinguer.
names = ["ensemble_stack_dedup", "ensemble_stack6_dedup", "ensemble_weighted_dedup"]
P = {n: np.load(PRED_DIR / f"{n}.npy") for n in names}
Q = {n: np.load(OOF_DIR / f"{n}.npy") for n in names}
for i in range(3):
    for j in range(i + 1, 3):
        a, b = names[i], names[j]
        print(f"{a} vs {b} : écart RMS test {np.sqrt(np.mean((P[a]-P[b])**2)):.2f} kg, OOF {np.sqrt(np.mean((Q[a]-Q[b])**2)):.2f} kg, "
              f"corr test {np.corrcoef(P[a], P[b])[0,1]:.5f}")
