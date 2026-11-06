"""Bruit d'un essai Optuna (journal 25.8) : la configuration de référence du MLP sur les plis 0-2, 1 seed, pour
3 seeds différentes de celle du tuning (42). Donne l'écart-type sigma de l'estimation d'un essai."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
import numpy as np
import models
from config import PARAM_DIR
from cv import prepare, run_cv
train, test, _, F_tr, F_te, folds = prepare()
params = {**json.loads((PARAM_DIR / "mlp_v1.json").read_text()), "te": True}
scores = [77.864]  # seed 42 (essai 0 du tuning)
for seed in (43, 44, 45):
    models.SEED = seed
    r = run_cv("noise", "mlp", train, test, F_tr, F_te, folds, target="C", params=params, only_folds=[0, 1, 2], verbose=False)
    scores.append(r["cv_rmse"]); print(f"seed {seed} : {r['cv_rmse']:.3f}  plis {r['fold_rmse']}", flush=True)
s = np.std(scores, ddof=1)
print(f"4 seeds : {[round(x, 3) for x in scores]}  moyenne {np.mean(scores):.3f}  sigma {s:.3f}  "
      f"optimisme attendu sigma*sqrt(2 ln 40) = {s * np.sqrt(2 * np.log(40)):.3f}")
