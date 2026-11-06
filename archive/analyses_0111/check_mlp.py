"""Contrôle du tuning MLP (journal 25.8) : configuration d'avant (mlp_v1) et tunée (mlp) sur les plis 3 et 4,
jamais vus par Optuna, 1 seed, cible C, avec target encoding."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du projet
from config import PARAM_DIR
from cv import prepare, run_cv
train, test, _, F_tr, F_te, folds = prepare()
for label, f in [("avant (mlp_v1)", "mlp_v1"), ("tuné (mlp)", "mlp")]:
    params = {**json.loads((PARAM_DIR / f"{f}.json").read_text()), "te": True}
    r = run_cv("check", "mlp", train, test, F_tr, F_te, folds, target="C", params=params, only_folds=[3, 4], verbose=False)
    print(f"{label:16s} plis 3-4 : {r['fold_rmse']}  moyenne {r['cv_rmse']:.3f}  ({r['minutes']:.1f} min)", flush=True)
