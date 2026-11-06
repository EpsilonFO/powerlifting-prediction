"""Tuning Optuna des GBM et des forêts sur les plis fixes (learning rate élevé ou forêt réduite pour aller
vite ; la version finale se réentraîne avec le learning rate / nombre d'arbres du modèle).

Les plis 3 et 4 ne servent jamais au tuning (par défaut `n_folds=3`) : comparer les scores sur ces deux plis
avant / après tuning corrige la malédiction du vainqueur (le meilleur essai est optimiste sur les plis 0-2).

MLP (02/11, cours 6) : la configuration courante est évaluée en premier (`BASELINES`) pour servir de référence dans
les mêmes conditions (1 seed, plis 0-2) ; l'espace décrit l'architecture (largeur, profondeur, rétrécissement), la
régularisation (dropout, weight decay), l'optimisation (lr, batch) et le plongement PLR. Les paramètres sauvegardés
sont ceux du modèle (`hidden`, `plr`...), pas ceux de l'espace de recherche (`width`, `n_layers`...)."""

import json
import time

import optuna

from powerlifting.config import PARAM_DIR
from powerlifting.cv import prepare, run_cv


def _lgbm_space(t: optuna.Trial) -> dict:
    return {
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 100, log=True),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "cat_smooth": t.suggest_float("cat_smooth", 1, 100, log=True),
        "cat_l2": t.suggest_float("cat_l2", 1, 50, log=True),
        "min_data_per_group": t.suggest_int("min_data_per_group", 10, 200, log=True),
    }


def _xgb_space(t: optuna.Trial) -> dict:
    return {
        "max_depth": t.suggest_int("max_depth", 4, 12),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True),
        "gamma": t.suggest_float("gamma", 1e-3, 5, log=True),
    }


def _cat_space(t: optuna.Trial) -> dict:
    return {
        "depth": t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 10, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0, 5),
    }


def _forest_space(t: optuna.Trial) -> dict:
    return {
        "max_features": t.suggest_float("max_features", 0.1, 1.0),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50, log=True),
    }


def _mlp_space(t: optuna.Trial) -> dict:
    width, n_layers = t.suggest_int("width", 128, 768, log=True), t.suggest_int("n_layers", 2, 4)
    taper = t.suggest_categorical("taper", [True, False])  # True : largeur divisée par 2 à chaque couche
    return {
        "hidden": [max(32, width // 2 ** i) if taper else width for i in range(n_layers)],
        "dropout": t.suggest_float("dropout", 0.0, 0.5),
        "lr": t.suggest_float("lr", 3e-4, 5e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": t.suggest_categorical("batch_size", [512, 1024, 2048]),
        "plr": {"sigma": t.suggest_float("plr_sigma", 0.01, 1.0, log=True),
                "k": t.suggest_categorical("plr_k", [8, 16, 32]), "d": t.suggest_categorical("plr_d", [8, 16, 32])},
    }


SPACES = {"lgbm": _lgbm_space, "xgb": _xgb_space, "cat": _cat_space, "rf": _forest_space, "et": _forest_space,
          "mlp": _mlp_space}
# réglages fixés pendant la recherche seulement, pour aller vite (le modèle final reprend ceux de models.MODELS) ;
# pour le MLP, `te` est aussi à passer au modèle final (--set te=true, journal 25.5)
TUNING_FIXED = {"lgbm": {"learning_rate": 0.08}, "xgb": {"learning_rate": 0.08}, "cat": {"learning_rate": 0.12},
                "rf": {"n_estimators": 150}, "et": {"n_estimators": 150}, "mlp": {"te": True}}
# configuration courante, évaluée en premier pour avoir la référence dans les conditions du tuning
BASELINES = {"mlp": {"width": 256, "n_layers": 3, "taper": True, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4,
                     "batch_size": 1024, "plr_sigma": 0.1, "plr_k": 16, "plr_d": 16}}


def tune(model: str, n_trials: int = 30, n_folds: int = 3, target: str = "A", outliers: str = "keep") -> dict:
    """Minimise la RMSE CV moyenne sur les `n_folds` premiers plis ; sauvegarde outputs/params/{model}.json
    (rechargé automatiquement par `main.py train`)."""
    train, test, _, F_tr, F_te, folds = prepare()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        t0, params = time.time(), {**SPACES[model](trial), **TUNING_FIXED[model]}
        r = run_cv(
            "tune", model, train, test, F_tr, F_te, folds, target=target, outliers=outliers, params=params,
            only_folds=list(range(n_folds)), verbose=False,
        )
        print(f"  essai {trial.number:3d} : RMSE {r['cv_rmse']:.3f}  plis {r['fold_rmse']}  "
              f"({(time.time() - t0) / 60:.1f} min)  {trial.params}", flush=True)
        return r["cv_rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    if model in BASELINES:
        study.enqueue_trial(BASELINES[model])
    study.optimize(objective, n_trials=n_trials)
    best = SPACES[model](optuna.trial.FixedTrial(study.best_params))  # paramètres du modèle, pas de l'espace
    print(f"Meilleur essai : RMSE {study.best_value:.3f}\n  {best}")
    (PARAM_DIR / f"{model}.json").write_text(json.dumps(best, indent=2))
    return best
