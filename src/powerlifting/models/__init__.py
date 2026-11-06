"""Les modèles et leur registre. Tous partagent la même signature :

    fn(Xtr, ytr, wtr, Xva, yva, wva, Xte, params) -> (prédiction validation, prédiction test)

Xtr / Xva / Xte sont les features de `features.build_features` (catégories en dtype `category`), ytr / yva la cible
déjà transformée (A, B ou C), wtr / wva des poids (ou None). `Xva` sert à l'early stopping des GBM et des réseaux.
Les paramètres de `MODELS` sont les défauts ; `main.py train` y ajoute ceux de `outputs/params/<modèle>.json`.
"""

from powerlifting.models.baselines import const_fn, hgb_fn, ridge_fn
from powerlifting.models.boosting import cat_fn, lgbm_fn, xgb_fn
from powerlifting.models.forests import et_fn, rf_fn
from powerlifting.models.neural import mlp_fn, realmlp_fn, tabm_fn

MODELS = {
    "const": {"fn": const_fn},
    "hgb": {"fn": hgb_fn},
    "ridge": {"fn": ridge_fn, "params": {"alpha": 10.0}},
    "rf": {"fn": rf_fn, "params": {"n_estimators": 300, "min_samples_leaf": 5, "max_features": 0.4, "max_samples": 0.8}},
    "et": {"fn": et_fn, "params": {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.6}},
    "lgbm": {"fn": lgbm_fn, "params": {
        "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 30, "colsample_bytree": 0.7,
        "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 5.0, "cat_smooth": 20, "cat_l2": 10,
        "min_data_per_group": 50, "max_cat_threshold": 32,
    }},
    "cat": {"fn": cat_fn, "params": {"learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 5.0}},
    "xgb": {"fn": xgb_fn, "params": {
        "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 5, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_lambda": 5.0, "max_cat_to_onehot": 4,
    }},
    "mlp": {"fn": mlp_fn},
    "tabm": {"fn": tabm_fn, "params": {"cat": "emb", "te": True, "device": "auto"}},
    "realmlp": {"fn": realmlp_fn, "params": {"te": True, "device": "auto"}},
}
