"""Recettes des 37 modèles de l'ensemble soumis (1er du public le 03/11, journal étape 26.1).

Chaque recette donne les arguments de `main.py train` qui ont produit `outputs/oof/<nom>.npy` et
`outputs/preds/<nom>.npy` ; `main.py reproduce` les relance (un processus par modèle). `None` : modèle entraîné
hors du pipeline (DRAGON, TabICLv2 sur GPU Kaggle), voir `external/`.

L'ordre compte : c'est celui des colonnes du stacking (la Ridge en dépend au dernier bit près).

Repères pour lire les options :
- `--features v1` : features du 28/09, avant l'ablation du 20/10 (journal étapes 18 et 24) ;
- `--no-tuned` : paramètres par défaut de `models.MODELS`, sans `outputs/params/<modèle>.json` ;
- `--params lgbm_v1` : LightGBM tuné le 29/09 (`lgbm.json` est le retuning du 31/10, étape 20) ;
- `--folds 10` : 10 sous-plis emboîtés dans les 5 plis de référence (étape 23).
"""

RECIPES: dict[str, str | None] = {
    # -- 10 plis, features v2 (31/10, étape 23)
    "rf_C_k10": "--model rf --target C --folds 10",
    "et_C_k10": "--model et --target C --folds 10",
    "hgb_raw_k10": "--model hgb --target A --folds 10",
    "lgbm_C_k10": "--model lgbm --target C --seeds 5 --folds 10",
    "lgbm_B_k10": "--model lgbm --target B --seeds 5 --folds 10",
    "lgbm_C_xt_k10": "--model lgbm --target C --seeds 3 --folds 10 --set extra_trees=true",
    "xgb_C_k10": "--model xgb --target C --seeds 5 --folds 10",
    "cat_C_k10": "--model cat --target C --folds 10",
    "mlp_C_plr_k10": "--model mlp --target C --seeds 5 --folds 10",
    "ridge_C_k10": "--model ridge --target C --folds 10",
    # -- 10 plis, anciens modèles réentraînés à l'identique (features v1, 01/11, étape 24)
    "ridge_A_k10": "--model ridge --target A --folds 10 --features v1",
    "rf_A_k10": "--model rf --target A --folds 10 --features v1 --no-tuned",
    "et_A_k10": "--model et --target A --folds 10 --features v1 --no-tuned",
    # -- 5 plis, features v1 (28-29/09, étapes 7 à 9)
    "lgbm_A": "--model lgbm --target A --features v1 --no-tuned",
    "lgbm_B": "--model lgbm --target B --features v1 --no-tuned",
    "lgbm_C": "--model lgbm --target C --features v1 --no-tuned",
    "lgbm_A_drop": "--model lgbm --target A --features v1 --no-tuned --outliers drop",
    "xgb_A": "--model xgb --target A --features v1 --no-tuned",
    "mlp_A": "--model mlp --target A --features v1 --no-tuned",
    "cat_A": "--model cat --target A --features v1 --no-tuned",
    "lgbm_A_tuned": "--model lgbm --target A --seeds 5 --features v1 --params lgbm_v1",
    "lgbm_B_tuned_k10": "--model lgbm --target B --seeds 5 --folds 10 --features v1 --params lgbm_v1",
    "lgbm_C_tuned": "--model lgbm --target C --seeds 5 --features v1 --params lgbm_v1",
    "mlp_A_s5": "--model mlp --target A --seeds 5 --features v1 --no-tuned",
    "mlp_C_plr_s5_k10": "--model mlp --target C --seeds 5 --folds 10 --features v1",
    "dragon_C": None,  # architecture cherchée par DRAGON (étape 14) : external/dragon/
    # -- 5 plis, features v2 (20-31/10, étapes 18 à 20)
    "lgbm_C_v2": "--model lgbm --target C --seeds 5 --params lgbm_v1",
    "cat_C_v2": "--model cat --target C --no-tuned",
    "ridge_A_v2": "--model ridge --target A",
    "lgbm_A_v2_lin": "--model lgbm --target A --seeds 3 --set linear_tree=true",
    "xgb_C_v2_default": "--model xgb --target C --seeds 5 --no-tuned",
    # -- 10 plis, information « compétition » donnée au MLP et à CatBoost (01/11, étape 25.5)
    "mlp_C_plr_te_k10": "--model mlp --target C --seeds 5 --folds 10 --set te=true",
    "cat_C_meet_k10": "--model cat --target C --folds 10 --set meet_cats=true",
    # -- 10 plis, méthodes 2024-2026 (03/11, étape 26.1)
    "lgbm_C_tabprep_k10": "--model lgbm --target C --seeds 3 --folds 10 --set tabprep=true n_jobs=4",
    "tabm_C_k10": "--model tabm --target C --folds 10",
    "realmlp_C_k10": "--model realmlp --target C --folds 10",
    "tabicl_C_k10": None,  # TabICLv2 sur GPU Kaggle (étape 26) : external/tabicl/
}

ENSEMBLE: list[str] = list(RECIPES)
