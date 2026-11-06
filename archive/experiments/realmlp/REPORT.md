# RealMLP (Holzmüller et al., NeurIPS 2024) sur le projet TotalKg

*03/11. Environnement `experiments/`, pytabkit 1.8.0, `RealMLP_TD_Regressor` (réglages par défaut « TD », aucune
recherche d'hyperparamètres). Aucun fichier du projet modifié.*

## Verdict : à intégrer

- Meilleur que notre MLP à 1 seed sur 5 plis sur 5 : **77,905** contre 78,058.
- Ajouté aux 33 modèles : stacking 76,3699 → **76,3534 (−0,0165)**, mieux sur 5 plis sur 5. Plus que MLP + TE et
  CatBoost + clé réunis (−0,013, étape 25.5).
- Résidus les moins corrélés aux 4 modèles clés (0,967 à 0,978).
- Borne basse : RealMLP est à 5 plis, les 33 modèles à 10.
- **Avec TabICLv2 (5 plis) en plus : 76,3088 (−0,061), mieux sur 5 plis sur 5** (`joint_eval.py realmlp tabicl_C`).

## Pli 0 (1 seed, MPS)

| Variante | RMSE |
|---|---|
| A (kg) + target encoding | 77,854 |
| **C (points DOTS, sans poids) + target encoding** | **77,326** |
| A sans target encoding | 78,025 |
| *notre MLP* | *77,607* |
| *LightGBM* | *76,656* |

La cible C gagne 0,53 sur A alors que pytabkit n'accepte pas de poids d'échantillon (C sans poids optimise la RMSE
en points DOTS, pas en kg). Le target encoding apporte 0,17.

## 5 plis

| Pli | 0 | 1 | 2 | 3 | 4 | CV |
|---|---|---|---|---|---|---|
| **RealMLP C + TE** | 77,347 | 78,130 | 77,584 | 78,152 | 78,309 | **77,905** |
| MLP actuel | 77,607 | 78,320 | 77,664 | 78,247 | 78,452 | 78,058 |
| LightGBM | 76,656 | 77,305 | 76,804 | 77,583 | 77,659 | 77,201 |
| Δ stacking (33 → 34) | −0,028 | −0,008 | −0,013 | −0,005 | −0,029 | **−0,0165** |

Corrélation des résidus : 0,973 (`lgbm_C_k10`), 0,978 (`mlp_C_plr_te_k10`), 0,967 (`cat_C_meet_k10`), 0,970
(`xgb_C_k10`). La règle « −0,2 seul ≈ −0,01 stacking » prédisait −0,008 : le gain vient de la diversité.

## Notes techniques

- `fit(X, y, X_val, y_val)` : 256 époques (schéma `coslog4`), garde la meilleure époque sur le pli de validation.
- Catégories détectées par le dtype (one-hot ≤ 9 modalités, embedding de taille 8 au-delà). NaN numériques
  interdits : médiane du pli + indicatrices `na_*`.
- ≈ 5 min par pli sur MPS libre (15 min partagé), ≈ 3 Go de mémoire réelle (`footprint` ; `ps -o rss` ne voit pas
  MPS). MPS n'est pas parfaitement déterministe (±0,02 sur un pli).
- `n_ens = 4` (4 réseaux vectorisés) : lancé puis arrêté (2 h 40 estimées, gain attendu faible).

## Intégration (non faite)

1. `pytabkit` dans les dépendances du projet ; `realmlp_fn` dans `models.py` (copie de `realmlp/run.py`, `wtr`
   ignoré).
2. `main.py train --model realmlp --target C --folds 10 --set te=true` (un processus PyTorch seul).
3. Stacking à 34 modèles.

Fichiers : `realmlp/run.py`, `realmlp/evaluate.py`, `realmlp/logs/`, `realmlp/out/`, `results/realmlp_{oof,test}.npy`.
