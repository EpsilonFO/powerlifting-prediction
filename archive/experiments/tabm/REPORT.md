# TabM sur TotalKg : évaluation (03/11)

TabM (Gorishniy, Kotelnikov, Babenko, ICLR 2025, arXiv 2410.24210), paquet `tabm` 0.0.3, évalué sans toucher au
projet, sur les 5 plis de référence (`common.cv_loop`, cible C pondérée) avec `common.evaluate`.

## Verdict : à intégrer (meilleur réseau du projet)

- Seul, à 5 plis : **77,18** contre 78,07 pour le MLP de référence (1 seed, `te=True`), mieux sur **5 plis sur 5**
  (−0,79 à −1,13). Il bat le MLP à 5 seeds (77,67) et égale le MLP à 10 plis × 5 seeds avec te (77,18).
  LightGBM tuné à 5 plis reste un peu devant (77,04).
- Stacking à 33 modèles : **76,370 → 76,343 (−0,027)**, mieux sur 5 plis sur 5 (−0,016 à −0,035).
- Témoin : quatre modèles existants entraînés eux aussi à 5 plis mais hors des 33 (`mlp_C_plr_v2`,
  `lgbm_C_v2_tuned`, `xgb_C_v2`, `cat_C_v2_tuned`) rapportent entre −0,0004 et +0,0014 : le gain de TabM n'est pas
  un artefact du découpage, c'est de la diversité.
- **Avec TabICLv2 et RealMLP en plus : 76,294 (−0,076), mieux sur 5 plis sur 5** (`joint_eval.py tabicl_C realmlp tabm`).
- Borne basse (5 plis contre 10). À 10 plis : environ 2 h 30 sur MPS (moitié avec patience 8 : meilleure époque
  entre 25 et 36). Une seule exécution suffit (ensemble interne de 32 têtes).

## Configuration retenue (`tabm_model.tabm_fn`, `{"cat": "emb"}`)

- Entrées de `mlp_fn` avec `te=True` (`models._with_te` dans le pli puis `models.nn_arrays`) : 15 catégories,
  38 continues, 22 autres numériques.
- Continues : `PiecewiseLinearEmbeddings` (rtdl, version « B », 48 bins par quantiles du pli d'entraînement, d = 16).
- Catégories : `nn.Embedding` comme le MLP du projet (défaut TabM : one-hot ; effet dans le bruit).
- Backbone `arch_type="tabm"` (BatchEnsemble), k = 32, 2 blocs de 512, ReLU, dropout 0,1 ; sortie
  `LinearEnsemble(512 → 1)`.
- AdamW lr 2e-3, weight decay 3e-4, batch 512, `share_training_batches=True`. Perte : MSE pondérée (en kg) moyennée
  sur les 32 têtes ; prédiction = moyenne des têtes ; early stopping sur la RMSE de la moyenne, patience 16,
  ≤ 100 époques.
- ≈ 1 Go RSS + 1-2 Go MPS ; ≈ 14 min par pli sur MPS (MPS ≈ 1,7× plus rapide que 2 threads CPU).

## Pli 0 (bruit ≈ ±0,3)

| Variante | RMSE pli 0 |
|---|---|
| MLP de référence | 77,61 |
| TabM défaut (PLE + one-hot) | 76,84 |
| **TabM PLE + embeddings des catégories** | **76,67** |
| TabM PLR σ = 0,1 + one-hot | 76,82 |
| TabM lr 1e-3 | 76,87 |

## 5 plis

| Modèle (5 plis) | RMSE | Pli 0 | Pli 1 | Pli 2 | Pli 3 | Pli 4 |
|---|---|---|---|---|---|---|
| MLP de référence (1 seed, te) | 78,068 | 77,608 | 78,403 | 77,657 | 78,230 | 78,438 |
| `mlp_C_plr_v2` (5 seeds, sans te) | 77,666 | 77,225 | 77,777 | 77,358 | 77,821 | 78,148 |
| `lgbm_C_v2_tuned` (5 seeds) | 77,044 | 76,572 | 77,091 | 76,650 | 77,415 | 77,489 |
| **TabM** | **77,178** | 76,707 | 77,273 | 76,840 | 77,436 | 77,632 |

Corrélation des résidus : `lgbm_C_k10` 0,983, `mlp_C_plr_te_k10` 0,983, `cat_C_meet_k10` 0,973, `xgb_C_k10` 0,981.

## Intégration (non faite)

Copier `TabMNet` / `tabm_fn` dans `models.py` (ou `tabnn.py`), l'enregistrer dans `MODELS`, ajouter `tabm` (tire
`rtdl_num_embeddings`) aux dépendances. Pistes non testées : `share_training_batches=False`, 2ᵉ seed, d_block 256,
patience 8.

Fichiers : `tabm_model.py`, `run.py`, `evaluate.py`, `controls.py`, `queue{1,2,3}.sh`, `logs/`,
`../results/tabm_{oof,test}.npy`.
