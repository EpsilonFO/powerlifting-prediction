# Archive de l'exploration

Ce dossier garde la trace des expériences qui ont mené à la soumission finale, mais **ne fait pas partie du
pipeline**. Le pipeline (`main.py`, `src/powerlifting/`) se suffit à lui-même ; les recettes des 37 modèles de
l'ensemble sont dans `src/powerlifting/recipes.py`.

Les scripts ci-dessous ont tourné sur l'ancienne arborescence (modules à plat à la racine). Pour les relancer tels
quels, repartir du commit `4dd1f8f` (« État final avant nettoyage ») : `git worktree add ../avant-nettoyage 4dd1f8f`.

| Dossier | Contenu | Journal |
|---|---|---|
| `experiments/` | évaluation hors projet des méthodes 2024-2026 : TabICLv2, TabM, RealMLP, HL-Gauss, TabPrep, méta-modèles (projet uv séparé, `common.py`). Conclusions dans `*/REPORT.md` et `*/results.jsonl` ; les prédictions `.npy`, le checkpoint TabICL et les caches ne sont pas versionnés | étape 26 |
| `analyses_0111/` | analyses ponctuelles : structure des résidus, appariement d'athlètes, courbes d'apprentissage, ablations du TE, tuning du MLP (`tune_mlp.log`), stacking à 6 modèles (`stack6.py`), ablation de l'ensemble (`ablation_ensemble.py`) | étapes 25.1 à 25.8 |
| `queues/` | files de calcul `queue_cours5.sh`, `queue_k10.sh`, `queue_k10_v1.sh`, remplacées par `recipes.py` et `main.py reproduce` | étapes 20, 23, 24 |
| `logs/` | journaux de ces files, des ensembles successifs (`ensemble_k10*.log`) et de l'intégration du 03/11 ; listes des ensembles à 31, 33 et 37 modèles (`ensemble_k10*_models.txt`) ; recherche DRAGON du 19/10 | étapes 14, 20, 23 à 26 |
| `params/` | `mlp_tuned.json` : réglage du MLP trouvé par Optuna, non retenu (gain nul sur les plis jamais vus) | étape 25.8 |
| `outputs/archive_*` | OOF et prédictions test des ensembles soumis précédemment (21/10, 31/10, 01/11 à 31 et 33 modèles) | étapes 18 à 26 |
| `outputs/oof`, `outputs/preds` | modèles abandonnés ou remplacés (versions 5 plis, variantes tunées sans gain, ensembles de diagnostic `ensemble_stack6*`). Leurs scores restent dans `outputs/experiments.csv` | étapes 19 à 25 |
