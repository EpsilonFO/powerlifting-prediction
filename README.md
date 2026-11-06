# Régression : performances en powerlifting

Prédire le total soulevé (`TotalKg`) d'un athlète de powerlifting à partir de 16 variables. Compétition Kaggle
[`performances-en-powerlifting`](https://www.kaggle.com/competitions/performances-en-powerlifting), métrique RMSE.

**Meilleure soumission : RMSE 76,247 sur Kaggle (1ʳᵉ place du classement public).** Elle combine 37 modèles
(LightGBM, XGBoost, CatBoost, forêts, Ridge, réseaux de neurones…), puis corrige les lignes du test qui existent à
l'identique dans le train.

Le détail des étapes et des choix est dans [`JOURNAL.md`](JOURNAL.md).

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et un compte Kaggle (sur Mac, aussi `brew install libomp`).

1. Sur Kaggle, rejoindre la compétition (*Join Competition*) et créer un token sur
   https://www.kaggle.com/settings/api.
2. Installer le projet :

   ```bash
   uv sync
   ```

Les données sont téléchargées automatiquement au premier lancement ; le token Kaggle est demandé à ce moment-là.

## Refaire la meilleure soumission

Les prédictions des 37 modèles sont déjà dans le dépôt. Deux commandes suffisent (1 minute) :

```bash
uv run main.py ensemble                # combine les 37 modèles
uv run main.py dedup ensemble_stack    # corrige les doublons du train : c'est la soumission finale
```

La soumission est écrite dans `outputs/submissions/ensemble_stack_dedup.csv`.

### Tout réentraîner

Pour réentraîner les modèles au lieu d'utiliser leurs prédictions (une douzaine d'heures), lancer d'abord :

```bash
uv run main.py reproduce
```

puis les deux commandes ci-dessus. Les commandes de chaque modèle sont listées dans
[`src/powerlifting/recipes.py`](src/powerlifting/recipes.py). Deux modèles se lancent à part, avec leur propre
procédure : DRAGON ([`external/dragon/`](external/dragon/README.md)) et TabICL, qui demande un GPU
([`external/tabicl/`](external/tabicl/README.md)). S'ils ne sont pas relancés, leurs prédictions déjà présentes
sont utilisées.

## Organisation du code

| Dossier | Contenu |
|---|---|
| `main.py` | point d'entrée (`uv run main.py --help` pour toutes les commandes) |
| `src/powerlifting/` | le code : données, features, validation croisée, modèles, ensemble, soumission |
| `external/` | les deux modèles entraînés hors du pipeline (DRAGON, TabICL) |
| `outputs/` | prédictions des 37 modèles, hyperparamètres, journal des scores (`experiments.csv`) |
| `tests/` | tests automatiques (`uv run pytest`) |
| `archive/` | expériences et analyses exploratoires, hors du pipeline |

## Résultats

| Étape | RMSE en validation croisée |
|---|---|
| Prédire la moyenne | 171,4 |
| Meilleur modèle seul (LightGBM) | 76,73 |
| Combinaison des 37 modèles | 76,21 |
| **+ correction des doublons (soumission finale)** | **76,075** (Kaggle : 76,247) |

La validation croisée est faite sur 5 plis fixes (`data/folds.csv`).
