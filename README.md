# Régression : performances en powerlifting

Prédire `TotalKg` (squat + développé couché + soulevé de terre) à partir de 16 variables sur l'athlète et la
compétition. Métrique : **RMSE**. Compétition Kaggle
[`performances-en-powerlifting`](https://www.kaggle.com/competitions/performances-en-powerlifting) (data challenge du
M2 Maths & IA).

**Soumission finale : stacking Ridge de 37 modèles + remplacement des doublons exacts du train. CV 76,075, public
76,247 (1er du classement public le 03/11).**

Toutes les étapes, les choix et les chiffres sont dans [`JOURNAL.md`](JOURNAL.md).

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et un compte Kaggle. Sur macOS, LightGBM et XGBoost ont besoin de
`brew install libomp`.

1. **Rejoindre la compétition** sur Kaggle (*Join Competition*), sinon le téléchargement échoue (erreur 403).
2. **Créer un token API** sur https://www.kaggle.com/settings/api (*Generate New Token*).
3. `uv sync` installe les dépendances et le paquet `powerlifting` (mode éditable).

Les données sont téléchargées par **kagglehub** dans `data/` au premier lancement. Le token est alors demandé une
fois, puis sauvegardé dans `~/.kaggle/` (jamais dans le dépôt). Sans invite : `export KAGGLE_API_TOKEN="KGAT_..."`.

## Reproduire la soumission

**À partir des prédictions versionnées (1 min).** Les prédictions OOF et test des 37 modèles sont dans
`outputs/oof/` et `outputs/preds/` :

```bash
uv run main.py ensemble                 # stacking Ridge des 37 modèles : CV 76,214
uv run main.py dedup ensemble_stack     # doublons exacts : CV 76,075 -> outputs/submissions/ensemble_stack_dedup.csv
```

Le fichier obtenu est identique au bit près à la soumission notée 76,247 (md5 `409605b8deb7a2b83e5cd28a48653026`).

**En réentraînant les modèles (une douzaine d'heures sur un Mac de 16 Go).** Chaque modèle de l'ensemble a sa recette
dans [`src/powerlifting/recipes.py`](src/powerlifting/recipes.py) :

```bash
uv run main.py reproduce --dry-run      # affiche les 35 commandes
uv run main.py reproduce                # les lance, un processus par modèle
uv run main.py reproduce lgbm_C_k10     # un seul modèle
uv run main.py reproduce --check        # recalcule sans rien écrire et compare aux OOF versionnées
```

Deux modèles sont entraînés hors du pipeline : `dragon_C` (recherche d'architecture DRAGON, environnement séparé,
voir [`external/dragon/`](external/dragon/README.md)) et `tabicl_C_k10` (TabICLv2 sur GPU Kaggle, voir
[`external/tabicl/`](external/tabicl/README.md)).

Contrôle du 06/11 avec `--check` (journal, étape 27) : les recettes vérifiées redonnent exactement les OOF
versionnées, dont celle du meilleur modèle seul (LightGBM + TabPrep). Exception : TabM et RealMLP, entraînés sur le
GPU du Mac (MPS), donnent des prédictions légèrement différentes d'une exécution à l'autre.

## Utilisation

```bash
uv run main.py train --model lgbm --target C --seeds 5   # CV d'un modèle : OOF, prédiction test, soumission
uv run main.py train --model lgbm --folds 10             # 10 sous-plis emboîtés dans les 5 plis de référence
uv run main.py train --model mlp --set te=true           # hyperparamètre ajouté à outputs/params/<modèle>.json
uv run main.py tune --model xgb --target C --trials 30   # Optuna sur les plis 0-2 -> outputs/params/xgb.json
uv run main.py ensemble [--models a b c] [--contrib]     # moyenne pondérée + stacking ; contribution marginale
uv run main.py dedup <expérience>                        # post-traitement des doublons exacts
uv run main.py analyze --oof lgbm_A                      # importances, résidus par segment, dépendances partielles
uv run main.py report                                    # tableau des expériences (CV, écart-type, score public)
uv run main.py score <expérience> 76.247                 # noter le score public d'une expérience
uv run pytest                                            # tests du pipeline
```

Modèles (`--model`) : `const`, `hgb`, `ridge`, `rf`, `et`, `lgbm`, `xgb`, `cat`, `mlp`, `tabm`, `realmlp`. Cibles
(`--target`) : `A` (`TotalKg`), `B` (log), `C` (points DOTS).

Lancer **un modèle par commande** : sur macOS, PyTorch et LightGBM/CatBoost/XGBoost ne cohabitent pas dans un même
processus (conflit OpenMP qui bloque le processus). Au plus deux entraînements lourds en même temps sur 16 Go.

## Organisation

```
main.py                     interface en ligne de commande
src/powerlifting/
  config.py  data.py        chemins, constantes ; téléchargement kagglehub
  features.py               nettoyage, features de domaine (Wilks / DOTS / IPF GL, âge, compétition…),
                            cibles A / B / C, target encoding du résidu appris dans le pli
  tabprep.py                variables relatives TabPrep (GroupBy, OAFE)
  cv.py                     plis fixes, sous-plis emboîtés, boucle de CV avec OOF
  experiment_log.py         journal outputs/experiments.csv
  models/                   registre MODELS ; baselines (const, hgb, ridge), forests, boosting, neural
  nets/                     réseaux PyTorch : MLP + PLR (partagé avec DRAGON), TabM
  tune.py                   Optuna
  ensemble.py               moyenne pondérée, stacking Ridge, contribution marginale (validation imbriquée)
  postprocess.py            doublons exacts du train
  submit.py                 écriture et contrôle des soumissions
  analysis.py               importances, résidus, dépendances partielles
  external.py               export pour DRAGON, journalisation des modèles entraînés ailleurs
  recipes.py                recettes des 37 modèles de l'ensemble
external/dragon/, tabicl/   modèles entraînés hors du pipeline (environnement séparé, GPU Kaggle)
tests/                      plis, absence de fuite, soumissions, recettes
data/folds.csv              plis de référence (versionnés)
outputs/                    experiments.csv, params/, oof/ et preds/ des 37 modèles, figures/
archive/                    exploration hors pipeline (évaluations, analyses, anciens ensembles), voir son README
```

## Protocole de validation

- **5 plis** stratifiés sur (décile de `TotalKg` × sexe), fixés une fois dans `data/folds.csv`. Les modèles à
  10 plis coupent chaque pli en deux sous-plis ; les scores restent calculés sur les 5 plis de référence.
- Tout ce qui dépend de la cible (target encoding du résidu `y − moyenne(Sex × Equipment)`) est appris **dans le
  pli**, avec une K-fold interne pour les lignes d'entraînement.
- Chaque expérience garde ses prédictions **OOF** et test (moyenne des modèles de pli) ; le journal
  `outputs/experiments.csv` garde la CV, son écart-type, les scores par pli et le score public.
- L'ensemble est évalué en **validation imbriquée** : méta-modèle appris sur 4 plis d'OOF, évalué sur le 5ᵉ.
- La soumission est choisie sur la CV, pas sur le classement public.

Points à connaître :

- early stopping sur le pli de validation : l'OOF est très légèrement optimiste ;
- la CV d'un modèle à 10 plis est optimiste d'environ 0,15 par rapport à 5 plis (journal, étape 23) : ne comparer
  que des modèles au même nombre de plis ;
- cible C (points DOTS) : perte pondérée par `(dénominateur/500)²`, équivalente à l'erreur quadratique en kg ;
- cible B (log) : correction du biais de retransformation `exp(p + σ²/2)` ;
- doublons exacts : en CV, les correspondances sont cherchées dans le train du pli seulement.

## Résultats (RMSE en kg, CV sur les 5 plis de référence)

| Étape | Méthode | CV | Public |
|---|---|---|---|
| repère | constante / HistGradientBoosting brut | 171,4 / 79,0 | |
| 7 | Ridge + splines, forêts, XGBoost, CatBoost, LightGBM, MLP | 82,3 à 78,1 | |
| 9 | LightGBM tuné, cible DOTS, 5 seeds | 77,49 | |
| 13 | MLP avec plongement PLR | 77,80 | |
| 16 | stacking + doublons exacts | 76,89 | 77,04 |
| 18 | features « compétition » (target encoding de la compétition, tailles) | 76,47 | |
| 23-25 | 10 plis, 33 modèles | 76,21 | 76,43 |
| 26 | + LightGBM TabPrep, TabM, RealMLP, TabICLv2 : 37 modèles | **76,075** | **76,247** |

Meilleurs modèles seuls (10 plis) : LightGBM + TabPrep 76,73, LightGBM 76,86, TabM 76,99, MLP + target encoding
77,18. Le bruit propre à chaque athlète (son niveau individuel, absent des données) limite tous les modèles : les
résidus de l'ensemble n'ont plus de structure exploitable (journal, étapes 22 et 25).
