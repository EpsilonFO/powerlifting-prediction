# Régression – Performances en powerlifting

Prédire `TotalKg` (squat + développé couché + soulevé de terre) à partir de 16 variables sur l'athlète et la
compétition. Métrique : **RMSE**. Compétition Kaggle
[`performances-en-powerlifting`](https://www.kaggle.com/competitions/performances-en-powerlifting).

Le journal des étapes et des choix (pour le rapport) est dans [`JOURNAL.md`](JOURNAL.md).

Le pipeline implémente le plan `propose-un-plan-complet-refactored-wave.md` : un protocole de validation unique,
du modèle simple (Ridge) au modèle complexe (GBM, MLP), puis un ensemble. Il a été complété (étape 26 du journal)
par des méthodes publiées en 2024-2026 : variables relatives TabPrep, TabM, RealMLP et le modèle de fondation
TabICLv2.

**Meilleure soumission : stacking Ridge de 37 modèles + doublons exacts, CV 76,075, public 76,247 (1er le 03/11).**
Voir [Reproduire la meilleure soumission](#reproduire-la-meilleure-soumission).

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), un compte Kaggle. Sur macOS, LightGBM et XGBoost ont besoin de
`brew install libomp`.

1. **Rejoindre la compétition** sur Kaggle (*Join Competition*, accepter les règles), sinon le téléchargement
   échoue avec une erreur 403.
2. **Créer un token API** : https://www.kaggle.com/settings/api → *Generate New Token* (`KGAT_...`).
3. **Installer et lancer** :

   ```bash
   uv sync
   uv run main.py train --model lgbm
   ```

   Les données sont téléchargées avec **kagglehub** dans `data/`. Au premier lancement, le token est demandé
   (saisie invisible) puis sauvegardé dans `~/.kaggle/access_token` : il n'est plus redemandé et n'est jamais
   dans le dépôt. Alternative sans prompt : `export KAGGLE_API_TOKEN="KGAT_..."`.

## Utilisation

```bash
uv run main.py train --model ridge                      # M1 : Ridge + splines + interactions Sex × *
uv run main.py train --model rf                         # M3 : Random Forest (et : ExtraTrees)
uv run main.py train --model lgbm                       # M4 : LightGBM
uv run main.py train --model cat                        # M5 : CatBoost
uv run main.py train --model xgb                        # M6 : XGBoost
uv run main.py train --model mlp --target C --seeds 5   # M7 : MLP PyTorch, embeddings + PLR (outputs/params/mlp.json)
uv run main.py train --model mlp --no-tuned             # M7 d'origine (sans PLR, dropout 0,15)
uv run main.py train --model lgbm --target C            # cible : A brut, B log, C points DOTS
uv run main.py train --model lgbm --outliers drop       # retire les lignes aberrantes du train de chaque pli
uv run main.py train --model lgbm --seeds 5             # moyenne sur 5 seeds
uv run main.py train --model lgbm --folds 10            # 10 plis emboîtés dans les 5 de référence (scores sur les 5)
uv run main.py train --model lgbm --target C --set tabprep=true   # + 345 variables relatives TabPrep (tabprep.py)
uv run main.py train --model tabm --target C            # TabM (tabm_net.py), GPU mps / cuda détecté
uv run main.py train --model realmlp --target C         # RealMLP (pytabkit), réglages par défaut « TD »

uv run main.py tune --model lgbm --trials 100           # Optuna ; les meilleurs paramètres sont rechargés ensuite
uv run main.py ensemble                                 # moyenne pondérée + stacking des modèles de ensemble_models.txt
uv run main.py ensemble --all                           # idem sur tous les modèles du journal
uv run main.py dedup ensemble_stack                     # doublons exacts du train -> ensemble_stack_dedup
uv run main.py analyze --oof lgbm_A                     # importance, résidus par segment, dépendances partielles
uv run main.py report                                   # tableau méthode -> RMSE CV -> RMSE publique
uv run main.py score lgbm_A 78.9                        # noter le score public Kaggle d'une expérience
uv run pytest                                           # tests du pipeline
```

### Recherche d'architecture du MLP avec DRAGON

[DRAGON](https://dragon-tutorial.readthedocs.io/en/latest/) (algorithme Mutant-UCB) impose `numpy < 2` : il a son
propre environnement dans `dragon/` et reçoit des matrices déjà prétraitées.

```bash
uv run main.py export-nn --target C                                # matrices des 5 plis -> outputs/nn_data/
uv run --project dragon python dragon/search.py search --iters 400 # recherche sur le pli 0 (~40 min)
uv run --project dragon python dragon/search.py final --seeds 5    # ré-entraîne les meilleures depuis zéro, 5 plis
uv run main.py register dragon_C --seeds 5                         # journal + soumission
```

### TabICLv2 sur GPU Kaggle

TabICLv2 (Qu et al., `tabicl` 2.2.0) a besoin d'un GPU (≈ 19 min par sous-pli sur une T4, 8,7 Go). Il tourne dans un
kernel Kaggle privé, sans internet : le paquet, `einops` et les poids sont fournis dans un dataset privé.

```bash
uv run python tabicl/export.py                          # entrées des 10 sous-plis -> tabicl/data/
uvx pip download tabicl==2.2.0 einops==0.8.2 --no-deps -d tabicl/data
uvx --from huggingface_hub hf download jingang/TabICL tabicl-regressor-v2-20260212.ckpt --local-dir tabicl/data
cp tabicl/dataset-metadata.json tabicl/data/ && uvx --with six kaggle datasets create -p tabicl/data   # « version » ensuite
uvx --with six kaggle kernels push -p tabicl/kernel     # GPU T4 (compte Kaggle vérifié par téléphone), ~3 h
uvx --with six kaggle kernels output felxoll/powerlifting-tabicl-gpu -p tabicl/out
uv run python tabicl/assemble.py tabicl/out             # -> outputs/oof|preds/tabicl_C_k10.npy
uv run main.py register tabicl_C_k10 --model tabicl --target C
```

Adapter l'identifiant `felxoll/...` dans `tabicl/dataset-metadata.json` et `tabicl/kernel/kernel-metadata.json`.
Un seul processus TabICL à la fois : à deux, le tampon de sortie (≈ 17 Go chacun) dépasse la RAM de la machine.

### Reproduire la meilleure soumission

L'ensemble soumis empile les 37 modèles de [`ensemble_models.txt`](ensemble_models.txt). Les prédictions OOF et
test des 33 premiers sont versionnées dans `outputs/oof/` et `outputs/preds/`. Leurs commandes sont dans
`logs/queue_k10.sh`, `logs/queue_k10_v1.sh`, `logs/queue_cours5.sh` et les étapes 13 à 25 du journal. Les 4 derniers
(étape 26.1) :

```bash
uv run main.py train --model lgbm --target C --folds 10 --seeds 3 --name lgbm_C_tabprep_k10 --set tabprep=true  # 45 min
uv run main.py train --model tabm --target C --folds 10 --name tabm_C_k10                                       # 2 h 30 (MPS)
uv run main.py train --model realmlp --target C --folds 10 --name realmlp_C_k10                                 # 1 h (MPS)
# tabicl_C_k10 : voir « TabICLv2 sur GPU Kaggle »
uv run main.py ensemble                     # stacking Ridge des 37 : 76,214
uv run main.py dedup ensemble_stack         # doublons exacts : 76,075 -> outputs/submissions/ensemble_stack_dedup.csv
```

Au plus deux entraînements lourds en même temps sur un Mac de 16 Go.

Chaque `train` écrit `outputs/submissions/<nom>.csv` (format `id,TotalKg`, contrôlé avant écriture).
Lancer **un modèle par commande** : PyTorch et LightGBM/CatBoost/XGBoost ne cohabitent pas dans un même processus
sur macOS (conflit OpenMP qui bloque le processus), ce que la CLI évite naturellement.

### Soumettre à Kaggle

`kagglehub` ne gère que le téléchargement ; la soumission se fait sur la page de la compétition
(*Submit Predictions*) ou avec la CLI `kaggle` :

```bash
kaggle competitions submit -c performances-en-powerlifting -f outputs/submissions/ensemble_stack.csv -m "stacking"
```

## Organisation du code

| Fichier | Rôle (phase du plan) |
|---|---|
| `data.py` | téléchargement via kagglehub, connexion unique au token |
| `config.py` | chemins, seed, nombre de plis |
| `features.py` | nettoyage « Unknown », `WeightClassKg`/`AgeClass` décomposés, features de domaine (Wilks / DOTS / IPF GL, âge, année de naissance, `BW/wc_limit`, interactions, fréquences, taille de la compétition), transformations de cible A/B/C, target encoding sans fuite, y compris de la compétition (fédération × année × lieu) (phases 1-3) |
| `cv.py` | plis fixes (`data/folds.csv`), OOF, journal `outputs/experiments.csv` (phase 0) |
| `models.py` | les 11 modèles (phase 4, + `tabm` et `realmlp`) ; `nn_arrays` : prétraitement des réseaux |
| `tabprep.py` | variables relatives TabPrep (GroupBy ratio / écart / rang dans le groupe, arithmétique OAFE), apprises sur le train du pli ; `lgbm --set tabprep=true` |
| `tabm_net.py` | réseau TabM (BatchEnsemble de 32 têtes, plongements linéaires par morceaux) et sa boucle d'entraînement |
| `tabicl/` | TabICLv2 sur GPU Kaggle : export des entrées, kernel, assemblage des 10 sous-plis |
| `ensemble_models.txt` | les 37 modèles de l'ensemble soumis (défaut de `main.py ensemble`) |
| `experiments/` | environnement séparé où les méthodes de l'étape 26 ont été évaluées avant intégration (rapports `*/REPORT.md`) |
| `tabnn.py` | briques PyTorch du MLP (embeddings, PLR, boucle d'entraînement), partagées avec DRAGON |
| `dragon/` | recherche d'architecture DRAGON (environnement séparé, `search.py`) |
| `tune.py` | Optuna sur les plis fixes |
| `ensemble.py` | moyenne pondérée + stacking Ridge, avec validation imbriquée (phase 5) |
| `submit.py` | contrôles et écriture des soumissions (phase 6) |
| `postprocess.py` | post-traitement des doublons exacts (prédiction remplacée par la cible moyenne des lignes identiques du train) |
| `analysis.py` | importance (gain, permutation, SHAP), résidus, dépendances partielles (phase 7) |
| `tests/` | vérifications du plan |

## Protocole de validation

- **5 plis** stratifiés sur (décile de `TotalKg` × sexe), fixés une fois dans `data/folds.csv` et **versionnés** :
  le binôme partage les mêmes plis.
- Tout ce qui dépend de la cible (target encoding du résidu `y − moyenne(Sex × Equipment)`) est appris **dans le pli** ;
  pour les lignes d'entraînement, une K-fold interne évite d'encoder une ligne avec sa propre cible.
- Chaque expérience garde ses prédictions **OOF** (`outputs/oof/`) et test (`outputs/preds/`, moyenne des 5 modèles de
  pli), utilisées par l'ensemble. `outputs/experiments.csv` est le journal partagé (CV ± écart-type, score public).
- Les estimations de l'ensemble sont faites en **validation imbriquée** (poids ou méta-modèle appris sur 4 plis d'OOF,
  évalués sur le 5ᵉ).
- Choix de la soumission finale : sur la **CV**, pas sur le leaderboard public.

## Choix d'implémentation à connaître

- Early stopping sur le pli de validation (comme dans le plan) : l'OOF est très légèrement optimiste ; l'écart avec
  le leaderboard public reste à surveiller.
- Cible B (log) : correction du biais de retransformation `exp(p + σ²/2)` avec σ² estimée sur le pli de validation.
- Cible C (points DOTS) : perte pondérée par `(dénominateur/500)²`, ce qui équivaut à minimiser l'erreur en kg.
- Les formules Wilks / DOTS / IPF GL ne servent que de features et de mise à l'échelle : leurs coefficients sont
  ceux publiés, mais aucune performance ne dépend de leur exactitude.
- Lignes « aberrantes » (option `--outliers drop`) : poids de corps < 30 kg ou `TotalKg/BW` < 1 ou > 11,5 (80 lignes).
  Elles ne sont retirées que du train de chaque pli, jamais de la validation ni du test.

## Résultats actuels (03/11, RMSE en kg, CV sur les 5 plis de référence)

| Expérience | Méthode | RMSE CV | Public |
|---|---|---|---|
| `lgbm_C_tabprep_k10` | LightGBM tuné + variables TabPrep, 10 plis, 3 seeds | **76,73** (meilleur modèle seul) | |
| `lgbm_C_k10` | LightGBM tuné, features v2, 10 plis, 5 seeds | 76,86 | |
| `tabm_C_k10` | TabM, 10 plis | **76,99** (meilleur réseau) | |
| `mlp_C_plr_te_k10` | MLP PLR + target encoding, 10 plis, 5 seeds | 77,18 | |
| `tabicl_C_k10` | TabICLv2 (modèle de fondation, apprentissage en contexte), 10 plis | 77,60 | |
| `realmlp_C_k10` | RealMLP, 10 plis | 77,74 | |
| `ensemble_stack` | stacking Ridge des 37 modèles (validation imbriquée) | 76,21 | |
| **`ensemble_stack_dedup`** | **+ doublons exacts** | **76,075** | **76,247** |

Avant l'étape 26, l'ensemble à 33 modèles faisait 76,213 en CV et 76,426 en public. Un méta-modèle non linéaire
ou contextuel ne fait pas mieux que la Ridge (étape 26.2).

## Premiers résultats (septembre-octobre, CV 5 plis, RMSE en kg)

| Expérience | Méthode | RMSE CV | Remarque |
|---|---|---|---|
| `const` | constante | 171,4 | repère du plan (171) |
| `hgb_raw` | HistGradientBoosting sur les 16 variables brutes | 79,0 | repère du plan (≈ 79) |
| `ridge_A` | M1 : Ridge + splines + interactions Sex × * | 81,6 | mieux que l'objectif du plan (95-100) |
| `rf_A` / `et_A` | M3 : Random Forest / ExtraTrees | 81,1 / 82,3 | |
| `xgb_A` | M6 : XGBoost | 78,9 | |
| `cat_A` | M5 : CatBoost | 78,1 | 13 min |
| `mlp_A_s5` | M7 : MLP avec embeddings, 5 seeds | 78,5 | 79,6 avec 1 seed |
| `mlp_C_plr_s5` | M7 + cible DOTS + plongement PLR des continues, dropout 0,3, lr 1e-3, 5 seeds | 77,80 | gagne sur les 5 plis |
| `dragon_C` | architecture du MLP cherchée par DRAGON (Mutant-UCB, 400 évaluations), 5 seeds | 78,27 | moins bon que `mlp_C_plr_s5` sur les 5 plis |
| `lgbm_A` / `_B` / `_C` | M4 : LightGBM, paramètres par défaut | 78,1 / 78,3 / 78,0 | cibles brut / log / DOTS |
| `lgbm_A_drop` | idem sans les 80 lignes aberrantes | 78,1 | pas de gain : on garde tout |
| `lgbm_C_tuned` | LightGBM tuné (Optuna, 30 essais), 5 seeds, cible DOTS | **77,49** | meilleur modèle seul |
| **ensemble pondéré** | 6 modèles à poids ≥ 0 (dont B/C tunés, CatBoost, MLP) | **77,14** | validation imbriquée (77,19 avant le nouveau MLP) |
| **stacking Ridge** | méta-modèle sur les OOF + Sex/Equipment | **77,06** | validation imbriquée (77,14 avant le nouveau MLP) |
| **`ensemble_stack_dedup`** | stacking + doublons exacts du train (867 lignes en CV, 465 en test) | **76,89** | mieux sur les 5 plis |

Le bruit irréductible est élevé (le niveau individuel de l'athlète n'est pas observé) : les écarts entre plis
sont de ±0,4, donc les différences de quelques dixièmes entre variantes proches ne sont pas significatives.
Les scores publics restent à renseigner avec `uv run main.py score <expérience> <score>`.

Constats de l'analyse (`uv run main.py analyze`) : les dénominateurs IPF GL / Wilks / DOTS dominent l'importance
(ils résument poids de corps × sexe × équipement) ; l'erreur est plus forte chez les hommes (84 contre 59),
en Multi-ply (109) et pour les athlètes de plus de 105 kg. Seules 9 lignes ont `Sex = Mx` (RMSE ≈ 236) : catégorie
trop rare pour être apprise.
