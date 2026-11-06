# TabPrep sur LightGBM : rapport d'ablation (03/11)

*Question posée :* les features automatiques de TabPrep (Tschalzev, Erickson et al., 2026, arXiv 2606.02384)
apportent-elles quelque chose que le feature engineering manuel (features v2 + TE résiduel) n'a pas déjà ?

*Protocole :* LightGBM tuné (`MODELS["lgbm"]["params"]` + `outputs/params/lgbm.json`), cible C, 5 plis de
référence, `n_jobs=3`. Les colonnes TabPrep sont **ajoutées** aux features v2 et au TE résiduel du projet
(`models._with_te`). Tout est appris dans le pli d'entraînement. Référence : **77,201** (plis 76,656 / 77,305 /
76,804 / 77,583 / 77,659). Relancée ici avec `n_jobs=3`, elle redonne exactement les mêmes chiffres (LightGBM est
déterministe à graine fixe). Aucun fichier du projet n'a été modifié.

## 1. Ce qui a été implémenté (`tp.py`)

AutoGluon n'est pas installé (le dépôt `atschalz/tabprep` n'est qu'un wrapper de `autogluon.features.generators`,
dont l'installation complète est lourde). Les générateurs sont réimplémentés en numpy/pandas d'après le code source
d'AutoGluon (`rsfc.py`, `groupby.py`, `arithmetic/`, `oof_target_encoder.py`), avec les mêmes défauts :

| Générateur | Défauts repris d'AutoGluon | Appliqué à |
|---|---|---|
| **OOF-TE** (commun) | 5 plis internes, lissage α vers la moyenne *non pondérée* des moyennes de modalités, modalité inconnue → cette moyenne | cible du modèle (points DOTS), sans poids |
| **RSFC** | 50 sous-ensembles (le 1er = toutes les variables), taille tirée dans [2, p−1], numériques arrondies à 2 décimales, OOF-TE **sans lissage** (α = 0) | les 16 variables brutes |
| **GroupBy relatifs** | `pct_rank` (rang dans la distribution du groupe sur le train) + `ratio` à la moyenne du groupe ; catégorielles classées selon la taille de leurs plus petits groupes (≥ 20), numériques par cardinalité ; budget 500 paires ; statistiques apprises sur le train du pli | 13 catégorielles brutes × 3 numériques brutes (78 colonnes) |
| **OAFE** | sélection « random » : paires dans un ordre aléatoire, opérateurs /, ×, −, + puis b/a ; ordre 3 = opération d'ordre 2 ∘ variable absente ; filtre de base (cardinalité ≥ 3, mode ≤ 90 %, NaN ≤ 95 %) ; remise à l'échelle | 3 numériques brutes (75 colonnes) |
| **CatPrep** (combinaisons) | OOF-TE (α = 10) de chaque catégorielle + 100 combinaisons aléatoires d'ordre 2-3 | 13 catégorielles brutes |

**Différences avec le papier :**
- Les catégorielles d'origine ne sont **pas** remplacées par leur OOF-TE : LightGBM garde ses catégorielles
  natives, dont `State` / `MeetState` qui valent 0,5 à elles seules (journal, étape 18). TabPrep est ajouté au
  pipeline existant, pas substitué.
- Pas de HPO sur les options TabPrep (le papier tire TabPrep dans 25 % de 200 configurations). Les hyperparamètres
  LightGBM restent ceux du tuning du projet. Les budgets sont fixés à la main : on teste quelques variantes.
- Le pipeline par défaut est appliqué aux **16 variables brutes**, comme TabPrep le ferait sur le jeu d'origine.
  Les variantes « larges » l'appliquent aussi à des colonnes v2 (`birth_year`, `bw_over_wc`, `dots_den`,
  `Sex_Equip`…) ou au TE.
- OAFE à 2 000 colonnes sur les ~30 numériques v2 (défaut du papier) n'a **pas** été lancé : avec 500 colonnes, le
  processus montait déjà à 3,4 Go (consigne : < 4 Go). Les 500 premières n'apportant rien (§ 2), l'extrapolation
  est sans risque.

## 2. Ablation (LightGBM seul, 1 seed, 5 plis ; écart par pli à la référence, négatif = mieux)

| Variante | Nb col. | RMSE | Écart par pli | Plis mieux | Temps (5 plis) | RAM max |
|---|---|---|---|---|---|---|
| référence | 0 | 77,202 | 0 / 0 / 0 / 0 / 0 | | 0,8 min | 0,6 Go |
| **RSFC** fidèle | 50 | 77,323 | +0,19 / +0,22 / +0,08 / +0,14 / −0,01 | 1/5 | 1,0 min | 0,8 Go |
| RSFC adapté (tailles 2-4, 16 quantiles, α = 20) | 50 | 77,369 | +0,19 / +0,16 / +0,20 / −0,04 / +0,33 | 1/5 | 0,9 min | 0,9 Go |
| **CatPrep** (TE + 100 combinaisons) | 113 | 77,282 | +0,06 / +0,14 / +0,02 / +0,08 / +0,10 | 0/5 | 1,5 min | 1,1 Go |
| **OAFE** fidèle (3 numériques brutes, ordre 3) | 75 | 77,188 | +0,23 / −0,06 / −0,06 / −0,07 / −0,11 | 4/5 | 1,4 min | 0,9 Go |
| OAFE sur les numériques v2 (budget 500) | 500 | 77,220 | +0,13 / +0,01 / +0,05 / −0,01 / −0,10 | 2/5 | 4,1 min | 3,4 Go |
| OAFE sur numériques brutes + 7 colonnes `te_*` (500) | 500 | 77,645 | +0,59 / +0,36 / +0,46 / +0,45 / +0,36 | 0/5 | 3,7 min | 2,8 Go |
| **GroupBy** fidèle (13 cat. × 3 num., ratio + rang) | 78 | **77,127** | +0,08 / −0,15 / −0,18 / −0,11 / −0,02 | 4/5 | 1,4 min | 1,0 Go |
| GroupBy, `diff` au lieu de `ratio` | 78 | 77,141 | +0,04 / −0,09 / −0,08 / −0,06 / −0,12 | 4/5 | 1,6 min | 1,0 Go |
| GroupBy transductif (stats sur train + val + test) | 78 | 77,169 | +0,03 / −0,02 / −0,09 / −0,05 / −0,04 | 4/5 | 1,5 min | 1,0 Go |
| GroupBy `WeightClassKg` seul | 6 | 77,187 | +0,05 / −0,01 / −0,01 / −0,05 / −0,05 | 4/5 | 0,8 min | 0,7 Go |
| GroupBy sans `WeightClassKg` | 72 | 77,192 | +0,12 / −0,10 / −0,12 / −0,06 / +0,10 | 3/5 | 1,3 min | 1,0 Go |
| GroupBy large (15 cat. × 6 num. v2, ratio + diff + rang) | 270 | 77,124 | +0,09 / −0,12 / −0,07 / −0,15 / −0,14 | 4/5 | 3,1 min | 2,0 Go |
| GroupBy relatif à la **compétition** (`Meet`, `Meet_Div` × 6 num.) | 36 | 77,581 | +0,55 / +0,42 / +0,38 / +0,25 / +0,30 | 0/5 | 0,8 min | 0,8 Go |
| idem, transductif | 36 | 77,432 | +0,24 / +0,26 / +0,11 / +0,36 / +0,18 | 0/5 | 1,0 min | 2,0 Go |
| GroupBy + OAFE | 153 | 77,145 | +0,10 / −0,21 / −0,15 / +0,01 / −0,04 | 3/5 | 2,0 min | 2,0 Go |
| GroupBy large + OAFE | 345 | 77,110 | +0,07 / −0,17 / −0,05 / −0,16 / −0,14 | 4/5 | 3,8 min | 2,2 Go |

Le calcul des features coûte 0,2 à 2 s par pli ; tout le surcoût vient de l'entraînement sur plus de colonnes
(×1,7 pour GroupBy, ×4,5 pour 300 à 500 colonnes). Le papier annonce ×6. Calcul total de l'étude : environ 75 min.

**Contrôle du bruit de seed** (3 seeds LightGBM, chaque variante comparée à la référence *de même seed*) :
l'écart-type de la référence entre seeds est de 0,02 à 0,07 par pli (77,202 / 77,210 / 77,184).

| Variante | Écart RMSE par seed | Moyenne | Écart moyen par pli (3 seeds) | (pli, seed) mieux |
|---|---|---|---|---|
| **GroupBy** fidèle | −0,075 / −0,108 / −0,074 | **−0,086** | −0,05 / −0,10 / −0,11 / −0,14 / −0,04 | **14/15** |
| GroupBy large | −0,078 / −0,117 / −0,053 | −0,083 | +0,02 / −0,06 / −0,14 / −0,15 / −0,08 | 13/15 |
| GroupBy large + OAFE | −0,092 / −0,097 / −0,051 | −0,080 | +0,07 / −0,08 / −0,08 / −0,15 / −0,16 | 12/15 |
| GroupBy + OAFE | −0,057 / −0,057 / −0,098 | −0,071 | +0,05 / −0,13 / −0,13 / −0,09 / −0,06 | 12/15 |
| GroupBy transductif | −0,033 / −0,084 / −0,010 | −0,042 | −0,03 / −0,06 / −0,06 / −0,04 / −0,03 | 11/15 |
| GroupBy `WeightClassKg` seul | −0,015 / −0,042 / −0,034 | −0,030 | 0,00 / +0,02 / −0,04 / −0,09 / −0,04 | 10/15 |
| GroupBy sans `WeightClassKg` | −0,010 / −0,030 / −0,057 | −0,032 | +0,04 / −0,07 / −0,12 / −0,06 / +0,05 | 9/15 |
| OAFE fidèle | −0,014 / −0,030 / −0,034 | −0,026 | +0,15 / −0,01 / −0,12 / −0,10 / −0,05 | 10/15 |

## 3. Gain dans le stacking Ridge à 33 modèles (`common.evaluate`, base 76,3699)

| Modèle ajouté | RMSE seul | Δ stacking | Δ par pli |
|---|---|---|---|
| *contrôle* : LightGBM de référence, 1 seed | 77,202 | +0,0011 | +0,003 / +0,001 / −0,000 / +0,001 / +0,002 |
| *contrôle* : LightGBM de référence, 3 seeds | 77,067 | +0,0006 | −0,000 / −0,001 / +0,002 / +0,001 / +0,002 |
| OAFE fidèle, 3 seeds | 77,022 | −0,0290 | +0,008 / −0,035 / −0,054 / −0,046 / −0,018 |
| GroupBy fidèle, 1 seed | 77,127 | −0,0305 | −0,020 / −0,039 / −0,051 / −0,040 / −0,003 |
| GroupBy fidèle, 3 seeds (`tabprep_gb_lgbm`) | 76,967 | −0,0425 | −0,048 / −0,052 / −0,037 / −0,056 / −0,019 |
| GroupBy + OAFE, 3 seeds | 76,971 | −0,0526 | −0,024 / −0,082 / −0,065 / −0,049 / −0,042 |
| GroupBy large, 3 seeds | 76,952 | −0,0621 | −0,037 / −0,066 / −0,075 / −0,079 / −0,054 |
| **GroupBy large + OAFE, 3 seeds (`tabprep_lgbm`)** | **76,948** | **−0,0709** | **−0,029 / −0,089 / −0,071 / −0,086 / −0,080** |

Corrélation des résidus de `tabprep_lgbm` avec `lgbm_C_k10` : 0,989 (0,993 pour le LightGBM de référence), 0,978 avec
le MLP, 0,976 avec CatBoost. Le contrôle est net : un LightGBM de plus *sans* TabPrep n'apporte rien (+0,001) ; avec
TabPrep, le gain est de −0,03 à −0,07, **sur les 5 plis**. C'est l'ordre de grandeur des meilleurs modèles neufs
(TabICLv2, RealMLP, TabM : −0,017 à −0,05), voire un peu plus. Ce gain est une borne basse : les 33 modèles sont à
10 plis, celui-ci à 5.

## 4. Lecture

- **Ce qui marche : les GroupBy relatifs sur les catégorielles « larges ».** Les colonnes les plus utilisées sont
  la position du poids de corps dans sa catégorie de poids (`BodyweightKg` ratio / rang / écart dans
  `WeightClassKg`, `bw_over_wc` rang dans `WeightClassKg`) et le rang de l'âge selon l'équipement, le sexe ×
  équipement et `Tested`. `bw_over_wc` (poids / limite) existait déjà, mais il n'est pas défini pour les catégories
  « + » et ne dit pas où se situe l'athlète parmi les autres de sa catégorie. Le rang de l'âge par équipement est
  une interaction qu'un arbre peut apprendre, mais qu'il obtient ici en un seul split. Les deux moitiés
  (`WeightClassKg` seul, le reste) apportent chacune environ −0,03 ; réunies, −0,086.
- **C'est réellement nouveau par rapport au projet** : les rangs déjà testés (étape 18) portaient sur la
  *compétition*. Ici aussi, rapporter le poids ou l'âge à la compétition **nuit** nettement (+0,25 à +0,55 sur
  5 plis). Les groupes sont trop petits (≈ 5 lignes en médiane) : la moyenne apprise sur le train inclut la ligne
  elle-même, mais pas pour la validation. Ce décalage train / validation est réduit en mode transductif, sans
  disparaître.
- **Stats apprises sur le train du pli (TabPrep) > transductif** pour les catégorielles larges (−0,086 contre
  −0,042). L'écart est peu interprétable, mais la variante fidèle est la meilleure.
- **OAFE** : marginal seul (−0,026, et le pli 0 se dégrade à chaque seed : +0,15). Il apporte surtout de la
  diversité au stacking. Sur les numériques v2 (formules de points, puissances…) ou sur les TE, l'expansion est
  neutre ou nuisible : le feature engineering manuel a déjà les bonnes transformations.
- **RSFC et CatPrep : nuisibles** (+0,08 à +0,17, 0 à 1 pli sur 5). Leurs colonnes target-encodées captent 54 à
  79 % du gain des arbres et font sur-apprendre LightGBM. C'est la même conclusion que les composites TE du journal
  (étapes 18 et 25.4) : au-delà de la compétition, les combinaisons target-encodées ajoutent du bruit. RSFC fidèle
  est en plus mal adapté au jeu : avec 16 variables, les sous-ensembles font en moyenne 8 colonnes et contiennent
  souvent le poids à 2 décimales. Les clés sont alors presque uniques et l'encodage revient à une constante.

## 5. Verdict

TabPrep apporte **un gain réel mais modeste**, et il vient d'un seul générateur : les **GroupBy relatifs sur les
catégorielles larges** (catégorie de poids, équipement, sexe, `Tested`…). LightGBM seul gagne −0,086 (3 seeds,
14 couples (pli, seed) sur 15). Dans le stacking à 33, le gain est de −0,04 à −0,07 sur les 5 plis, contre +0,001
pour un LightGBM de plus sans TabPrep. RSFC et CatPrep dégradent ; OAFE est marginal seul mais ajoute un peu de
diversité.

**Recommandation :** ajouter au projet les GroupBy relatifs du poids et de l'âge (ratio / écart / rang dans
`WeightClassKg`, `Equipment`, `Sex_Equip`, `Equip_Tested`, `Sex`, `Tested`, `AgeClass`…), calculés sur le train du
pli. Réentraîner ensuite LightGBM (et les autres modèles) à 10 plis, puis vérifier dans le stacking. Attendu : −0,05
à −0,1 pour LightGBM seul, quelques centièmes pour l'ensemble. Le gain dans l'ensemble risque de diminuer si
*tous* les modèles reçoivent ces features, puisqu'il vient ici de la diversité.

## Fichiers

- `tp.py` : générateurs (RSFC, GroupBy relatifs, OAFE, CatPrep, OOF-TE) ; `run.py` : variantes et boucle CV
  (`uv run python tabprep/run.py VARIANTE [--seed s] [--folds k]`) ; `summ.py`, `seeds.py` : tableaux ;
  `stack_eval.py` : gain dans le stacking ; `final.py` : livrable.
- `results.jsonl` : un enregistrement par run (plis, temps, RAM, part du gain des nouvelles colonnes, top
  features) ; `logs/` ; `cache/` : OOF et test de chaque run à 5 plis.
- `experiments/results/tabprep_lgbm_{oof,test}.npy` : GroupBy large + OAFE, moyenne de 3 seeds (76,948 ;
  Δ stacking −0,0709) ; `tabprep_gb_lgbm_{oof,test}.npy` : GroupBy fidèle, 3 seeds (76,967 ; Δ −0,0425).
