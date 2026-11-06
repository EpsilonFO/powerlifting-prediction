# Journal de bord du projet – Régression TotalKg (powerlifting)

Ce journal retrace, dans l'ordre, ce qui a été fait, **pourquoi** et ce que ça a donné. Il sert de matière
première pour le rapport. Les chiffres sont des **RMSE en kg, en validation croisée sur 5 plis** (sauf mention
contraire) et viennent de `outputs/experiments.csv`.

> Repère pour lire les écarts : la RMSE varie d'environ **±0,4** d'un pli à l'autre. Une différence de quelques
> dixièmes entre deux variantes proches n'est donc pas significative, sauf si elle va dans le même sens sur les
> 5 plis.

## Vue d'ensemble

| Date | Étape | Meilleur score à ce stade |
|---|---|---|
| 28/09 | EDA, protocole de validation, prétraitement, features | – |
| 28/09 | Repères : constante, HistGradientBoosting brut | 79,0 |
| 28/09 | Modèles classiques en parallèle (Ridge, RF, ET, LightGBM, XGBoost, CatBoost, MLP) | 78,0 (LightGBM, cible C) |
| 28/09 | Choix de la cible (A / B / C) et des valeurs aberrantes | 78,0 |
| 29/09 | Tuning Optuna de LightGBM + moyenne sur 5 seeds | 77,49 |
| 29/09 | Ensemble : moyenne pondérée et stacking | 77,14 |
| 29/09 | Analyse : importances, résidus par segment | – |
| 19/10 | Bilan, test des pistes « doublons » et « pseudo-athlètes » | (77,31 en post-traitement, non intégré) |
| 19/10 | MLP retravaillé : diagnostic, cible C, plongement PLR | 77,80 (MLP seul) |
| 19/10 | Recherche d'architecture avec DRAGON | 78,27 (moins bon que le MLP réglé à la main) |
| 19/10 | Ensemble avec le nouveau MLP | 77,06 |
| 19/10 | Post-traitement des doublons exacts sur le stacking | 76,89 (public : 77,04) |
| 20/10 | Lecture des cours, analyse et ablation des features, nouvelles features « compétition » | LightGBM seul : 77,61 → 77,18 (1 seed) |
| 21/10 | Modèles réentraînés avec les features v2, nouvel ensemble + doublons | 76,47 |
| 31/10 | Cours 5 : méta-modèle, contribution marginale, forêts et XGBoost tunés, variantes LightGBM | 76,43 (public : 76,47) |
| 01/11 | 10 modèles principaux réentraînés sur 10 plis emboîtés | 76,24 (public : 76,440, 2ᵉ du classement) |
| 01/11 | 5 anciens modèles (features v1) réentraînés sur 10 plis | **76,23** (public : 76,443) |
| 01/11 | Mesures complémentaires sur le prétraitement (athlète non identifiable par construction, TE au plafond) ; MLP + target encoding : −0,21, CatBoost + clé de compétition : −0,15 (10 plis) ; ensemble à 33 modèles | 76,21 (public : 76,426, 2ᵉ) |
| 03/11 | Veille 2024-2026 ; TabICLv2, TabM, RealMLP, HL-Gauss, TabPrep évalués hors projet | (76,23 estimé avec 4 nouveaux modèles) |
| 03/11 | 4 nouveaux modèles intégrés à 10 plis (LightGBM + TabPrep, TabM, RealMLP, TabICLv2) ; ensemble à 37 modèles | **76,075 (public : 76,247, 1er)** |
| 03/11 | Méta-modèle non linéaire ou contextuel | pas mieux que la Ridge |

---

## 1. Contexte et contraintes (28/09)

- Compétition Kaggle `performances-en-powerlifting` (data challenge du M2 Maths & IA, Paris-Saclay).
- Prédire `TotalKg` (squat + développé couché + soulevé de terre) à partir de 16 variables sur l'athlète et la
  compétition. Métrique : **RMSE**.
- Contraintes du cours : au moins 1 soumission avant le cours 4, et **au moins 3 soumissions issues de 3 méthodes
  différentes avant le 09/11/2026**. 15 soumissions par jour au maximum.
- Démarche choisie : un protocole de validation unique, puis une progression du plus simple au plus complexe
  (utile pour le rapport), puis un ensemble.

## 2. Exploration des données (28/09)

Constats qui ont guidé tous les choix suivants :

- **Taille** : 105 000 lignes en train et 45 000 en test. Aucun NaN, mais beaucoup de valeurs **« Unknown »** :
  `State` 64 %, `MeetState` 34 %, `ParentFederation` 22 %, `BirthYearClass` 3 %, `Country` 2 %,
  `WeightClassKg` 1 %.
- **Cible** : moyenne de 493 kg, écart-type de 171, de 24 à 1363 kg, peu asymétrique (0,27). Le log la rend
  asymétrique à gauche (−0,59), donc il n'apporte rien *a priori*.
- **Découpage train/test aléatoire, pas temporel** : même distribution de `MeetYear` des deux côtés, catégories
  absentes du train quasi inexistantes. → **K-fold aléatoire** comme validation.
- **Variables clés** : `BodyweightKg` (corrélation de 0,66), `Sex` (F ≈ 323, M ≈ 569 kg), `Equipment` (Multi-ply
  ≈ 669, Raw ≈ 469), `Tested` (+52 kg pour les compétitions non contrôlées), effet `MeetYear` négatif
  (démocratisation du sport), effet de l'âge non linéaire.
- **Cardinalités élevées** : `Federation` 399, `MeetState` 240, `Country` 149, `WeightClassKg` 146 (formats
  mélangés : `90`, `93+`, `-93`, `+`, `Unknown`).
- **Âge** : 48 % des valeurs en x,5 (âge estimé à partir de l'année de naissance).
- **Valeurs aberrantes** : poids de corps < 35 kg (107 lignes), âge < 10 ans (140 lignes), total < 100 kg
  (68 lignes).
- **Hétéroscédasticité** : l'erreur est plus forte chez les hommes et en Multi-ply.
- **Doublons** : 558 lignes du train identiques sur toutes les variables, et 465 lignes du test identiques à une
  ligne du train (exploité à l'étape 10).

## 3. Protocole de validation (28/09)

| Choix | Pourquoi |
|---|---|
| 5 plis **stratifiés sur (décile de `TotalKg` × sexe)** | des plis comparables en niveau et en proportion hommes/femmes |
| Plis **fixés une fois** dans `data/folds.csv` et versionnés | le binôme compare ses modèles sur exactement les mêmes plis |
| Tout ce qui dépend de la cible est appris **dans le pli** | éviter les fuites (score de CV trop optimiste) |
| Chaque modèle garde ses **prédictions OOF** et test | indispensable pour construire l'ensemble ensuite |
| Journal partagé `outputs/experiments.csv` | suivre CV ± écart-type et, plus tard, le score public |
| Choix final sur la **CV**, pas sur le leaderboard public | le public ne porte que sur une partie des 45 000 lignes |

Tests automatiques (`uv run pytest`) : les plis ne se recouvrent pas, le target encoding n'utilise jamais la cible
de la ligne encodée, les fichiers OOF correspondent au journal, et les soumissions ont le bon format.

## 4. Prétraitement (28/09) – `features.py`

- **« Unknown » → valeur manquante**, plus des indicatrices `is_unknown_*` pour `State`, `MeetState`,
  `ParentFederation` et `BirthYearClass`. *Pourquoi :* l'absence d'information est informative en elle-même (par
  exemple, une fédération sans `ParentFederation` est souvent petite).
- **`WeightClassKg` décomposée** en borne numérique (`wc_limit`), catégorie ouverte (`wc_is_plus`) et inconnue
  (`wc_unknown`) ; la version catégorielle brute est conservée.
- **`AgeClass` et `BirthYearClass`** converties en bornes basse, haute et milieu (variables ordinales) en plus de
  la version catégorielle.
- **Modalités rares** : regroupées pour les modèles qui en ont besoin (one-hot du Ridge : < 25 lignes ; HGB : 200
  modalités les plus fréquentes ; MLP : < 5 lignes). LightGBM et CatBoost les gèrent nativement.
- **Écrêtage (winsorisation) au 0,5 % / 99,5 %** des variables numériques pour Ridge et MLP, sensibles aux valeurs
  extrêmes.

## 5. Feature engineering (28/09) – `features.py`

Idée directrice : injecter la connaissance du domaine, puisque le niveau individuel de l'athlète n'est pas observé.

- **Physiologie** : `log(BW)`, `BW^(2/3)` (loi d'échelle allométrique), et les **dénominateurs des formules Wilks,
  DOTS et IPF GL**. Ce sont des fonctions du poids de corps (et du sexe, et de l'équipement pour IPF GL) utilisées
  en powerlifting pour comparer des athlètes de poids différents.
- **Âge** : `Age²`, distance au pic de performance (~30 ans), indicatrices junior (< 23) et master (≥ 40), et
  indicatrice d'âge en x,5.
- **Relation à la catégorie de poids** : `BW / wc_limit`. Être proche de la limite haute de sa catégorie signale
  un compétiteur qui a préparé sa pesée.
- **Interactions** : `Sex × Equipment`, `Equipment × Tested`, `Federation × Equipment`, `BW × sexe`.
- **Contexte** : athlète étranger (`Country ≠ MeetCountry`), époque (avant 2000, période COVID), fréquence de
  chaque fédération / pays / lieu.
- **Target encoding** de `Federation`, `ParentFederation`, `Country`, `MeetCountry` et `MeetState`. On n'encode
  pas la moyenne brute de la cible mais celle du **résidu `y − moyenne(Sex × Equipment)`**, lissée : c'est le
  « niveau » de chaque fédération ou pays, à sexe et équipement donnés. Il est appris dans le pli, avec une K-fold
  interne pour qu'une ligne ne soit jamais encodée avec sa propre cible.

Au total : 56 variables (40 numériques, 16 catégorielles), plus 5 de target encoding pour les modèles qui l'utilisent.

## 6. Points de repère (28/09)

| Expérience | RMSE CV | Rôle |
|---|---|---|
| `const` (moyenne) | 171,4 | plancher absolu ; vérifie le protocole (≈ écart-type de la cible) |
| `hgb_raw` (HistGradientBoosting sur les 16 variables brutes) | 79,0 | ce qu'on obtient « sans rien faire » |

Ces deux repères reproduisent ceux de l'EDA (171 et ≈ 79), ce qui confirme que le pipeline n'a pas de bug.

## 7. Modèles classiques en parallèle (28/09) – `models.py`

Tous entraînés sur les mêmes plis et les mêmes features, cible brute (A), paramètres par défaut raisonnables :

| Modèle | RMSE CV | Remarque |
|---|---|---|
| M1 Ridge + splines (BW, âge) + interactions Sex × tout + splines × (Sex × Equipment) | 81,6 | bien mieux que l'objectif du plan (95-100), grâce aux features de domaine |
| M3 Random Forest | 81,1 | |
| M3 ExtraTrees | 82,3 | |
| M6 XGBoost | 78,9 | non tuné |
| M5 CatBoost | 78,1 | encodage natif des catégories, mais 13 min |
| M4 LightGBM | 78,1 | rapide (1 min) : devient le modèle principal |
| M7 MLP PyTorch avec embeddings des catégories (1 seed) | 79,6 | 78,5 en moyennant 5 seeds |

**Constats :**
- Le **boosting** (LightGBM, CatBoost, XGBoost) domine nettement : environ 3 kg de mieux que Ridge et les forêts.
  Il capte seul les non-linéarités et les interactions qu'il faut écrire à la main pour Ridge.
- Les trois GBM **plafonnent au même niveau (~78)**, alors que ce sont des implémentations différentes. C'est le
  signe que la limite vient de l'information disponible (le niveau propre de l'athlète n'est pas observé), plus
  que du modèle.
- Le MLP est un peu en dessous, mais il fonctionne de façon très différente des arbres : intéressant pour l'ensemble.

## 8. Choix de la cible et des valeurs aberrantes (28/09)

Comparés avec LightGBM et des features identiques :

| Variante | RMSE CV |
|---|---|
| A : `TotalKg` brut | 78,08 |
| B : `log(TotalKg)` (avec correction du biais de retransformation `exp(p + σ²/2)`) | 78,27 |
| C : points DOTS = `TotalKg × 500 / dénominateur DOTS`, perte pondérée par `(dénominateur/500)²` | 78,02 |
| A sans les 80 lignes aberrantes (BW < 30 kg ou `TotalKg/BW` hors de [1 ; 11,5]) | 78,13 |

- **Cible C** : on prédit un « niveau relatif » débarrassé de l'effet du poids de corps, puis on remultiplie. La
  pondération de la perte rend l'optimisation équivalente à minimiser l'erreur en kg. Score équivalent à A, mais
  les modèles sur A, B et C se trompent un peu différemment, ce qui sert à l'ensemble : **on garde les trois**.
- **Valeurs aberrantes** : les retirer du train n'apporte rien → **on garde tout**. (On ne retire jamais de lignes
  de la validation ni du test.)

## 9. Tuning et seeds (29/09) – `tune.py`

- **Optuna** sur LightGBM : 30 essais évalués sur 3 plis, avec un learning rate élevé (0,08) pour aller vite. Le
  modèle final se réentraîne avec le learning rate normal (0,03) et le nombre d'arbres fixé par early stopping.
  Paramètres retenus dans `outputs/params/lgbm.json`.
- Chaque modèle final est la **moyenne de 5 seeds**, ce qui réduit la variance.

| Expérience | RMSE CV |
|---|---|
| `lgbm_A_tuned` | 77,58 |
| `lgbm_B_tuned` | 77,64 |
| `lgbm_C_tuned` | **77,49** (meilleur modèle seul) |

Gain d'environ 0,5 kg par rapport à LightGBM par défaut, dû à la fois au tuning et à la moyenne des seeds.

## 10. Ensemble (29/09) – `ensemble.py`

Deux méthodes, évaluées en **validation imbriquée** : les poids sont appris sur 4 plis d'OOF et évalués sur le 5ᵉ,
pour ne pas surestimer le gain.

- **Moyenne pondérée** : poids ≥ 0, de somme 1, optimisés sur les OOF → **77,19**.
- **Stacking** : un méta-modèle Ridge reçoit les prédictions OOF de tous les modèles, plus `Sex` et `Equipment`,
  et apprend à les combiner → **77,14**.

*Pourquoi ça marche :* les modèles ne se trompent pas exactement sur les mêmes lignes. Les deux méthodes donnent
presque le même score : le gain vient surtout du mélange de modèles différents, pas de la sophistication du
méta-modèle.

## 11. Analyse des erreurs (29/09) – `analysis.py`

- **Importance** (gain, permutation, SHAP) : les dénominateurs **IPF GL, Wilks et DOTS** dominent (ils résument
  poids de corps × sexe × équipement), puis `BW × homme`, `AgeClass` et `MeetYear`. Les target encodings de
  `Country`, `ParentFederation` et `MeetCountry` apportent peu. Figures dans `outputs/figures/`.
- **Où le modèle se trompe** : chez les hommes (RMSE 84 contre 59 chez les femmes), en Multi-ply (109) et chez les
  plus de 105 kg. Seules 9 lignes ont `Sex = Mx` (RMSE ≈ 236) : catégorie trop rare pour être apprise.

---

## 12. Bilan et pistes testées (19/10)

À ce stade : le stacking est à 77,14 et tous les GBM plafonnent vers 78. Deux pistes pour aller chercher
l'information manquante sur l'athlète ont été testées sur les OOF de `lgbm_C_tuned` (77,49) :

| Piste | RMSE CV | Décision |
|---|---|---|
| **Doublons exacts** : si une ligne est identique à des lignes du train, prédire la moyenne de leur cible | **77,31 (−0,17)** | **intégrée** (étape 16) |
| **Pseudo-athlètes** : regrouper les lignes par (sexe, pays, état, fédération, équipement, année de naissance = `MeetYear − Age`) et encoder le résidu moyen du groupe | 77,46 à 78,1 | **abandonnée** |

- **Doublons** : dans le train, les lignes identiques sur toutes les variables ont un écart-type de la cible de
  seulement **10 kg**, contre 77 d'erreur pour le modèle. Il s'agit très probablement du même athlète à la même
  compétition. Le test en contient un peu plus que la CV (1 % des lignes contre 0,8 %) : gain attendu d'environ
  0,2 sur Kaggle. Intégré à l'étape 16.
- **Pseudo-athlètes** : avec ces variables, les groupes mélangent trop de personnes différentes pour isoler un
  athlète. Aucun gain, piste abandonnée.

## 13. MLP retravaillé (19/10) – `tabnn.py`, `models.py`

*Pourquoi :* le MLP était le seul modèle jamais réglé, et c'est le plus différent des arbres, donc le plus utile
pour l'ensemble.

**Code :** les briques du réseau ont été déplacées dans `tabnn.py` (partagé avec DRAGON). Le refactor reproduit
exactement l'ancien score du pli 0 (79,075).

**Diagnostic (pli 0)** : courbes d'apprentissage du MLP d'origine.

| Époque | RMSE train | RMSE validation |
|---|---|---|
| 0 | 83,4 | 83,0 |
| 8 | 77,1 | 79,6 |
| 18 | 73,3 | 79,1 (meilleure) |
| 26 | 71,3 | 79,5 |

À partir de l'époque 8, le train continue de baisser alors que la validation stagne et oscille de ±0,4 d'une
époque à l'autre. Le réseau **sur-apprend légèrement et apprend de façon instable** : il lui faut plus de
régularisation et un apprentissage plus doux, pas plus de capacité.

**Variantes testées (pli 0, 1 seed ; bruit ≈ ±0,3)** :

| Variante | RMSE pli 0 |
|---|---|
| MLP d'origine (cible A) | 79,08 |
| cible C | 78,89 |
| cible C + PLR σ = 0,03 / 0,1 / 0,5 / 1,5 | 78,47 / **78,18** / 78,46 / 78,70 |
| cible A + PLR σ = 0,5 | 78,67 |
| C + PLR 0,1 + dropout 0,3 | 78,12 |
| C + PLR 0,1 + learning rate 1e-3 | 78,09 |
| C + PLR 0,1 + weight decay 1e-2 | 78,24 |
| C + PLR 0,1 avec dimension 8 au lieu de 16 | 78,52 |
| **C + PLR 0,1 + dropout 0,3 + learning rate 1e-3** | **77,96** |

- **PLR** (*Periodic-Linear-ReLU*, Gorishniy et al., 2022) : chaque variable continue (âge, poids de corps,
  dénominateurs…) est transformée en `ReLU(Linear([cos 2πcx, sin 2πcx]))`, avec des fréquences `c` apprises.
  Le réseau dispose ainsi de fonctions de base riches pour chaque variable, au lieu d'une seule entrée linéaire.
  C'est le changement le plus utile (environ −0,7 sur le pli 0).
- **Cible C** : aide aussi le MLP (la cible est plus homogène une fois l'effet du poids de corps retiré).
- Les réglages de régularisation sont, pris un par un, dans le bruit. Leur combinaison donne le meilleur résultat
  et répond au diagnostic.

Configuration retenue : `outputs/params/mlp.json`. Résultat sur 5 plis × 5 seeds :

| Expérience | RMSE CV | Plis |
|---|---|---|
| `mlp_A_s5` (avant) | 78,50 | 77,95 / 78,75 / 78,13 / 78,74 / 78,95 |
| `mlp_C_plr_s5` (après) | **77,80** | 77,28 / 78,01 / 77,37 / 78,06 / 78,27 |

Amélioration de 0,5 à 0,8 kg **sur chacun des 5 plis** : le gain est réel. Le MLP dépasse LightGBM par défaut
(78,08) et se rapproche du meilleur LightGBM tuné (77,49).

## 14. Recherche d'architecture avec DRAGON (19/10) – `dragon/search.py`

*Pourquoi :* tester une méthode d'AutoDL (recherche automatique d'architecture) pour le MLP, plutôt que de fixer
l'architecture à la main.

- **DRAGON** (*DiRected Acyclic Graphs OptimizatioN*) représente un réseau comme un graphe acyclique dont les nœuds
  sont des opérations PyTorch. Il cherche à la fois les opérations, les connexions et les hyperparamètres.
- **Espace de recherche** : l'entrée (embeddings + PLR, identique au MLP de l'étape 13) et la sortie sont fixes.
  DRAGON cherche le graphe entre les deux (couches denses jusqu'à 512 neurones, normalisation, dropout, identité ;
  combinaison des entrées par somme, produit ou concaténation ; activation de chaque nœud) et le learning rate.
- **Algorithme Mutant-UCB** : chaque architecture est un « bras » de bandit. À chaque itération, il choisit
  l'architecture la plus prometteuse (meilleure erreur + bonus d'incertitude), puis soit il **prolonge son
  entraînement** (4 époques), soit il crée une **mutation** qui hérite de ses poids. Le calcul se concentre ainsi
  sur les bonnes pistes.
- **Protocole** : recherche sur le **pli 0** uniquement (400 évaluations). Les meilleures architectures sont ensuite
  **réentraînées depuis zéro** (l'héritage des poids biaise les scores de la recherche), puis la meilleure est
  évaluée sur les 5 plis avec 5 seeds.
- **Contrainte technique** : DRAGON impose `numpy < 2`, incompatible avec le reste du projet. Il a donc son propre
  environnement (`dragon/pyproject.toml`) et reçoit des données déjà prétraitées (`main.py export-nn`) ; ses
  prédictions sont ensuite journalisées avec `main.py register`.
- **Point de vigilance pour le rapport** : l'architecture étant choisie sur le pli 0, le score de ce pli est
  légèrement optimiste. Il faut regarder les plis 1 à 4 pour juger.

**Déroulement de la recherche** (pli 0, 400 évaluations de 4 époques, 143 architectures, 33 min) :

| | RMSE pli 0 |
|---|---|
| Population initiale aléatoire (20 architectures) : médiane / meilleure | 81,9 / 79,9 |
| Meilleure architecture en fin de recherche (entraînement partiel, poids hérités) | 78,47 |

DRAGON converge vers des réseaux **peu profonds** : 6 des 8 meilleures architectures sont
`entrée → dropout ≈ 0,3 (+ Tanh) → 1 couche dense de 211 neurones (ReLU) → sortie`, avec un learning rate
d'environ 6e-4. Notre MLP réglé à la main en a trois (256 → 128 → 64). C'est cohérent avec le diagnostic de
l'étape 13 : la régularisation compte plus que la profondeur.

**Réentraînement depuis zéro des 5 meilleures (pli 0, 1 seed)** : de 78,05 à 78,35. L'architecture retenue
(idx 77) fait 78,05, contre **77,96 pour le MLP de l'étape 13** dans les mêmes conditions.

**Évaluation complète (5 plis × 5 seeds)** :

| Expérience | RMSE CV | Plis |
|---|---|---|
| `mlp_C_plr_s5` (réglé à la main, étape 13) | **77,80** | 77,28 / 78,01 / 77,37 / 78,06 / 78,27 |
| `dragon_C` (architecture trouvée par DRAGON) | 78,27 | 77,71 / 78,66 / 77,91 / 78,43 / 78,63 |

**Conclusion** : l'architecture de DRAGON est **moins bonne de 0,4 à 0,65 sur chacun des 5 plis**, et n'apporte
rien à l'ensemble (poids nul dans la moyenne pondérée, −0,05 dans le stacking) : elle ressemble trop au MLP réglé
à la main, en moins bien. On ne la retient pas pour la soumission, mais l'expérience est utile pour le rapport :

- elle confirme que **l'architecture n'est pas le facteur limitant** : une recherche automatique de 400
  évaluations ne fait pas mieux qu'un MLP standard bien régularisé, et les entrées (cible C, PLR, étape 13)
  ont eu bien plus d'effet ;
- explications possibles de l'écart : la recherche ne voit que le pli 0, avec des entraînements courts (4 époques)
  qui favorisent les réseaux qui apprennent vite plutôt que ceux qui généralisent le mieux ; et le score de la
  recherche, obtenu avec héritage des poids, ne prédit qu'imparfaitement le score après réentraînement
  (classement des 5 finalistes différent) ;
- pistes si on voulait insister : budget par évaluation plus long, recherche sur plusieurs plis, ou partir de
  l'architecture de l'étape 13 comme population initiale (option `models` de Mutant-UCB).

Fichiers : `dragon/runs/archs.jsonl` (toutes les évaluations), `dragon/runs/final.log`,
`dragon/runs/dragon_C_architecture.txt`.

## 15. Ensemble avec le nouveau MLP (19/10)

| Méthode | Avant (29/09) | Après (19/10) |
|---|---|---|
| Moyenne pondérée | 77,19 | 77,14 |
| Stacking Ridge | 77,14 | **77,06** |

L'ajout de `dragon_C` ne change rien (77,06 dans les deux cas).

## 16. Post-traitement des doublons exacts (19/10) – `postprocess.py`

*Pourquoi :* la piste la plus rentable de l'étape 12 (−0,17 mesuré sur LightGBM).

- **Principe** : si une ligne est identique, sur les 16 variables d'origine, à une ou plusieurs lignes du train, on
  remplace la prédiction du modèle par la moyenne de leur `TotalKg`. Les autres lignes ne changent pas.
- **Validation honnête** : en CV, les correspondances d'une ligne de validation sont cherchées uniquement dans le
  train de son pli (867 lignes remplacées sur 105 000). Pour le test, on utilise tout le train (465 lignes sur
  45 000, soit un peu plus qu'en CV : le gain sur Kaggle devrait être du même ordre, voire un peu plus grand).
- **Est-ce légitime ?** On n'utilise que les données du train, jamais les réponses du test : c'est l'équivalent
  d'un « plus proche voisin » limité aux voisins exactement identiques. En revanche, ce n'est pas de la
  modélisation : on exploite le fait que la même performance apparaît des deux côtés du découpage train/test, ce
  qui ne servirait pas pour un nouvel athlète. À présenter comme tel dans le rapport, en donnant les scores
  avec et sans post-traitement.
- **Pour que le post-traitement s'applique à l'ensemble**, celui-ci sauvegarde désormais ses prédictions OOF
  (issues de la validation imbriquée) et test comme une expérience normale.

| Expérience | Sans | Avec doublons | Gain |
|---|---|---|---|
| `lgbm_C_tuned` | 77,49 | 77,31 | −0,17 |
| `mlp_C_plr_s5` | 77,80 | 77,58 | −0,22 |
| `ensemble_weighted` | 77,13 | 76,94 | −0,19 |
| `ensemble_stack` | 77,06 | **76,89** | −0,17 |

Le gain est le même sur chacun des 5 plis (stacking : 76,39 / 77,14 / 76,41 / 77,18 / 77,31). Commande :
`uv run main.py dedup ensemble_stack` → expérience et soumission `ensemble_stack_dedup`.

Le nouveau MLP reçoit le plus gros poids de la moyenne pondérée (0,26, devant CatBoost, LightGBM B et C tunés),
alors qu'il n'est pas le meilleur modèle seul : c'est sa différence avec les arbres qui compte.

## 17. Première soumission et méthodes du cours (20/10)

**Score public de `ensemble_stack_dedup` : 77,04** (CV : 76,89). L'écart CV / public est faible et cohérent.
Des camarades sont à 76,30.

- **L'écart est-il réel ?** Le cours 5 met en garde contre la « malédiction du vainqueur » : en multipliant les
  soumissions, le meilleur score public devient optimiste, d'environ σ·√(2 log N). Mesuré sur nos OOF : entre deux
  bons modèles notés sur les mêmes lignes publiques, l'écart de score ne fluctue que d'environ 0,06 à 0,10. Même
  avec 20 soumissions, la chance n'expliquerait qu'environ 0,25 de leur avance : **l'essentiel de l'écart est réel**.
- **Méthodes des cours passées en revue** :

| Cours | Méthode | Verdict |
|---|---|---|
| 5 | Optimisation bayésienne des hyperparamètres (Optuna) | à faire : CatBoost et XGBoost jamais tunés, LightGBM sommairement |
| 5 (et 6) | Greedy Ensemble Selection | à tester, gain attendu faible |
| 4 | SVR / ridge à noyau (avec approximation de Nyström) | optionnel, pour la diversité de l'ensemble |
| 1 | k plus proches voisins | **testé, abandonné** : les quasi-doublons (tout identique sauf poids, âge ou année) n'apportent rien en plus des doublons exacts |
| 5 | Bagging | déjà fait (moyenne de seeds) |
| 6 | Modèles de fondation tabulaires | cours pas encore disponible |

## 18. Analyse des features par ablation (20/10)

*Pourquoi :* vérifier si certaines features sont inutiles et en chercher de nouvelles, avant de relancer le tuning.

**Redondances** : pour un arbre, `log_bw`, `bw_pow23` et `age_sq` sont des copies exactes de `BodyweightKg` et
`Age` (transformations croissantes), et les bornes basse / haute / milieu d'`AgeClass` et `BirthYearClass` portent
la même information trois fois. Elles ne servent qu'au Ridge et au MLP. Une douzaine de features ont une
importance par permutation quasi nulle (`te_ParentFederation`, `is_junior`, `is_master`, `wc_limit`, `pre2000`,
indicatrices « Unknown », `Sanctioned`, `is_covid`).

**Ablation** (LightGBM tuné, cible C, 1 seed, 5 plis ; référence 77,61) :

| Changement | RMSE | Lecture |
|---|---|---|
| − indicatrices « Unknown » / bornes de classes / contexte / physio / Wilks + IPF GL | 77,56 à 77,65 | aucun effet : redondantes, mais inoffensives |
| − fréquences | 77,71 | elles aident un peu |
| − target encoding | 77,73 | il aide (+0,13) |
| − features d'âge (distance au pic, âge², junior, master) | 77,94 | **elles comptent** (+0,33) |
| − `State` et `MeetState` (catégorielles) | 78,12 | **elles comptent beaucoup** (+0,51) |
| − `Fed_Equip` (> 1 000 modalités) | 77,53 | **gain**, sur 5 plis sur 5 : sur-apprentissage |
| + année de naissance (`MeetYear − Age`) | 77,50 | **gain** (−0,11), 5/5 |
| + taille de la compétition / catégorie (comptages sans la cible) | 77,49 | **gain** (−0,12), 5/5 |
| + rangs du poids / de l'âge dans la compétition | 77,68 | rien |
| + target encoding de `State`, `WeightClassKg`, `AgeClass` | 77,68 | rien |
| + target encoding de la **compétition** (fédération × année × pays × état, et × sexe × équipement) | 77,41 | **gain** (−0,20), 5/5 |
| **les 4 gains combinés** | **77,18** | **−0,43, sur chacun des 5 plis** |

- Il n'y a pas d'identifiant de compétition dans les données : la combinaison fédération × année × pays × état en
  est une approximation (≈ 10 000 groupes, 34 lignes en médiane). Son target encoding mesure le « niveau » de la
  compétition (un championnat national n'a pas le niveau d'une compétition locale).
- Fait notable : un test préalable (corriger les résidus du stacking par groupe de compétition) ne montrait aucun
  gain. Donnée au modèle comme feature, la même information aide nettement : elle interagit avec les autres
  variables.
- Les 4 changements sont intégrés à `features.py` (`birth_year`, `meet_size`, `div_size`, `wc_size`,
  `te_Meet`, `te_Meet_Div`, et `Fed_Equip` réduite à sa fréquence). Les expériences suivantes portent le
  suffixe `_v2`.
- Note pratique : sur batterie, le Mac se met en veille pendant les longs calculs (un entraînement de 5 s en a
  pris 627). Les calculs longs sont lancés avec `caffeinate -i`.

**Réentraînement avec les features v2 (21/10)** :

| Modèle | Avant | v2 | Écart |
|---|---|---|---|
| LightGBM cible C (5 seeds) | 77,49 | **77,05** | −0,44 |
| LightGBM cible B (5 seeds) | 77,64 | 77,33 | −0,31 |
| MLP PLR cible C (5 seeds) | 77,80 | 77,67 | −0,13 (il ne reçoit pas le target encoding) |
| CatBoost cible C (1 seed, non tuné) | 78,15 (cible A) | 77,66 | −0,49 |

| Ensemble | Avant | v2 |
|---|---|---|
| Moyenne pondérée | 77,14 | 76,73 |
| Stacking | 77,06 | 76,64 |
| **Stacking + doublons** | 76,89 | **76,47** (plis : 76,07 / 76,63 / 76,05 / 76,78 / 76,85) |

LightGBM v2 reçoit le plus gros poids de la moyenne pondérée (0,37), puis LightGBM B v2, CatBoost v2 et le MLP v2.
Score public attendu : autour de 76,6 (même écart CV / public que la première soumission).

Correction annexe : `data.py` contactait Kaggle à chaque lancement, même avec les données en local, et une coupure
réseau a fait planter un entraînement ; il lit désormais les fichiers locaux en priorité.

## 19. Cours 5 : revoir l'ensemble (31/10) – `ensemble.py`

*Pourquoi :* le cours (slides 49-53) met en garde contre un méta-modèle qui reçoit beaucoup de prédictions très
corrélées : il risque de sur-apprendre. Il propose comme remèdes des poids positifs ou une sélection gloutonne
(Greedy Ensemble Selection, Caruana et al.). Notre stacking reçoit 21 colonnes (versions v1 et v2 des mêmes modèles,
corrélation des résidus de 0,95 à 0,99) et sa Ridge a des coefficients libres, dont 9 sont négatifs.

Toutes les variantes ont été évaluées sur les mêmes OOF, en validation imbriquée (poids appris sur 4 plis, évalués sur
le 5ᵉ) :

| Méta-modèle | RMSE CV | Plis |
|---|---|---|
| Moyenne pondérée (poids ≥ 0, somme 1) | 76,73 | 76,28 / 76,87 / 76,35 / 77,02 / 77,15 |
| Sélection gloutonne (20, 40 ou 80 étapes, avec remise) | 76,73 | identique |
| Moindres carrés à poids positifs (NNLS) | 76,73 | 76,28 / 76,87 / 76,34 / 76,99 / 77,14 |
| Sélection gloutonne puis stacking Ridge | 76,73 | |
| **Stacking Ridge à coefficients libres (actuel)** | **76,64** | 76,19 / 76,80 / 76,24 / 76,91 / 77,04 |
| Idem sans les indicatrices `Sex` / `Equipment` | 76,64 | elles ne servent à rien |

- **Le piège annoncé par le cours ne se produit pas ici** : avec 105 000 lignes pour 21 colonnes, le méta-modèle
  ne sur-apprend pas, et c'est même l'ensemble sans contrainte qui gagne (0,09 de mieux, sur chacun des 5 plis). Les
  coefficients négatifs ne sont pas du bruit : en combinant, par exemple, `lgbm_C_v2` (+0,44) et `lgbm_C_tuned`
  (−0,11), le méta-modèle accentue l'apport des features v2. Toutes les méthodes à poids positifs retombent au même
  niveau (76,73), quelle que soit la façon de choisir les poids.
- *Note technique :* `Ridge(positive=True)` de scikit-learn donne 77,00 avec des poids quasi uniformes, car son
  solveur ne converge pas sur des prédictions de l'ordre de 500 ; le vrai NNLS (`LinearRegression(positive=True)`)
  donne 76,73.
- **Contribution marginale** (slide 50 ; `uv run main.py ensemble --contrib`) : hausse de la RMSE du stacking quand
  on retire un modèle.

| Modèle | RMSE seul | Retrait |
|---|---|---|
| `lgbm_C_v2` | 77,05 | +0,079 |
| `cat_C_v2` | 77,66 | +0,031 |
| `ridge_A` | 81,61 | +0,022 |
| `xgb_A` | 78,94 | +0,021 |
| `et_A` | 82,35 | +0,020 |
| `rf_A` | 81,09 | +0,018 |
| `lgbm_B_v2` | 77,33 | +0,011 |
| 14 autres | | entre −0,002 et +0,009 |

  C'est l'illustration exacte de la slide 50 : **Ridge, ExtraTrees et Random Forest sont 4 à 5 kg moins bons seuls,
  mais chacun pèse autant dans l'ensemble que le deuxième meilleur GBM**, parce qu'ils se trompent autrement. À
  l'inverse, les anciens LightGBM (v1) et les anciens MLP n'apportent plus rien. Retirer un à un les 11 modèles les
  moins utiles ne coûte que 0,01 : on garde tout, c'est plus simple.
- **Décomposition d'ambiguïté** (slide 49), pour la moyenne pondérée : RMSE moyenne des membres 77,48, ensemble 76,72.
  Tout le gain de l'ensemble vient de la dispersion des membres autour de leur moyenne (ambiguïté), qui reste faible
  (MSE de 116 sur 6 002) car les modèles sont très corrélés.

**Conséquence pour la suite (point 2)** : les modèles qui se trompent autrement que les GBM (forêts, Ridge, MLP) n'ont
jamais été réentraînés avec les features v2 ni réglés. C'est là qu'il faut chercher du gain, plutôt qu'en améliorant
le meilleur GBM.

## 20. Cours 5 : diversité et hyperparamètres (31/10) – `tune.py`, `main.py train --set`

Tout est sur les features v2, avec la cible C sauf mention contraire. File de calculs : `logs/queue_cours5.sh` et
son journal `logs/queue_cours5.log` (≈ 2 h 30, dont 1 h 15 pour le tuning de CatBoost).

**Hyperparamètres (slides 39-47)** : Optuna (TPE), 40 essais pour LightGBM, 30 pour XGBoost, 20 pour CatBoost,
évalués sur les plis 0 à 2. **Les plis 3 et 4 n'ont jamais servi au tuning** : comparer avant / après sur ces deux
plis donne une mesure du gain qui n'est pas biaisée par la malédiction du vainqueur (une version simple de la
validation imbriquée de la slide 47).

| Modèle | Avant | Après tuning | Écart plis 0-2 (tuning) | Écart plis 3-4 (jamais vus) |
|---|---|---|---|---|
| LightGBM (5 seeds), retuné sur v2 / cible C | 77,05 | 77,04 | −0,018 | **+0,015** |
| XGBoost (5 seeds), tuné pour la 1re fois | 77,69 | **77,29** | −0,37 | **−0,43** |
| CatBoost (1 seed), tuné pour la 1re fois | 77,66 | 77,62 | −0,035 | −0,035 |

- **LightGBM : aucun gain, c'est la malédiction du vainqueur en direct.** Le meilleur essai semble gagner un peu sur
  les plis qui ont servi à le choisir, et perd un peu sur les deux autres. Les paramètres trouvés sont d'ailleurs
  presque ceux de septembre (`num_leaves` 64 contre 60, `colsample_bytree` 0,46 contre 0,45, un peu plus de
  régularisation) : les features v2 n'ont pas déplacé l'optimum. `outputs/params/lgbm.json` contient les nouveaux
  paramètres ; les anciens sont dans `lgbm_v1.json`.
- **XGBoost : vrai gain de 0,4, sur chacun des 5 plis**, plus grand sur les plis jamais vus que sur ceux du tuning.
  Ses paramètres par défaut (profondeur 8, peu de régularisation) étaient loin de l'optimum (profondeur 9,
  `reg_lambda` 26, `colsample_bytree` 0,44). Il passe de 6ᵉ à 3ᵉ meilleur modèle seul.
- **CatBoost : petit gain régulier**, pour un coût élevé (1 h 15 de tuning).

**Diversité (slides 23, 28 et 48-50)** : modèles qui se trompent autrement que les GBM.

| Modèle | RMSE CV | Remarque |
|---|---|---|
| Random Forest v2, tuné (`max_features` 0,14, `min_samples_leaf` 4) | 79,19 | contre 81,09 en v1 non tuné (−1,9 sur chaque pli) |
| ExtraTrees v2, tuné (`max_features` 0,93, `min_samples_leaf` 5) | 79,29 | contre 82,35 (−3,1 sur chaque pli) |
| Ridge v2, cible C / cible A | 81,43 / 81,79 | contre 81,61 |
| LightGBM `extra_trees` (seuils de coupure aléatoires, 3 seeds) | 77,29 | |
| LightGBM `linear_tree` (feuilles linéaires, cible A, 3 seeds) | 77,55 | |

- Les forêts gagnent énormément au tuning et aux features v2, mais restent à 2 kg du boosting : comme l'annonce le
  cours, le bagging réduit la variance mais pas le biais, ce que fait le boosting.
- Forêts optimales très différentes : RF tire peu de variables par coupure (14 %), ExtraTrees presque toutes (93 %).
  ExtraTrees met déjà le hasard dans les seuils ; il n'a pas besoin d'en ajouter dans le choix des variables.

**Ensemble avec les 31 modèles** (validation imbriquée) :

| Ensemble | 21/10 (21 modèles) | 31/10 (31 modèles) |
|---|---|---|
| Moyenne pondérée | 76,73 | 76,69 |
| Stacking | 76,64 | **76,59** |
| Stacking + doublons | 76,47 | **76,43** (plis : 76,01 / 76,58 / 75,99 / 76,73 / 76,83) |

Gain de 0,02 à 0,06 **sur chacun des 5 plis** : réel mais faible. La moyenne pondérée donne désormais le plus gros
poids à XGBoost tuné (0,21), devant LightGBM B v2 (0,18).

- **Limite de la contribution marginale** : avec 31 modèles, aucun ne coûte plus de 0,016 quand on le retire (RF v1,
  ExtraTrees v1, MLP, LightGBM `extra_trees` en tête). Ce n'est pas qu'ils sont inutiles : chacun a un
  remplaçant presque identique dans l'ensemble (RF v1 et RF v2, etc.). Retirer un modèle à la fois ne mesure que
  ce qu'il apporte *en plus de tous les autres* ; c'est le même piège que l'importance par permutation avec des
  variables corrélées (slide 26).
- **Bilan des points 1 à 3 du cours 5** : +0,05 sur le stacking. Le cours se vérifie qualitativement (diversité,
  malédiction du vainqueur, tuning utile seulement là où les paramètres étaient loin de l'optimum), mais
  l'ensemble était déjà proche de son plafond. On reste sur la conclusion de l'étape 11 : la limite vient de
  l'information disponible.
- Sauvegarde : les fichiers de l'ensemble du 21/10 sont dans `outputs/archive_2110/`.

**Score public de `ensemble_stack_dedup` (31/10) : 76,47** (CV : 76,43), 3ᵉ du classement (76,29 et 76,44 devant).
L'écart CV / public passe de 0,15 à 0,04. Correction de `cv.log_experiment` : relancer une expérience efface
maintenant son score public (avant, il était conservé alors que les prédictions avaient changé, et le 77,04 du
19/10 s'affichait à côté de la CV de la version du 31/10).

## 21. Réentraîner sur 100 % du train ? (31/10)

*Pourquoi :* la prédiction test est la moyenne des 5 modèles de pli, chacun entraîné sur 80 % du train. Le cours
(slide 52) réentraîne plutôt chaque modèle une fois sur tout le train.

*Comment le mesurer sans Kaggle :* on reproduit la situation **à l'intérieur de chaque pli** (LightGBM cible C,
1 seed). Sur les 80 % d'entraînement du pli, on compare la moyenne de 4 modèles entraînés chacun sur 3/4 (stratégie
actuelle) à 1 modèle entraîné sur tout (réentraînement, nombre d'arbres = moyenne des early stoppings × facteur).
Les deux sont évalués sur le pli, qu'aucun n'a vu.

| Stratégie (simulée) | RMSE CV |
|---|---|
| 1 modèle sur 60 % du train | 77,76 |
| **Moyenne de 4 modèles sur 60 % (stratégie actuelle)** | **77,15** |
| 1 modèle sur 80 %, nombre d'arbres × 1,0 / 1,1 / 1,25 / 1,4 | 77,25 / 77,24 / 77,22 / 77,22 |

- **Le réentraînement sur tout le train est moins bon** (+0,07 à +0,10). On ne le fait pas.
- Deux effets se font concurrence. Passer de 60 à 80 % des données fait gagner 0,5 à un modèle seul (la courbe
  d'apprentissage n'est pas plate), mais moyenner 4 modèles entraînés sur des sous-échantillons différents fait
  gagner 0,6 (effet bagging, cours 5). Le réentraînement garde le premier effet et perd le second.
- Cela suggère de cumuler les deux avec **plus de plis** : à 10 plis, chaque modèle voit 90 % des données et on en
  moyenne 10. Même simulation, avec 8 sous-plis contre 4 dans chaque pli :

| Stratégie (simulée) | RMSE CV | Écart par pli |
|---|---|---|
| Moyenne de 4 modèles sur 60 % (≈ 5 plis) | 77,15 | |
| **Moyenne de 8 modèles sur 70 % (≈ 10 plis)** | **77,07** | −0,11 / −0,07 / −0,18 / −0,06 / +0,01 |

  Gain d'environ 0,08 sur un LightGBM à 1 seed. Avec 5 seeds, une partie de l'effet de moyenne est déjà acquise :
  le gain réel sera probablement plus petit.

## 22. Analyse approfondie des données : reste-t-il du signal ? (31/10)

*Pourquoi :* on est 3ᵉ à 0,18 du premier ; vérifier si un meilleur prétraitement peut encore apporter quelque chose.

*Outil de mesure :* pour chaque piste, on regarde si les résidus OOF du stacking (76,59) varient selon le groupe
proposé. La correction par groupe est apprise sur les autres plis (sinon, même des groupes tirés au hasard semblent
utiles : +0,5 avec des tranches d'`id`).

| Piste | Constat | Gain |
|---|---|---|
| `id` | aléatoire : aucune corrélation avec la cible, l'année ou la fédération, même entre `id` voisins | aucun |
| Précision du poids de corps (entier, 1 ou 2 décimales, conversion depuis les livres) | erreur plus forte pour les poids entiers (80,5) que pour 2 décimales (74,2), mais résidu moyen nul | −0,003 |
| Âge en x,5 (estimé à partir de l'année de naissance) | 48 % des lignes, de 74 % avant 2000 à 41 % après 2020 ; erreur 73 contre 80 pour l'âge exact, résidu moyen nul | −0,002 |
| Cohérence âge / `AgeClass` / `BirthYearClass` | 10 % hors classe, uniquement des âges x,5 en limite de classe (17,5 → 18-19) : arrondi, pas erreur ; 100 % cohérent avec `BirthYearClass` | – |
| Système de catégories de poids (IPF depuis 2011 / traditionnel / Texas lycées) | capté par `Federation` et `WeightClassKg` | −0,003 |
| Poids au-dessus de la limite de catégorie | 107 lignes, presque toutes des conversions livres → kg (99,79 pour 99,7) | – |
| Quasi-doublons (identiques sauf une variable) | quelques dizaines de lignes, sauf poids / âge / année (corrélation 0,79, moins bon que le modèle) | – |
| Groupes plus fins que la compétition (× catégorie, × âge, × `Tested`), cohortes, pseudo-athlètes, `State` × sexe, fédération × catégorie… | aucun signal restant dans les résidus | tous < 0 |
| « Compétition » reconstituée | 98 % des lignes test la retrouvent dans le train ; mais elle regroupe plusieurs vraies compétitions (`USPA` Californie 2019 = 249 lignes), et rien ne permet de les séparer | – |

**Test global** : un LightGBM entraîné à prédire le résidu du stacking à partir de toutes les features (plus la
prédiction elle-même), croisé sur les plis, s'arrête à **1 ou 2 arbres** et ne corrige rien (76,586 → 76,595).

**Conclusion** : avec ces 16 variables, l'ensemble a extrait tout le signal accessible. Les résidus ne dépendent plus
d'aucune variable ni combinaison de variables. Il reste ~76 kg de bruit propre à chaque athlète (son niveau
individuel, absent des données). Un prétraitement plus poussé ne fera pas mieux ; le seul levier mesuré est le
nombre de plis (section 21).

## 23. Passage à 10 plis (31/10 – 01/11) – `cv.make_subfolds`, `main.py train --folds 10`

*Pourquoi :* la section 21 montre que plus de plis cumule deux gains : chaque modèle apprend sur 90 % du train au
lieu de 80 %, et la prédiction test moyenne 10 modèles au lieu de 5.

*Comment, sans casser la comparaison :* chaque pli de référence est coupé en deux sous-plis (même stratification),
emboîtés (`make_subfolds`, test `test_subfolds_are_nested_in_the_reference_folds`). L'entraînement se fait sur les
10 sous-plis, mais les scores sont calculés sur les 5 plis de référence, et le stacking garde sa validation imbriquée
sur ces 5 plis. Les OOF restent donc honnêtes : la prédiction d'une ligne du pli k vient d'un modèle qui n'a pas vu
cette ligne, et le méta-modèle évalué sur le pli k est appris sur les 4 autres.

On réentraîne les 10 modèles qui comptent le plus dans l'ensemble, avec leurs hyperparamètres actuels
(`logs/queue_k10.sh`, environ 2 h 30) :

| Modèle | 5 plis | 10 plis | Écart |
|---|---|---|---|
| `lgbm_C` (tuné, 5 seeds) | 77,04 | **76,86** | −0,19 |
| `lgbm_C_xt` (extra_trees, 3 seeds) | 77,29 | 77,09 | −0,20 |
| `xgb_C` (tuné, 5 seeds) | 77,29 | 77,09 | −0,21 |
| `lgbm_B` (5 seeds) | 77,33 | 77,16 | −0,17 |
| `mlp_C_plr` (5 seeds) | 77,67 | 77,40 | −0,27 |
| `cat_C` (tuné) | 77,62 | 77,48 | −0,14 |
| `hgb_raw` | 78,98 | 78,86 | −0,11 |
| `rf_C` | 79,19 | 78,96 | −0,23 |
| `et_C` | 79,29 | 79,07 | −0,22 |
| `ridge_C` | 81,43 | 81,38 | −0,06 |

Tous les modèles gagnent, la Ridge presque pas : un modèle linéaire n'a pas besoin de plus de données. Les modèles
flexibles (MLP, forêts) gagnent le plus, signe que leur courbe d'apprentissage monte encore à 80 % du train.

Ensembles (stacking Ridge, validation imbriquée, avant post-traitement des doublons) :

| Ensemble | Stacking | Plis |
|---|---|---|
| Ces 10 modèles à 5 plis | 76,638 | 76,18 / 76,77 / 76,25 / 76,92 / 77,08 |
| **Ces 10 modèles à 10 plis** | **76,453** | 75,98 / 76,63 / 76,06 / 76,67 / 76,92 |
| 31 modèles à 5 plis (ensemble du 31/10) | 76,586 | 76,13 / 76,75 / 76,17 / 76,86 / 77,01 |
| **10 modèles à 10 plis + 21 anciens non remplacés** | **76,400** | 75,95 / 76,58 / 75,99 / 76,61 / 76,86 |
| Les 41 modèles | 76,398 | identique au précédent |

- Le gain (−0,19) est le même sur les 5 plis : il est significatif.
- Garder aussi les versions 5 plis des modèles remplacés n'apporte rien (76,398 contre 76,400). L'ensemble retenu
  a donc 31 modèles (`logs/ensemble_k10_models.txt`).
- Après le post-traitement des doublons : **76,242** (plis 75,83 / 76,41 / 75,80 / 76,48 / 76,68). Avant, on
  avait 76,428.
- Ce que la CV mesure : seulement l'effet « 90 % des données au lieu de 80 % », car chaque ligne OOF est prédite
  par un seul modèle de pli.

**Score public : 76,440** (2ᵉ du classement), contre 76,470 pour la version à 5 plis. Le gain public (−0,03) est
**6 fois plus petit que le gain en CV (−0,19)** :

| Version | CV | Public | Écart public − CV |
|---|---|---|---|
| 31/10, 5 plis | 76,428 | 76,470 | +0,04 |
| 01/11, 10 plis (section 23) | 76,242 | 76,440 | +0,20 |
| 01/11, 10 plis + anciens modèles (section 24) | 76,226 | 76,443 | +0,22 |

- **Ce n'est pas le hasard du découpage public.** En tirant au hasard 13 500 ou 22 500 lignes des OOF, l'écart
  entre les deux ensembles vaut −0,20 ± 0,06 (ou ± 0,04) ; un écart de −0,03 ou moins n'arrive que dans 0,15 % des
  tirages.
- **Explication : les deux effets de la section 21 ne s'additionnent pas, ils se recouvrent.** Une ligne OOF est
  prédite par un seul modèle ; une ligne test par la moyenne des modèles de pli. Avoir plus de données réduit
  surtout la variance d'un modèle seul. Or la moyenne de 5 modèles sur le test réduisait déjà cette variance. La
  CV compare un modèle seul sur 80 % à un modèle seul sur 90 % et voit tout le gain (−0,19). Le test compare une
  moyenne de 5 modèles à une moyenne de 10, et ne voit que ce qui reste (−0,03). C'est cohérent avec la simulation
  de la section 21 : −0,08 pour un LightGBM à 1 seed, « probablement plus petit » avec 5 seeds.
- **Conséquence pour le rapport et la suite :** la CV d'un modèle à 10 plis est optimiste d'environ 0,15 par
  rapport à celle d'un modèle à 5 plis. On ne peut plus comparer directement les deux. Les comparaisons restent
  valables entre modèles qui ont le même nombre de plis (section 24, par exemple). L'erreur était dans
  l'hypothèse initiale de cette section : l'effet de la moyenne sur le test ne s'ajoute pas au gain mesuré en CV,
  il en couvre la plus grande partie.
- L'early stopping sur des sous-plis de 10 % n'y est pour rien : les forêts, qui n'en ont pas, gagnent autant en
  CV que les GBM.
- L'ensemble du 31/10 (public 76,47) est archivé dans `outputs/archive_3110/`, avec sa ligne du journal des
  expériences.

## 24. Anciens modèles v1 à 10 plis (01/11) – `main.py train --features v1 --params`

*Pourquoi :* l'ensemble garde 21 anciens modèles entraînés à 5 plis. D'après la contribution marginale (section
23), 5 d'entre eux comptent encore : `rf_A` (0,016), `et_A` (0,011), `lgbm_B_tuned` (0,007), `mlp_C_plr_s5` (0,004),
`ridge_A` (0,002). Les autres pèsent 0,002 ou moins, sont des variantes de modèles déjà passés à 10 plis
(`lgbm_C_v2`, `cat_C_v2`), ou viennent de DRAGON, qui n'exporte que 5 plis (`dragon_C`).

*Difficulté :* ces modèles datent d'avant les features v2. Les relancer avec le code actuel les rapprocherait des
autres modèles. Or leur utilité tient justement à leur différence : `ridge_A` (v1) contribue 0,002, `ridge_A_v2`
contribue 0. Il faut donc les réentraîner **à l'identique**, seul le nombre de plis changeant.

*Code :*
- `features.use_v1_features()` : `Fed_Equip` redevient catégorielle, les 4 features v2 (`birth_year`, `meet_size`,
  `div_size`, `wc_size`) et le target encoding de la compétition disparaissent. On retrouve 56 colonnes.
- `main.py train --features v1` et `--params lgbm_v1` (fichier d'hyperparamètres au choix).
- **Vérification** : à 5 plis, `ridge_A` et `rf_A` sont reproduits au millième près sur chacun des 5 plis, et
  `lgbm_B_tuned` (avec `lgbm_v1.json`) sur le pli 0 (77,160).

File : `logs/queue_k10_v1.sh` (sans argument : modèles rapides ; avec `long` : LightGBM et MLP).

| Modèle | 5 plis | 10 plis | Écart |
|---|---|---|---|
| `ridge_A` | 81,60 | 81,54 | −0,07 |
| `rf_A` (paramètres par défaut) | 81,09 | 80,94 | −0,15 |
| `et_A` (paramètres par défaut) | 82,35 | 82,08 | −0,27 |

| `lgbm_B_tuned` (`lgbm_v1.json`, 5 seeds) | 77,64 | 77,47 | −0,16 |
| `mlp_C_plr_s5` (5 seeds) | 77,80 | 77,53 | −0,26 |

On retrouve les gains de la section 23 : environ −0,2, plus pour le MLP et ExtraTrees, presque rien pour la Ridge.

| Ensemble (31 modèles) | Stacking | Plis | + doublons |
|---|---|---|---|
| Section 23 (10 modèles à 10 plis) | 76,400 | 75,95 / 76,58 / 75,99 / 76,61 / 76,86 | 76,242 |
| + `ridge_A`, `rf_A`, `et_A` à 10 plis | 76,393 | 75,94 / 76,56 / 75,99 / 76,61 / 76,86 | |
| **+ `lgbm_B_tuned`, `mlp_C_plr_s5` à 10 plis** | **76,382** | 75,95 / 76,57 / 75,97 / 76,58 / 76,84 | **76,226** |

- **Score public : 76,443** (contre 76,440 pour la version de la section 23) : aucune différence visible, comme
  prévu. D'après la section 23, le gain réel sur le test est d'ailleurs encore plus petit que les −0,016 de la CV.
- Gain final de −0,016 après les doublons (plis 0 / −0,017 / −0,016 / −0,028 / −0,019) : il ne se dégrade sur
  aucun pli, mais il est **8 fois plus petit que le passage à 10 plis des modèles principaux**. C'est l'ordre de
  grandeur attendu : ces modèles pèsent peu dans l'ensemble.
- Les forêts v1 (`rf_A_k10`, `et_A_k10`) sont désormais les deux modèles dont le retrait coûte le plus (0,024 et
  0,021), devant XGBoost : ce sont les plus différents des GBM.
- Ensemble précédent (soumis le 01/11, 2ᵉ du classement) archivé dans `outputs/archive_0111/`. Liste des
  modèles retenus : `logs/ensemble_k10_models.txt`.

## 25. Mesures complémentaires sur le prétraitement (01/11) – `logs/analyse_0111/`

*Pourquoi :* second regard, indépendant, sur la question de la section 22 (« reste-t-il du signal ? »), avec d'autres
outils de mesure. Les conclusions se recoupent ; cette section ne garde que ce qui est nouveau. Chiffres LightGBM à
1 seed, cible C, paramètres tunés, **5 plis** (référence 77,20) ; analyses de résidus sur l'OOF de `ensemble_stack`
du 31/10 (76,59). Scripts et journaux dans `logs/analyse_0111/`.

### 25.1 Quantifier la structure de groupe restante

Pour chaque variable, variance des moyennes de résidu par groupe (pondérée par la taille) contre sa valeur attendue
si le résidu était du bruit pur (σ²/n par groupe) :

| Groupe | Variance observée | Attendue si bruit |
|---|---|---|
| compétition (fédé × année × pays × état), 9 955 groupes | 582 | 556 |
| compétition × sexe × équipement, 20 626 groupes | 1 207 | 1 152 |
| `Federation` | 12,0 | 22,3 |
| `MeetState` / `Country` / `State` | 8,6 / 6,8 / 4,5 | 13,4 / 8,3 / 6,8 |
| `WeightClassKg` / `AgeClass` | 4,7 / 0,5 | 8,2 / 1,1 |

L'excès sur la compétition (26 kg²) est récupérable à 13 % au mieux après lissage (groupes de 5 lignes en médiane),
soit 0,02 de RMSE. Même conclusion que le test global de la section 22, par une autre voie.

### 25.2 Pourquoi l'athlète n'est pas identifiable

Appariement des lignes qui *pourraient* être le même athlète : même sexe, équipement, pays, état, `Tested` (et
fédération selon la variante), année de naissance à ±1 an, poids de corps à ±2,5 kg.

| Appariement (dans le train) | Lignes appariées | Corrélation des résidus | Écart-type de la différence de total |
|---|---|---|---|
| + même fédération, même état et année de compétition, poids à ±0,5 kg | 8 % | 0,06 | 93 kg |
| + même fédération, autre année | 53 % | 0,007 | 122 kg |
| sans fédération, autre année | 69 % | 0,007 | 125 kg |
| sans fédération, toute année, poids à ±1 kg | 59 % | 0,013 | 120 kg |
| *doublons exacts (étape 12)* | *0,5 %* | | *10 kg* |

Les lignes appariées sont **des personnes différentes** (un même athlète aurait des totaux proches). L'explication
est la taille du jeu : 150 000 lignes alors qu'OpenPowerlifting en compte plus de 3 millions, soit un tirage
d'environ **5 % des inscriptions**. La probabilité que deux compétitions d'un même athlète tombent toutes les deux
dans l'échantillon est quasi nulle. Les seuls doublons qui survivent sont les inscriptions multi-divisions d'une
même compétition (une performance copiée dans deux divisions : 73 % des groupes de doublons ont un écart-type nul),
et leur fréquence (0,5 %) est cohérente avec ce tirage. **Les pistes « pseudo-athlètes » (étape 12) et kNN
(étape 17) ne pouvaient pas marcher, par construction du jeu** : c'est l'argument à donner dans le rapport.

### 25.3 Forme des résidus

- Asymétrie −0,25, kurtosis 1,5. **5 % des lignes (résidu > 150 kg) portent 36 % de l'erreur quadratique**, 0,5 %
  (> 250 kg) en portent 9 % : des athlètes exceptionnellement forts ou faibles pour leur profil.
- Dans des cellules fines (sexe × équipement × tranche de poids), l'asymétrie du total est nulle : la **perte L2
  est la bonne** ; une perte robuste (Huber) viserait la médiane et ferait moins bien en RMSE.
- Pas de mélange d'épreuves (ratio total / poids unimodal) ; les totaux très bas sont des enfants de 5 à 12 ans. Les
  16 lignes à plus de 200 kg de poids de corps (résidu moyen −79, écart-type 221) sont en partie des erreurs de
  saisie ; les corriger toutes vaudrait 0,05 au plus. Résidu moyen nul par segment, sauf `Mx` (9 lignes, −60).

### 25.4 Ablations du target encoding lui-même (LightGBM, référence 77,20)

| Variante | RMSE CV | Plis |
|---|---|---|
| référence (lissage 20) | 77,20 | 76,66 / 77,31 / 76,80 / 77,58 / 77,66 |
| lissage 5 | 77,16 | 4 plis sur 5 un peu mieux, écarts de 0,00 à 0,07 |
| lissage 50 | 77,26 | |
| + composites fédé × année, fédé × sexe × équipement, compétition × catégorie de poids | 77,33 | |
| + composite compétition × `Tested` (16 % des groupes mélangent testé / non testé) | 77,19 | 2 plis mieux, 3 moins bien |
| TE « en deux étapes » : résidu d'un modèle de base (`hgb_raw`) au lieu de la moyenne sexe × équipement | 77,27 | |

Rien de significatif : le lissage de 20 est bon, les raffinements ajoutent du bruit plus que du signal. Un
méta-modèle **non linéaire** (LightGBM sur les 30 prédictions OOF + sexe, poids, équipement, âge, année, `Tested`)
fait 76,84 contre 76,59 pour le Ridge, et un LightGBM sur les résidus du Ridge 76,593 : le stacking linéaire est au
plafond. Courbe d'apprentissage retrouvée indépendamment (pli 0 : 78,07 / 77,37 / 76,66 pour 42 000 / 63 000 /
84 000 lignes), et moyenne de 4 sous-plis (76,82) contre 1 modèle sur tout (76,72 à 76,78) : même conclusion que
la section 21.

### 25.5 Les modèles privés de l'information « compétition »

Le MLP ne reçoit pas le target encoding et CatBoost ne reçoit pas la clé de compétition (seulement `meet_size`).
Ce sont les deux seuls modèles de l'ensemble sans l'information qui a valu −0,20 à LightGBM (étape 18).

| Modèle (1 seed, 5 plis) | Sans | Avec | Plis |
|---|---|---|---|
| MLP PLR cible C + les 7 colonnes `te_*` en entrée numérique (PLR) | 78,39 | **78,06** | **mieux sur 5 plis sur 5** (−0,27 à −0,47) |
| CatBoost tuné + `Meet` et `Meet_Div` en catégorielles natives (encodage ordonné) | 77,62 | **77,42** | **mieux sur 5 plis sur 5** (−0,07 à −0,28) |

Les deux gains sont du même ordre que celui des features v2 pour LightGBM (étape 18) : la même information,
donnée à deux modèles qui ne l'avaient pas. Pour CatBoost, l'encodage ordonné natif joue le rôle de notre target
encoding, sans fuite. Options ajoutées (sans elles, rien ne change) : `main.py train --model mlp ... --set te=true`
(`mlp_fn` appelle `_with_te` avant `nn_arrays`) et `main.py train --model cat ... --set meet_cats=true` (`cat_fn`
construit `Meet` et `Meet_Div` et les déclare catégorielles).

**Réentraînement à 10 plis et nouvel ensemble (01/11 soir)** :

| Modèle (10 plis) | Sans | Avec l'info compétition | Écart par pli |
|---|---|---|---|
| `mlp_C_plr_te_k10` (5 seeds, `--set te=true`) | 77,40 | **77,18** | −0,13 à −0,30, 5 plis sur 5 (40 min) |
| `cat_C_meet_k10` (`--set meet_cats=true`) | 77,48 | **77,33** | −0,02 à −0,27, 5 plis sur 5 (30 min) |

| Ensemble | 31 modèles (étape 24) | 33 modèles (+ les deux nouveaux) | Plis |
|---|---|---|---|
| Moyenne pondérée | 76,516 | 76,506 | poids 0,10 (MLP + TE) et 0,13 (CatBoost + clé) |
| Stacking Ridge | 76,381 | **76,369** | |
| Stacking + doublons | 76,226 | **76,213** | 75,82 / 76,37 / 75,78 / 76,45 / 76,65 (mieux sur 5 plis sur 5, −0,003 à −0,021) |

Gain de −0,013 sur l'ensemble pour −0,21 et −0,15 sur les modèles seuls : les autres modèles avaient déjà
l'information, seule la diversité compte, comme à l'étape 24. L'ensemble à 31 modèles est archivé dans
`outputs/archive_0111_31modeles/` ; liste des 33 modèles dans `logs/ensemble_k10_v2_models.txt`, journal
`logs/ensemble_k10_v2.log`. Soumission : `ensemble_stack_dedup.csv` (score public à saisir avec `main.py score`).

### 25.6 Faut-il 33 modèles ? Ablation de l'ensemble (01/11) – **à mentionner dans le rapport**

*Pourquoi :* vérifier qu'aucun des 33 modèles ne nuit, et savoir combien comptent vraiment. Deux mesures, sur les
mêmes OOF, en validation imbriquée (Ridge α = 10, poids appris sur 4 plis, évalués sur le 5ᵉ) ; ensemble complet :
76,369.

**Retrait un par un (leave-one-out)** : les retraits les plus coûteux sont `rf_A_k10` (+0,025), `et_A_k10`
(+0,023), `xgb_C_k10` (+0,016), `lgbm_C_xt_k10` (+0,015), `mlp_C_plr_s5_k10` (+0,010). Sept modèles peuvent être
retirés sans rien changer ; le meilleur retrait gagne 0,001, c'est du bruit. **Aucun modèle ne nuit** : avec
105 000 lignes, le Ridge ne sur-apprend pas 33 colonnes (confirme l'étape 19). Limite de cette mesure, déjà vue à
l'étape 20 : la plupart des modèles ont un quasi-jumeau dans l'ensemble (`lgbm_C` et `lgbm_C_v2`, les deux MLP…) ;
retirer l'un ne coûte rien parce que l'autre prend le relais. C'est le piège de l'importance par permutation avec
des variables corrélées (slide 26) : le leave-one-out mesure la redondance d'un modèle, pas sa pertinence.

**Sélection gloutonne** (on ajoute à chaque étape le modèle qui améliore le plus le stacking) :

| Modèles | RMSE CV |
|---|---|
| 1 : `lgbm_C_k10` | 76,86 |
| 2 : + `mlp_C_plr_k10` | 76,65 |
| 3 : + `cat_C_meet_k10` | 76,58 |
| 4 : + `xgb_C_k10` | 76,53 |
| 5 : + `ridge_A_k10` | 76,49 |
| 6 : + `lgbm_B_k10` | 76,47 |
| 10 : + `et_C_k10`, `mlp_C_plr_s5_k10`, `lgbm_C_xt_k10`, `rf_A_k10` | 76,43 |
| 33 | 76,37 |

- **6 modèles font 80 % du gain de l'ensemble**, un par famille : LightGBM, MLP, CatBoost, XGBoost, Ridge, LightGBM
  cible log. Les 27 autres ajoutent 0,10 à eux tous, dont 0,06 pour les 23 derniers.
- Le nouveau CatBoost (clé de compétition) entre en 3ᵉ position, devant XGBoost, alors qu'il est 5ᵉ modèle seul :
  il apporte quelque chose de différent, pas seulement un bon score.
- La sélection est faite sur les mêmes OOF que l'évaluation : les petits sous-ensembles sont légèrement optimistes
  (un stacking à 10 modèles vaut plutôt 76,44 que 76,43).

**Décision** : on soumet avec les 33 (le gain est gratuit et rien ne nuit).

**Un stacking plus petit ferait-il mieux ? (02/11)** Question posée après la sélection gloutonne : un modèle
pourrait-il « brider » les autres, de sorte qu'un sous-ensemble batte les 33 ? Vérifications, toujours en
validation imbriquée sur les mêmes OOF :

- *Élimination arrière* (on retire à chaque étape le modèle dont le retrait aide le plus, tant que le gain dépasse
  0,0005) : s'arrête à 23 modèles et 76,365, contre 76,369 pour les 33. Gain 0,004 = bruit de sélection. Retirés,
  dans l'ordre : `mlp_A`, `xgb_C_v2_default`, `lgbm_C`, `ridge_C_k10`, `ridge_A_v2`, `lgbm_A`, `rf_C_k10`,
  `lgbm_A_v2_lin`, `cat_C_v2`, `lgbm_A_tuned` (tous ont un jumeau dans la liste).
- *Coût des 33 coefficients* : stacking sans imbrication (méta-modèle ajusté et évalué sur les mêmes lignes) 76,335,
  imbriqué 76,369 : l'erreur d'estimation du méta-modèle vaut 0,03 au total, pour 33 coefficients sur 105 000 lignes.
- *Poids ≥ 0 de somme 1* (SLSQP, `ensemble_weighted`) : 33 modèles 76,504, 6 modèles 76,509 ; Ridge libre sur les
  6 : 76,469.

Lecture : avec un Ridge à coefficients libres, « retirer un modèle » est un cas particulier (coefficient zéro) que le
méta-modèle considère déjà ; ajouter un modèle ne peut dégrader que par l'erreur d'estimation, et l'élimination
arrière montre qu'elle ne dépasse pas 0,004 ici. Un modèle n'a pas besoin que les autres se taisent pour
« s'exprimer » : son coefficient est fixé en tenant compte de ce que les autres apportent déjà. L'intuition du
« bridage » est juste pour les combineurs contraints (poids ≥ 0, sélection gloutonne) : là, 27 modèles sur 33
n'apportent rien. Mais ces combineurs sont 0,13 moins bons que le Ridge libre, parce que le gain est dans les
coefficients négatifs, c'est-à-dire des contrastes entre modèles corrélés (`lgbm_C_k10` +0,21 contre `lgbm_C_xt_k10`
−0,22, `rf_A_k10` −0,17, `dragon_C` −0,10).

Réserve réelle, que la CV ne peut pas mesurer : ces contrastes sont appris sur les OOF (chaque ligne prédite par
**un** modèle de pli) et appliqués au test (chaque ligne = moyenne de **10** modèles de pli, plus lisse). Si une
partie d'un contraste capte du bruit propre à un modèle plutôt qu'une différence systématique, elle transfère mal ;
l'écart public − CV passé de +0,04 à +0,20 avec les 10 plis (étape 23) est compatible avec ce mécanisme. Indice dans
le même sens : l'écart RMS entre le stacking à 33 et le stacking à 6 est de 4,4 kg sur les OOF mais de 3,6 kg sur le
test.

**Diagnostic par soumission** (fichiers prêts ; aucun n'écrase la version à 33 modèles) :

| Soumission | Combineur | CV (après dedup) | Écart RMS test avec les 33 | Public |
|---|---|---|---|---|
| `ensemble_stack_dedup.csv` | Ridge libre, 33 modèles | 76,213 | – | **76,426** (01/11) |
| `ensemble_stack6_dedup.csv` | Ridge libre, 6 modèles de la sélection gloutonne | 76,290 | 3,6 kg | **76,500** (02/11) |
| `ensemble_weighted_dedup.csv` | poids ≥ 0 de somme 1, 33 modèles | 76,326 | 4,4 kg | **76,574** (02/11) |

**Verdict (02/11)** : l'ordre de la CV est conservé en public et les écarts se transfèrent presque exactement.
Écart public − CV : +0,213 pour les 33, +0,210 pour les 6, +0,248 pour les poids positifs. L'avantage des 33 sur
les 6 vaut 0,077 en CV et 0,074 en public ; celui du Ridge libre sur les poids positifs vaut 0,11 en CV et 0,15 en
public. **Les contrastes à coefficients négatifs transfèrent au test** : l'hypothèse du « bridage » est écartée, et
la réserve sur le passage « un modèle de pli » → « moyenne de 10 modèles de pli » ne se matérialise pas (le
combineur contraint transfère même un peu moins bien, dans le bruit). L'écart de +0,21 est le même pour tous les
combineurs : c'est l'optimisme de la CV à 10 plis (étape 23), pas un effet du stacking. Décision confirmée : la
soumission finale reste le stacking libre à 33 modèles. Bruit d'une comparaison publique : ±0,06 à 0,10
(étape 17), les trois écarts mesurés ici le dépassent ou l'atteignent. Script : `logs/analyse_0111/stack6.py` ; `ensemble_stack6`,
`ensemble_stack6_dedup` et `ensemble_weighted_dedup` (recalculé sur les 33) sont dans `experiments.csv`.

### 25.7 Peut-on rattraper les 0,13 de la tête ? (02/11)

Classement public au 02/11 : 76,298 (1ᵉʳ, une seule soumission le 17/10), **76,426 (nous)**, 76,444 (3ᵉ), puis 76,89.

**Bruit d'une comparaison entre deux équipes.** Entre deux de nos propres soumissions (très corrélées), l'écart
public a un bruit de ±0,04 (1 σ, calculé sur les OOF pour un public de 13 500 à 22 500 lignes). Mais entre deux
pipelines *différents* (proxy : deux de nos modèles de familles différentes et de score voisin, `xgb_C_k10` contre
`mlp_C_plr_te_k10`), ce bruit vaut **±0,12 à ±0,17**. L'écart de 0,13 avec le 1ᵉʳ est donc de l'ordre d'un σ, et
notre avance de 0,018 sur le 3ᵉ est un pile ou face. Le classement privé (autres lignes) peut réordonner les trois.

**Le tuning des 6 modèles clés ne peut pas rattraper 0,13.** Cinq des six sont déjà tunés par Optuna (étape 20) ;
le re-tuning de LightGBM y a donné 0 (malédiction du vainqueur mesurée sur les plis jamais vus), XGBoost 0,4 parce
que ses défauts étaient loin de l'optimum, CatBoost 0,035. Et un gain sur un modèle seul ne passe qu'en partie dans
l'ensemble : −0,22 sur le MLP et −0,15 sur CatBoost (étape 25.5) ont donné −0,012 sur le stacking. Il faudrait
1 à 2 kg sur un modèle seul pour gagner 0,13 sur l'ensemble ; aucun tuning ne fait ça.

Ce qui reste mesurable, avec l'ordre de grandeur attendu sur l'ensemble : CatBoost à 5 seeds au lieu d'une
(≈ −0,15 seul, −0,01 ensemble, 2 h 30 à 10 plis) ; recherche d'architecture du MLP, largeur / profondeur / lr /
weight decay / epochs, seul modèle non tuné systématiquement (−0,1 à −0,3 seul, −0,01 à −0,03 ensemble). Total
plausible : 0,02 à 0,05 en CV. **Fait pour le MLP (étape 25.8) : 0.** La vraie limite reste celle de l'étape 22 : les résidus n'ont plus de structure, et
la courbe d'apprentissage (étape 25.3) dit que c'est la quantité d'information par athlète qui manque.

### 25.8 Tuning Optuna du MLP (02/11, cours 6) – `main.py tune --model mlp` – **à mentionner dans le rapport**

*Pourquoi :* le MLP est le seul modèle clé jamais tuné systématiquement (étape 25.7). *Comment :* même protocole
qu'à l'étape 20 (40 essais TPE, plis 0-2, 1 seed, cible C, avec target encoding `te=true`), plis 3-4 jamais vus
pendant la recherche. Espace : largeur 128-768, 2 à 4 couches, rétrécissement ou non, dropout 0-0,5, lr 3e-4 à
5e-3, weight decay 1e-6 à 1e-2, batch 512/1024/2048, PLR (sigma 0,01-1, k et d dans {8, 16, 32}). La configuration
courante est évaluée en premier (essai 0) comme référence. 1 h 20 de calcul ; journal `logs/analyse_0111/tune_mlp.log`.

| | Plis 0-2 (tuning) | Plis 3-4 (jamais vus) |
|---|---|---|
| Configuration courante (256-128-64, dropout 0,3, lr 1e-3, PLR sigma 0,1) | 77,864 | 78,349 |
| Meilleur essai (175×3, dropout 0,48, lr 4,5e-4, batch 512, PLR k = d = 8) | **77,758** | 78,336 |
| Écart | −0,106 | **−0,013** |

**Le gain de 0,11 sur les plis de tuning disparaît sur les plis jamais vus : 0,013, à une seed.** Même constat que
pour LightGBM à l'étape 20. On peut cette fois le quantifier avec la slide 12 du cours 6 :

- Bruit d'un essai : la configuration courante relancée avec 3 autres seeds donne 77,697 / 77,872 / 77,796
  (`noise_mlp.py`), soit **σ = 0,08** sur la moyenne de 3 plis, et une vraie valeur de 77,81 : l'essai 0 (seed 42)
  était un tirage défavorable.
- Optimisme attendu du minimum de N = 40 essais : au plus σ·√(2 ln N) = **0,22**. L'écart observé de 0,11 entre
  l'essai 0 et le meilleur essai est entièrement dans cette marge, et le meilleur essai n'est qu'à 0,05 (0,6 σ)
  sous la vraie valeur de la configuration courante.
- Les 10 meilleurs essais se ressemblent (dropout 0,4-0,5, lr 4e-4 à 9e-4, batch 512, pas de rétrécissement) et
  tiennent dans 77,76-77,84 : l'optimiseur a trouvé un plateau, pas un optimum plus bas.

**Décision** : pas de réentraînement (gain attendu sur l'ensemble ≈ 0,001, pour 1 h de calcul et une
configuration deux fois plus lente). `outputs/params/mlp.json` reste l'ancien réglage ; le réglage tuné est gardé
dans `mlp_tuned.json`. Bilan du tuning sur ce projet, pour le rapport : XGBoost −0,4 (défauts loin de l'optimum),
CatBoost −0,035, LightGBM 0, MLP 0. Le tuning rapporte quand on part de loin, et rien sinon ; et sans plis
réservés, on aurait cru gagner 0,11.

**Pour le rapport** : présenter cette courbe avec la contribution marginale (`main.py ensemble --contrib`) et
expliquer pourquoi les deux mesures ne disent pas la même chose (redondance contre pertinence). C'est l'illustration
exacte de la slide 50 du cours 5 : ce qui compte est la **diversité** (un modèle par famille, Ridge et forêts
inclus malgré leurs 4 à 5 kg de retard seuls), pas le nombre de modèles ni leur score individuel. Script :
`logs/analyse_0111/ablation_ensemble.py`.

*Note technique :* un script qui entraîne LightGBM puis un réseau PyTorch dans le **même processus** s'est figé
21 h sans consommer de CPU (deux bibliothèques OpenMP chargées en même temps, problème connu sur Mac). Garder un
modèle par processus, comme le fait `main.py train`.

## 26. Veille 2024-2026 et méthodes récentes (03/11) – `experiments/`

### Veille bibliographique

*Pourquoi :* toutes les méthodes « classiques » ont été épuisées (étapes 1 à 25). Les résidus n'ont plus de
structure (étapes 22 et 25) et l'athlète n'est pas identifiable : aucune méthode ne contourne ça. Une nouvelle
méthode ne peut gagner qu'en **diversité dans l'ensemble**. D'après nos mesures (−0,2 seul ≈ −0,01 au stacking), on
attendait 0,02 à 0,10 pour une famille vraiment nouvelle, presque rien pour une variante d'un modèle existant.

**Ce que disent les papiers :**
- **TabArena** (NeurIPS 2025, benchmark de référence) : les GBDT restent très forts, les réseaux profonds les
  rattrapent quand on les ensemble, les modèles de fondation dominent sur les petits jeux, et les ensembles entre
  familles de modèles font le meilleur score.
- Depuis 2026, les **modèles de fondation tabulaires** (apprentissage en contexte, sans descente de gradient sur
  nos données) passent à l'échelle : TabICLv2 (ICML 2026), TabPFN-2.5 / 3 / 3.5, Mitra-v2. Leurs classements
  viennent surtout de leurs propres auteurs.
- **TabPrep** (juin 2026) : entre 10 000 et 100 000 lignes, LightGBM et TabM restent souvent devant TabPFN-2.5.

**Pistes retenues, par priorité a priori :**

| Priorité | Méthode | Pourquoi | Résultat |
|---|---|---|---|
| 1 | TabICLv2 (Qu, Holzmüller, Varoquaux, Le Morvan) | seule famille vraiment nouvelle, licence permissive, tient 100 000 lignes ; sujet du cours 6 | −0,058 au stacking |
| 2 | TabM (Gorishniy et al., ICLR 2025) | meilleur réseau des benchmarks (32 sous-réseaux à poids partagés) ; gain attendu faible car nos 5 seeds font déjà une partie du travail | meilleur réseau du projet |
| 3 | RealMLP (Holzmüller et al., NeurIPS 2024) | prétraitement différent du nôtre, réglages par défaut appris sur 118 jeux | petit gain, résidus les moins corrélés |
| 4 | Perte HL-Gauss (Imani & White 2018 ; Farebrother et al., ICML 2024) | change la façon dont le réseau se trompe ; aucune preuve publiée sur du tabulaire | rien |
| 5 | TabPrep (Tschalzev, Erickson et al., 2026) | priorité basse : les étapes 18 et 25.4 avaient testé l'équivalent à la main sans gain | **meilleur gain des cinq** |

Non testés : TabPFN-3 (licence recherche uniquement, régresseur limité à 100 000 lignes) et Mitra-v2 (Apache-2.0,
via AutoGluon) restent des candidats pour un deuxième modèle de fondation.

**Écartés :** plus proches voisins (TabR, ModernNCA : aucun voisin utile, étape 25.2), données synthétiques
(TabDDPM : aucune information nouvelle), KAN (résultats mitigés en tabulaire), nouvelle recherche d'architecture
(DRAGON a montré que l'architecture ne limite pas), données OpenPowerlifting externes (contiennent sans doute les
réponses du test : fuite).

**Leçon pour le rapport :** le classement a priori s'est trompé sur TabPrep. Ce qui avait échoué à la main, c'étaient
les composites *target-encodés* et les rangs *dans la compétition* ; TabPrep apporte des statistiques *sans cible*
sur des groupes larges (poids relatif à la catégorie de poids, âge relatif à l'équipement), qu'on n'avait pas
essayées.

Sources : TabArena (arXiv 2506.16791) ; TabICLv2 (arXiv 2602.11139), TabICL (arXiv 2502.05564,
github.com/soda-inria/tabicl) ; TabPFN-2.5 (arXiv 2511.08667), TabPFN-3 (arXiv 2605.13986), TabPFN-3.5
(arXiv 2609.17895) ; Mitra-v2 (arXiv 2609.04540) ; TabM (arXiv 2410.24210) ; RealMLP (arXiv 2407.04491, pytabkit) ;
TabPrep (arXiv 2606.02384) ; Stop Regressing (arXiv 2403.03950), HL-Gauss (arXiv 1806.04613), Investigating the
Histogram Loss (arXiv 2402.13425).

### Évaluation hors projet (5 plis)

**Question :** les méthodes tabulaires publiées depuis 2024 apportent-elles quelque chose à un ensemble déjà très
travaillé ? Tout est évalué **hors du code du projet**, dans `experiments/` (projet uv séparé, `common.py` réutilise
`cv.prepare`, les 5 plis de référence, la cible C et le TE résiduel appris dans le pli). Critère principal : le gain
quand on ajoute le modèle aux 33 du stacking (base 76,3699, `common.evaluate`). Ces modèles sont à 5 plis contre 10
pour les 33 : le gain est une **borne basse**.

**Témoin :** des modèles existants à 5 plis hors des 33 (`mlp_C_plr_v2`, `lgbm_C_v2_tuned`, `xgb_C_v2`,
`cat_C_v2_tuned`, LightGBM de référence) apportent entre −0,0004 et +0,0014 : un gain de −0,02 ou plus n'est pas un
artefact du découpage, c'est de la diversité.

| Méthode (papier) | RMSE seul (5 plis) | Δ stacking | Plis | Verdict |
|---|---|---|---|---|
| TabICLv2 (Qu et al. 2025-26, modèle de fondation, GPU T4 Kaggle) | 77,761 | −0,050 | 4/5 | à garder |
| TabM (Gorishniy et al., ICLR 2025, BatchEnsemble k = 32) | 77,178 | −0,027 | 5/5 | à garder |
| RealMLP (Holzmüller et al., NeurIPS 2024, défauts TD) | 77,905 | −0,017 | 5/5 | à garder |
| HL-Gauss (Imani & White 2018 ; Farebrother et al. 2024) sur le MLP | 78,221 | +0,001 | – | inutile |
| TabPrep, GroupBy larges + OAFE, LightGBM 3 seeds (2025) | 76,948 | −0,071 | 5/5 | à garder |
| **Les quatre gardés ensemble** | – | **−0,124 → 76,246** | 5/5 | |

Pli par pli, les quatre ensemble font −0,123/−0,154/−0,107/−0,146/−0,091.

- **TabICLv2** (`experiments/tabicl/`, Kaggle sans internet, wheels + checkpoint dans un dataset privé) : environ
  9 min par pli sur T4, 8,7 Go de mémoire au pic. La cible C fait mieux que A (77,19 contre 77,54 au pli 0), et A
  n'ajoute rien quand C est déjà là. Les colonnes à beaucoup de modalités sont retirées (le modèle les traite mal), le
  TE résiduel du pli est ajouté.
- **TabM** : PiecewiseLinearEmbeddings + embeddings des catégories, 32 têtes. C'est le **meilleur réseau du projet**
  (77,18 à 5 plis contre 78,07 pour notre MLP à 1 seed), environ 14 min par pli sur MPS.
- **RealMLP** : pytabkit n'accepte pas de poids, mais la cible C sans poids reste meilleure que A (77,33 contre
  77,85 au pli 0). Ses résidus sont les moins corrélés aux modèles clés (0,967 à 0,978).
- **HL-Gauss** (perte histogramme, K = 200) : moins bon que le MLP sur les 5 plis, aucune diversité. Moyenner MLP et
  HL-Gauss fait moins bien que moyenner deux seeds du MLP. Changer la perte ne change pas les erreurs.
- **TabPrep** (`experiments/tabprep/`, générateurs d'AutoGluon réimplémentés) : seuls les **GroupBy relatifs sur
  les catégorielles larges** aident (−0,086 pour LightGBM seul sur 3 seeds, 14 couples pli × seed sur 15). Les
  colonnes utiles : position du poids de corps dans `WeightClassKg` (ratio, rang, écart), rang de l'âge par
  équipement et sexe × équipement. RSFC et CatPrep dégradent : leurs colonnes target-encodées font sur-apprendre,
  comme les composites TE des étapes 18 et 25.4. Les GroupBy relatifs à la compétition dégradent (groupes de
  5 lignes, comme les rangs de l'étape 18). OAFE sur les numériques v2 n'apporte rien seul : le feature engineering
  manuel contient déjà les bonnes transformations.

Rapports détaillés : `experiments/{tabicl,tabm,realmlp,hlgauss,tabprep}/REPORT.md` (sauf TabICL, ci-dessus).
Évaluation conjointe : `cd experiments && PYTHONPATH=. uv run python joint_eval.py tabprep_lgbm tabicl_C realmlp tabm`.

**TabICLv2 à 10 plis** (`experiments/tabicl/kernel10/`, `assemble10.py`, environ 19 min par sous-pli sur T4 avec
repli sur disque ; deux workers en parallèle saturaient la RAM de l'hôte) : 77,761 → **77,605** seul, Δ stacking
−0,050 → **−0,058** (5/5 plis : −0,088/−0,085/−0,006/−0,099/−0,013). Avec TabPrep, RealMLP et TabM (ceux-ci encore
à 5 plis) : **76,2336 (−0,136)**, 5/5 plis.

### 26.1 Intégration à 10 plis et soumission (03/11) – `logs/integration_0311/`, `logs/ensemble_k10_v3*`

Code ajouté au projet : `tabprep.py` (GroupBy relatifs + OAFE, appelé par `lgbm_fn` avec `--set tabprep=true`),
`tabm_net.py` + `models.tabm_fn`, `models.realmlp_fn` (dépendances `tabm`, `pytabkit`, `rtdl-num-embeddings`).
`lgbm_fn` accepte aussi `n_jobs`. Contrôle : LightGBM + TabPrep intégré redonne exactement le pli 0 de
l'expérience (76,723). TabICLv2 (Kaggle) est journalisé avec `main.py register tabicl_C_k10 --model tabicl`.

| Modèle (10 plis) | Commande | RMSE seul | Temps |
|---|---|---|---|
| `lgbm_C_tabprep_k10` (3 seeds) | `train --model lgbm --target C --folds 10 --seeds 3 --set tabprep=true n_jobs=4` | **76,733** (`lgbm_C_k10`, 5 seeds : 76,858) | 44 min |
| `tabm_C_k10` | `train --model tabm --target C --folds 10` | **76,985** (meilleur réseau ; MLP + TE : 77,18) | 2 h 34 (MPS) |
| `realmlp_C_k10` | `train --model realmlp --target C --folds 10` | 77,735 | 2 h 08 (MPS partagé) |
| `tabicl_C_k10` | Kaggle T4, `experiments/tabicl/kernel10/` | 77,604 | 3 h 07 (T4) |

| Ensemble | 33 modèles | **37 modèles** |
|---|---|---|
| Moyenne pondérée | 76,506 | 76,375 (poids : TabPrep 0,29, TabM 0,17, TabICL 0,10) |
| Stacking Ridge | 76,369 | **76,214** |
| Stacking + doublons | 76,213 | **76,075** (75,71 / 76,18 / 75,69 / 76,25 / 76,54, mieux sur 5 plis sur 5) |
| **Public** | 76,426 | **76,247** (−0,179 ; 1er du classement public, devant 76,298) |

Le gain public est même plus grand que le gain CV : ces modèles apportent une diversité réelle, pas un ajustement
aux plis. Dans le stacking, les quatre nouveaux modèles reçoivent les plus gros coefficients (TabPrep 0,38, TabICL
0,20, TabM 0,18). L'ensemble à 33 modèles est archivé dans `outputs/archive_0311_33modeles/` ; liste des 37 dans
`logs/ensemble_k10_v3_models.txt`.

### 26.2 Méta-modèle non linéaire ou contextuel ? (03/11) – `experiments/meta/meta.py`, `results.jsonl`

*Pourquoi :* la Ridge donne à chaque modèle le même poids pour tous les athlètes. Un méta-modèle qui dépend du
contexte (sexe × équipement, âge, poids de corps) pourrait exploiter le fait qu'un modèle est meilleur sur un segment.
Mêmes OOF des 37 modèles, même validation imbriquée ; référence Ridge α = 10 : **76,2144**.

| Méta-modèle | RMSE | Écart |
|---|---|---|
| FWLS (Sill et al. 2009) : prédictions × indicatrices sexe × équipement, α = 10 → 10⁶ | 76,384 → 76,240 | +0,17 → +0,025 |
| FWLS : prédictions × âge et poids de corps standardisés, α = 10 → 10⁶ | 76,232 → 76,218 | +0,017 → +0,004 |
| LightGBM sur le résidu d'une Ridge honnête (Ridge imbriquée dans les 4 plis), + contexte | 76,217 | +0,002 (s'arrête à 10 arbres) |
| Idem + écarts des 5 meilleurs modèles à la Ridge | 76,217 | +0,002 |
| LightGBM direct sur [37 prédictions, contexte] | 76,470 | +0,255 |
| Ridge α = 0,1 / 10³ / 10⁵ | 76,214 / 76,216 / 76,212 | dans le bruit |
| Ridge en points DOTS (pondérée ou non) | 76,215 / 76,216 | dans le bruit |

**Conclusion : la Ridge est déjà optimale.** Le LightGBM sur le résidu s'arrête presque immédiatement : il ne reste
aucune structure dépendant du contexte dans l'erreur de l'ensemble, et chaque degré de liberté ajouté au méta-modèle
coûte plus qu'il ne rapporte. Le LightGBM direct découpe des prédictions continues en paliers et perd 0,25. Pour
gagner encore, il faut de nouveaux modèles qui se trompent autrement, pas un meilleur combinateur.

### 26.3 Le pipeline du top 1 devient le code principal (03/11)

- `ensemble_models.txt` (racine) : les 37 modèles soumis, liste par défaut de `main.py ensemble` (`--all` pour
  tout le journal). `main.py ensemble` puis `main.py dedup ensemble_stack` redonnent la soumission publique 76,247
  **au bit près** (vérifié par empreinte md5).
- `tabicl/` (racine, comme `dragon/`) : `export.py` (entrées des 10 sous-plis : redonne exactement les fichiers
  envoyés sur Kaggle), `kernel/` (script et métadonnées du kernel GPU), `assemble.py` (→ `outputs/oof|preds`).
- TabM et RealMLP choisissent seuls leur matériel (`device=auto` : mps, sinon cuda, sinon cpu), pour le binôme.
- Deux tests ajoutés : les variables TabPrep d'une ligne de validation ne dépendent que du train du pli, et chaque
  modèle de l'ensemble a ses prédictions. 22 tests passent.
- README : commandes des nouveaux modèles, section « Reproduire la meilleure soumission », résultats à jour.
- `experiments/` reste l'archive des évaluations (rapports `*/REPORT.md`), hors du pipeline.

*Note technique Kaggle :* les kernels n'ont ni internet (paquets `tabicl` / `einops` et poids fournis en *wheels*
dans un dataset privé, `pip install --no-index`) ni GPU tant que le compte n'est pas vérifié par téléphone ; après
vérification, il a fallu un nouveau kernel (l'ancien restait sur une image CPU). Quota : 30 h de GPU par semaine.

*Note technique :* quatre agents et un test TabICL en parallèle ont fait planter le Mac (16 Go). Au plus deux
calculs lourds à la fois. `ps -o rss` ne voit pas la mémoire MPS, il faut `footprint <pid>`.

---

## Points de vigilance pour le rapport

- L'early stopping est fait sur le pli de validation : l'OOF est très légèrement optimiste. L'écart avec le
  leaderboard public permettra de le mesurer.
- Les écarts de quelques dixièmes ne sont pas significatifs (±0,4 entre plis). Pour conclure, s'appuyer sur la
  cohérence pli par pli, comme pour le MLP de l'étape 13.
- Le bruit irréductible est élevé : sans identifiant d'athlète, aucun modèle ne descend nettement sous 77.
- L'ensemble à 33 modèles n'est pas un empilement aveugle : 6 modèles (un par famille) font 80 % du gain, aucun
  ne nuit, et le leave-one-out sous-estime les modèles qui ont un jumeau dans l'ensemble (étape 25.6).
- Un nouveau modèle évalué à 5 plis dans un ensemble à 10 plis est désavantagé : son gain est une borne basse
  (étape 26 : TabICL −0,050 → −0,058, TabPrep −0,071 → −0,085 en passant à 10 plis).
- Étape 26.1 : le gain public (−0,179) a dépassé le gain CV (−0,138). Les nouveaux modèles apportent une diversité
  réelle, pas un ajustement aux plis.
- Ne pas comparer la CV de modèles entraînés avec des nombres de plis différents (étape 23) : le passage de 5 à
  10 plis gagne 0,19 en CV mais seulement 0,03 en public, car l'OOF mesure un modèle seul et le test une moyenne de
  modèles.
