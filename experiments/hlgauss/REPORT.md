# Perte HL-Gauss pour le MLP (03/11)

**Verdict : ça ne vaut pas le coup.** Seule, la tête HL-Gauss fait 0,13 à 0,15 kg de plus que le MLP de
référence. Ses erreurs ne sont pas plus différentes de celles du MLP actuel que celles d'une simple autre graine,
et elle n'apporte rien au stacking à 33 modèles (+0,0008, du bruit ; RealMLP : −0,0165).

## Implémentation (`hl.py`)

- Même réseau que `models.mlp_fn` : `tabnn.TabularInput` (embeddings + PLR σ = 0,1) puis `tabnn.TabularMLP`
  256-128-64 (BatchNorm, SiLU, dropout 0,3), `te=True`, paramètres `outputs/params/mlp.json`. Seule la dernière
  couche `Linear(64, 1)` est remplacée par `Linear(64, K)`.
- Support : K intervalles réguliers sur la cible C standardisée, de [min − 0,5 ; max + 0,5] (≈ [−4,7 ; 5,7]
  écarts-types ; largeur 0,10 pour K = 100).
- Cible : masse de N(y, σ²) sur chaque intervalle, renormalisée sur le support (fonction erf, comme dans
  l'annexe de Farebrother et al.). σ = 0,75 × largeur par défaut.
- Perte : cross-entropy pondérée par w = (dénominateur/500)² normalisé, comme la MSE du projet. Option « mse » :
  CE + λ·w·(E − y)² sur l'espérance (λ = 1).
- Prédiction : espérance de l'histogramme, softmax(logits) · centres.
- Boucle `fit_hl` : copie de `tabnn.fit` (AdamW lr 1e-3, wd 1e-4, batch 1024, ReduceLROnPlateau, early stopping
  patience 8 sur la RMSE en kg de l'espérance). Rien n'a été modifié dans le projet. Pour la référence, on appelle
  `models.mlp_fn` après avoir neutralisé `torch.set_num_threads(8)`.
- Ressources : CPU, 2 threads, environ 650 Mo (`footprint`). 25 à 35 s par entraînement (≈ 30 époques) ; 5 plis
  en 3 min, autant que le MLP de référence.

## Pli 0 (1 graine, puis 3 graines pour lisser le bruit de ±0,1)

| Config | RMSE pli 0, graine 0 | Moyenne sur 3 graines | corr. résidus avec la réf. | Moyenne (ref@a + X@b)/2 |
|---|---|---|---|---|
| **MLP de référence** | **77,608** | **77,563 ± 0,04** | 0,9809 (entre deux graines) | **77,192** (deux graines ≠) |
| K = 50, σ = 0,75 l | 77,838 | 77,776 ± 0,10 | 0,9782 | 77,245 |
| K = 100, σ = 0,75 l | 77,915 | 77,756 ± 0,11 | 0,9786 | 77,240 |
| K = 200, σ = 0,75 l | 77,774 | **77,687 ± 0,08** | 0,9794 | 77,221 |
| K = 100, σ = 0,30 (≈ 3 l) | 77,986 | – | 0,9767 | 77,319 |
| K = 100 + MSE (λ = 1) | 77,713 | 77,706 ± 0,07 | 0,9800 | 77,243 |
| K = 200 + MSE (λ = 1) | – | 77,725 ± 0,09 | 0,9800 | 77,253 |

(l = largeur d'un intervalle. Scripts : `hl.py fold0`, `diag_seeds.py` ; journaux `fold0.log`,
`fold0_seeds.log`, `diag_seeds.log`, `diag0.log`.)

- Toutes les variantes sont moins bonnes que la référence, de 0,12 à 0,42. K compte peu (200 ≥ 50 ≈ 100, dans le
  bruit). Un σ plus large (lissage fort) dégrade. Le terme MSE réduit un peu l'écart mais rapproche aussi les
  erreurs de celles de la référence.
- **Test de diversité décisif** : la corrélation des résidus entre HL-Gauss et la référence (0,979-0,980) est à
  peine plus basse que celle entre deux graines de la référence (0,981). La moyenne d'une référence et d'un
  HL-Gauss (77,22-77,25) est **moins bonne** que la moyenne de deux graines de la référence (77,19). HL-Gauss
  n'apporte donc rien de plus qu'une graine supplémentaire, et le fait en moins bien.
- Le stacking local sur le seul pli 0 (`diag0.log`) est trop bruité pour départager les variantes : y ajouter la
  référence elle-même coûte +0,005.

## 5 plis (config retenue : K = 200, σ = 0,75 l, CE pure ; graine 0)

| Pli | 0 | 1 | 2 | 3 | 4 | CV |
|---|---|---|---|---|---|---|
| MLP de référence (relancé, même graine) | 77,608 | 78,403 | 77,657 | 78,230 | 78,438 | **78,068** |
| **HL-Gauss K = 200** (`hlgauss`) | 77,774 | 78,478 | 77,728 | 78,565 | 78,557 | 78,221 (+0,15) |
| HL-Gauss K = 100 + MSE (contrôle) | 77,713 | 78,336 | 77,824 | 78,468 | 78,673 | 78,203 (+0,14) |

Le MLP HL-Gauss est moins bon sur 5 plis sur 5. Le relancement de la référence retrouve les chiffres du
coordinateur (77,607 / 78,32 / 77,664 / 78,247 / 78,452) à 0,01-0,08 près : le multithread n'est pas
bit-à-bit déterministe.

**Corrélation des résidus** (OOF 5 plis) :

| Modèle | `mlp_C_plr_te_k10` | `lgbm_C_k10` | `cat_C_meet_k10` | `xgb_C_k10` |
|---|---|---|---|---|
| MLP de référence, 1 graine, 5 plis | 0,985 | 0,974 | 0,967 | 0,969 |
| HL-Gauss K = 200 | 0,981 | 0,972 | 0,966 | 0,968 |
| HL-Gauss K = 100 + MSE | 0,982 | 0,973 | 0,967 | 0,968 |

**Stacking Ridge à 33 modèles** (`common.evaluate`, référence 76,3699) :

| Ajout | Stacking | Delta | Delta par pli |
|---|---|---|---|
| HL-Gauss K = 200 | 76,3707 | **+0,0008** | −0,0004 / +0,0028 / −0,0013 / +0,0044 / −0,0015 |
| HL-Gauss K = 100 + MSE | 76,3705 | +0,0006 | +0,0001 / +0,0012 / +0,0011 / +0,0005 / +0,0003 |
| MLP de référence (contrôle) | 76,3701 | +0,0002 | ≈ 0 sur chaque pli |
| HL-Gauss K = 200 + référence | 76,3709 | +0,0010 | |
| *pour comparaison : RealMLP* | | *−0,0165* | |

Aucun gain : delta positif, de l'ordre du bruit, mitigé selon les plis. Pour mémoire, le MLP 5 plis est désavantagé
d'environ 0,15-0,25 face aux modèles à 10 plis (cf. `common.evaluate`), mais le contrôle (même MLP en tête
scalaire) subit le même handicap, et RealMLP dans les mêmes conditions apporte −0,0165.

## Pourquoi ça ne marche pas ici

- Les avantages rapportés par Farebrother et al. viennent de cibles non stationnaires et de très grands réseaux
  (RL, Transformers), où la MSE optimise mal. Ici le problème est dominé par le bruit : la RMSE résiduelle
  (≈ 76-78 kg) est du même ordre que l'écart-type de la cible C (≈ 80 points, soit environ 80 kg). La distribution
  conditionnelle couvre donc environ un écart-type, soit des dizaines d'intervalles, et elle est symétrique
  (journal, 25.3).
- Dans ce régime, la CE ajuste un histogramme entier, une estimation de densité, alors que seule l'espérance
  compte. C'est statistiquement moins efficace que la MSE, qui estime directement la moyenne (perte optimale
  pour la RMSE, journal 25.3) : d'où le +0,15.
- L'optimisation est différente, mais le réseau, les features et le signal sont les mêmes. Les erreurs restent
  dominées par le bruit propre à l'athlète, que personne ne voit. La diversité obtenue n'est pas plus grande que
  celle d'une graine.

## Fichiers

- `hl.py` : implémentation et lanceur (`fold0 [config@graine …]`, `diag0`, `full <config>`).
- `diag_seeds.py` : diagnostic multi-graines du pli 0.
- `preds/` : prédictions du pli 0 par config et graine ; OOF et test 5 plis de la référence
  (`hlgauss_ref_mlp_te_s1_*`) et de K100_mse.
- `../results/hlgauss_{oof,test}.npy` : HL-Gauss K = 200, 5 plis (via `common.save`).
- Journaux : `fold0.log`, `fold0_seeds.log`, `diag0.log`, `diag_seeds.log`, `full_K200.log`, `full_ref.log`,
  `full_K100_mse.log`.
