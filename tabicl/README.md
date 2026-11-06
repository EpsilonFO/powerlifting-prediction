# TabICLv2 sur GPU Kaggle (journal, étape 26)

TabICLv2 (Qu et al., paquet `tabicl` 2.2.0) est un modèle de fondation tabulaire : il prédit par apprentissage en
contexte, sans descente de gradient sur nos données. Il a besoin d'un GPU (environ 19 min par sous-pli sur une T4,
8,7 Go). Il tourne donc dans un kernel Kaggle privé, sans internet : le paquet, `einops` et les poids sont fournis
dans un dataset privé.

| Fichier | Rôle |
|---|---|
| `export.py` | entrées des 10 sous-plis (features + target encoding appris dans le pli) -> `external/tabicl/data/` |
| `kernel/run.py` | script exécuté sur Kaggle (un sous-pli à la fois, repli sur disque) |
| `assemble.py` | rassemble les sorties du kernel -> `outputs/oof|preds/tabicl_C_k10.npy` |

Depuis la racine du projet :

```bash
uv run python external/tabicl/export.py
uvx pip download tabicl==2.2.0 einops==0.8.2 --no-deps -d external/tabicl/data
uvx --from huggingface_hub hf download jingang/TabICL tabicl-regressor-v2-20260212.ckpt --local-dir external/tabicl/data
cp external/tabicl/dataset-metadata.json external/tabicl/data/ && uvx --with six kaggle datasets create -p external/tabicl/data
uvx --with six kaggle kernels push -p external/tabicl/kernel     # GPU T4 (compte Kaggle vérifié par téléphone), ~3 h
uvx --with six kaggle kernels output felxoll/powerlifting-tabicl-gpu -p external/tabicl/out
uv run python external/tabicl/assemble.py external/tabicl/out
uv run main.py register tabicl_C_k10 --model tabicl --target C
```

Adapter l'identifiant `felxoll/...` dans `dataset-metadata.json` et `kernel/kernel-metadata.json`. Un seul processus
TabICL à la fois : à deux, le tampon de sortie (environ 17 Go chacun) dépasse la RAM d'un Mac de 16 Go.

Notes Kaggle : les kernels n'ont pas internet (paquets fournis en *wheels*, `pip install --no-index`) et pas de GPU
tant que le compte n'est pas vérifié par téléphone. Quota : 30 h de GPU par semaine.
