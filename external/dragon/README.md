# DRAGON : recherche d'architecture du MLP (journal, étape 14)

[DRAGON](https://dragon-tutorial.readthedocs.io/en/latest/) (algorithme Mutant-UCB) cherche le graphe du réseau entre
l'entrée (embeddings + PLR, `powerlifting.nets.mlp`) et la sortie. Il impose `numpy < 2`, incompatible avec le reste
du projet : il a son propre environnement uv et reçoit des matrices déjà prétraitées.

Résultat : `dragon_C` (78,27 en CV) est moins bon que le MLP réglé à la main (77,80) sur les 5 plis, mais il fait
partie des 37 modèles de l'ensemble soumis.

Depuis la racine du projet :

```bash
uv run main.py export-nn --target C                                                  # matrices des 5 plis -> outputs/nn_data/
uv run --project external/dragon python external/dragon/search.py search --iters 400 # recherche sur le pli 0 (~40 min)
uv run --project external/dragon python external/dragon/search.py final --seeds 5    # réentraîne les meilleures, 5 plis
uv run main.py register dragon_C --seeds 5                                           # journal + soumission
```

Les évaluations de la recherche sont écrites dans `external/dragon/runs/` (non versionné) ; le journal de la recherche
du 19/10 est dans `archive/logs/dragon_2026-10-19.log`.
