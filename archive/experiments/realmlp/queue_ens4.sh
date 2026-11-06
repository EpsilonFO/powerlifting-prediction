#!/bin/zsh
# 5 plis, cible C + TE, n_ens = 4 (4 réseaux vectorisés par pli, époque choisie sur leur moyenne), puis évaluation.
cd "$(dirname $0)/.."
uv run python realmlp/run.py --name C_te_ens4 --target C --te 1 --set '{"device": "mps", "n_ens": 4}' --save realmlp_ens4
uv run python realmlp/evaluate.py results/realmlp_ens4_oof.npy > realmlp/logs/eval_realmlp_ens4.json
echo "EVAL DONE"
