#!/bin/zsh
# Attend la fin du pli 0, puis 5 plis de la config retenue (cible C + TE, TD, MPS), puis évaluation séparée.
cd "$(dirname $0)/.."
while pgrep -f queue_fold0.sh >/dev/null; do sleep 20; done
uv run python realmlp/run.py --name C_te --target C --te 1 --set '{"device": "mps"}' --save realmlp
uv run python realmlp/evaluate.py results/realmlp_oof.npy > realmlp/logs/eval_realmlp.json
echo "EVAL DONE"
