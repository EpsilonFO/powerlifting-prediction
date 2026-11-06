#!/bin/zsh
# Pli 0 seul, réglages TD, MPS. Une variante par processus.
cd "$(dirname $0)/.."
M='"device": "mps"'
uv run python realmlp/run.py --name f0_A_te   --target A --te 1 --folds 0 --set "{$M}"
uv run python realmlp/run.py --name f0_C_te   --target C --te 1 --folds 0 --set "{$M}"
uv run python realmlp/run.py --name f0_A_note --target A --te 0 --folds 0 --set "{$M}"
