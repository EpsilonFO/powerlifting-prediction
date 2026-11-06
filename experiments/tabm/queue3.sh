#!/bin/zsh
# Meilleure config du pli 0 (ple_emb) sur les 5 plis, puis évaluation (processus séparé).
cd "$(dirname "$0")/.."
uv run python -u -W ignore tabm/run.py tabm '{"device":"mps","cat":"emb"}' --save && uv run python -u tabm/evaluate.py tabm
