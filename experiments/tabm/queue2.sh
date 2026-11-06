#!/bin/zsh
# Variantes sur le pli 0, séquentiel (attend la fin de queue1).
cd "$(dirname "$0")/.."
while pgrep -f "tabm/queue1.sh" > /dev/null; do sleep 20; done
R() { uv run python -u -W ignore tabm/run.py "$@" }
R ple_emb '{"device":"mps","cat":"emb"}' 0
R plr_onehot '{"device":"mps","emb":"periodic","sigma":0.1,"n_freq":16,"d_emb":16}' 0
R ple_onehot_lr1e3 '{"device":"mps","lr":0.001}' 0
