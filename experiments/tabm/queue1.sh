#!/bin/zsh
# Un seul processus Python lourd à la fois (séquentiel).
cd "$(dirname "$0")/.."
R() { uv run python -u -W ignore tabm/run.py "$@" }
R ple_onehot_d512 '{"device":"mps","threads":2}' 0
R mlp_ref '{}' 0 --mlp
