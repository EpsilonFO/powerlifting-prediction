#!/bin/zsh
cd "$(dirname "$0")/.."
uv run python tabprep/run.py gb_wide+oafe > tabprep/logs/q4_s0.log 2>&1
for s in 1 2; do uv run python tabprep/run.py gb_wide gb_wide+oafe --seed $s > tabprep/logs/q4_s$s.log 2>&1; done
echo done > tabprep/logs/queue4.done
