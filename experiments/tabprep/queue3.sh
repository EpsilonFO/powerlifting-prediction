#!/bin/zsh
cd "$(dirname "$0")/.."
for s in 0 1 2; do uv run python tabprep/run.py gb_wc gb_nowc --seed $s > tabprep/logs/q3_s$s.log 2>&1; done
echo done > tabprep/logs/queue3.done
