#!/bin/zsh
cd "$(dirname "$0")/.."

uv run python tabprep/run.py base gb gbT gb_diff gb_wide gb_meetT gb+oafe > tabprep/logs/q2_s0.log 2>&1
uv run python tabprep/run.py base gb oafe gbT gb+oafe --seed 1 > tabprep/logs/q2_s1.log 2>&1
uv run python tabprep/run.py base gb oafe gbT gb+oafe --seed 2 > tabprep/logs/q2_s2.log 2>&1
echo done > tabprep/logs/queue2.done
