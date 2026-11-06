#!/bin/zsh
cd "$(dirname "$0")/.."
for v in rsfc rsfc_small gb oafe cat gb_meet oafe_te oafe_F500; do
  uv run python tabprep/run.py $v > tabprep/logs/$v.log 2>&1
done
echo QUEUE1_DONE >> tabprep/logs/queue1.done
