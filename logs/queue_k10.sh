#!/bin/zsh
# File « 10 plis » (31/10) : modèles utiles à l'ensemble réentraînés sur 10 plis emboîtés (journal, section 21).
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%H:%M') $*"; uv run main.py "$@" --folds 10 2>&1 | grep --line-buffered -v Warning; }
run train --model rf --target C --name rf_C_k10
run train --model et --target C --name et_C_k10
run train --model hgb --target A --name hgb_raw_k10
run train --model lgbm --target C --seeds 5 --name lgbm_C_k10
run train --model lgbm --target B --seeds 5 --name lgbm_B_k10
run train --model lgbm --target C --seeds 3 --name lgbm_C_xt_k10 --set extra_trees=true
run train --model xgb --target C --seeds 5 --name xgb_C_k10
run train --model cat --target C --name cat_C_k10
run train --model mlp --target C --seeds 5 --name mlp_C_plr_k10
echo "=== $(date '+%H:%M') fin"
