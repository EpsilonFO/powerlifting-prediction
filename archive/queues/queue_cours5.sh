#!/bin/zsh
# File de calculs « cours 5 » (points 2 et 3), lancée le 31/10. Un calcul à la fois.
cd "$(dirname "$0")/.."
while pgrep -f "main.py tune --model lgbm" > /dev/null; do sleep 30; done
run() { echo "=== $(date '+%H:%M') $*"; uv run main.py "$@" 2>&1 | grep -v Warning; }
# point 3 : LightGBM retuné sur v2 / cible C (à comparer à lgbm_C_v2 sur les plis 3-4)
run train --model lgbm --target C --seeds 5 --name lgbm_C_v2_tuned
# point 2 : variantes LightGBM (diversité)
run train --model lgbm --target C --seeds 3 --name lgbm_C_v2_xt --set extra_trees=true
run train --model lgbm --target A --seeds 3 --name lgbm_A_v2_lin --set linear_tree=true
# point 2 : forêts tunées sur v2 / cible C
run tune --model rf --target C --trials 20
run train --model rf --target C --name rf_C_v2
run tune --model et --target C --trials 20
run train --model et --target C --name et_C_v2
# point 3 : XGBoost (référence non tunée, puis tuné)
run train --model xgb --target C --seeds 5 --no-tuned --name xgb_C_v2_default
run tune --model xgb --target C --trials 30
run train --model xgb --target C --seeds 5 --name xgb_C_v2
# point 3 : CatBoost (le plus long ; référence non tunée : cat_C_v2)
run tune --model cat --target C --trials 20
run train --model cat --target C --name cat_C_v2_tuned
echo "=== $(date '+%H:%M') fin"
