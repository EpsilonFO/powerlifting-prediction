#!/bin/zsh
# File « 10 plis, anciens modèles » (01/11) : modèles v1 encore utiles à l'ensemble, réentraînés à l'identique
# (features v1, mêmes hyperparamètres) sur 10 plis emboîtés (journal, section 24).
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%H:%M') $*"; uv run main.py "$@" --folds 10 --features v1 2>&1 | grep --line-buffered -v Warning; }
if [[ "$1" != "long" ]]; then
  run train --model ridge --target A --name ridge_A_k10
  run train --model rf --target A --no-tuned --name rf_A_k10
  run train --model et --target A --no-tuned --name et_A_k10
else  # 2e partie, lancée avec l'argument « long »
  run train --model lgbm --target B --seeds 5 --params lgbm_v1 --name lgbm_B_tuned_k10
  run train --model mlp --target C --seeds 5 --name mlp_C_plr_s5_k10
fi
echo "=== $(date '+%H:%M') fin"
