#!/bin/bash


ENV_NAME="mw_drawer-open-v2"
#ENV_NAME="mw_sweep-into-v2"
#ENV_NAME="mw_plate-slide-v2"
#ENV_NAME="mw_door-open-v2"


# Make sure experiments for each environment exist in following paths
if   [ "${ENV_NAME,,}" == "mw_drawer-open-v2" ]; then
  RUN_PATH=(
    "training_logs/${ENV_NAME}/mw_drawer-open-v2_plare"
    "training_logs/${ENV_NAME}/mw_drawer-open-v2_rlvlmf"
  )

elif [ "${ENV_NAME,,}" == "mw_sweep-into-v2" ]; then
  RUN_PATH=(
    "training_logs/${ENV_NAME}/mw_sweep-into-v2_plare"
    "training_logs/${ENV_NAME}/mw_sweep-into-v2_rlvlmf"
  )

elif [ "${ENV_NAME,,}" == "mw_plate-slide-v2" ]; then
  RUN_PATH=(
    "training_logs/${ENV_NAME}/mw_plate-slide-v2_plare"
    "training_logs/${ENV_NAME}/mw_plate-slide-v2_rlvlmf"
  )

elif [ "${ENV_NAME,,}" == "mw_door-open-v2" ]; then
  RUN_PATH=(
    "training_logs/${ENV_NAME}/mw_door-open-v2_plare"
    "training_logs/${ENV_NAME}/mw_door-open-v2_rlvlmf"
  )

else
  echo "Unknown env: ${ENV_NAME}"
  exit 1
fi


python scripts/plot.py --path "${RUN_PATH[@]}"





