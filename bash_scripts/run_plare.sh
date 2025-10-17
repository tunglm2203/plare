#!/bin/bash


ENV="mw_drawer-open-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/plare.yaml \
  --exp_name "${ENV}_plare" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --ignore_unsure True \
  --dropout 0.4 \
  --seed 1


ENV="mw_sweep-into-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/plare.yaml \
  --exp_name "${ENV}_plare" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --ignore_unsure True \
  --dropout 0.25 \
  --seed 1


ENV="mw_plate-slide-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/plare.yaml \
  --exp_name "${ENV}_plare" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --ignore_unsure True \
  --dropout 0.4 \
  --seed 1


ENV="mw_door-open-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/plare.yaml \
  --exp_name "${ENV}_plare" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --ignore_unsure True \
  --dropout 0.25 \
  --seed 1