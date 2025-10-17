#!/bin/bash


ENV="mw_drawer-open-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/rlvlmf.yaml \
  --exp_name "${ENV}_rlvlmf" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --seed 1


ENV="mw_sweep-into-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/rlvlmf.yaml \
  --exp_name "${ENV}_rlvlmf" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --seed 1


ENV="mw_plate-slide-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/rlvlmf.yaml \
  --exp_name "${ENV}_rlvlmf" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --seed 1


ENV="mw_door-open-v2"
DATA_PATH="datasets/vlm_feedback/${ENV}_ep2500_n0.3_vlm_label.npz"
python scripts/train.py \
  --config configs/rlvlmf.yaml \
  --exp_name "${ENV}_rlvlmf" \
  --env_name ${ENV} \
  --data_path ${DATA_PATH} \
  --seed 1