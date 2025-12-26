#!/bin/bash

python scripts/render_metaworld_dataset.py \
  --env "mw_drawer-open-v2" \
  --path datasets/pref/mw_drawer-open-v2_ep2500_n0.3.npz \
  --output datasets/pref_image_only/mw_drawer-open-v2_ep2500_n0.3_img200.pkl

python scripts/render_metaworld_dataset.py \
  --env "mw_sweep-into-v2" \
  --path datasets/pref/mw_sweep-into-v2_ep2500_n0.3.npz \
  --output datasets/pref_image_only/mw_sweep-into-v2_ep2500_n0.3_img200.pkl

python scripts/render_metaworld_dataset.py \
  --env "mw_plate-slide-v2" \
  --path datasets/pref/mw_plate-slide-v2_ep2500_n0.3.npz \
  --output datasets/pref_image_only/mw_plate-slide-v2_ep2500_n0.3_img200.pkl

python scripts/render_metaworld_dataset.py \
  --env "mw_door-open-v2" \
  --path datasets/pref/mw_door-open-v2_ep2500_n0.3.npz \
  --output datasets/pref_image_only/mw_door-open-v2_ep2500_n0.3_img200.pkl
