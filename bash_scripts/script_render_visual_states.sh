#!/bin/bash

python scripts/render_metaworld_dataset.py \
  --env "mw_drawer-open-v2" \
  --path datasets/pref/mw_drawer-open-v2_ep2500_n0.3.npz \
  --output datasets/pref_image_only/mw_drawer-open-v2_ep2500_n0.3_img200.pkl