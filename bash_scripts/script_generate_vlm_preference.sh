#!/bin/bash

python scripts/generate_vlm_preference.py --env_name "mw_drawer-open-v2"
python scripts/generate_vlm_preference.py --env_name "mw_sweep-into-v2"
python scripts/generate_vlm_preference.py --env_name "mw_plate-slide-v2"
python scripts/generate_vlm_preference.py --env_name "mw_door-open-v2"
