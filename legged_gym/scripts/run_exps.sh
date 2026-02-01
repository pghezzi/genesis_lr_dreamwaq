#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_pact_water_eval

python exp_water_test.py --liquid_volume 6.0  --liquid_type water --headless
python exp_water_test.py --liquid_volume 8.0  --liquid_type water --headless
python exp_water_test.py --liquid_volume 10.0 --liquid_type water --headless
python exp_water_test.py --liquid_volume 12.0 --liquid_type water --headless