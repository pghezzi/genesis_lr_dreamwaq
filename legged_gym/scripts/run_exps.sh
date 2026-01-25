#!/bin/bash

. /home/oscaryoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oscaryoungquist/.conda/envs/gensis_pact

python exp_water_test.py --liquid_volume 6.0 --liquid_type water --headless
python exp_water_test.py --liquid_volume 8.0 --liquid_type water --headless
python exp_water_test.py --liquid_volume 10.0 --liquid_type water --headless
python exp_water_test.py --liquid_volume 12.0 --liquid_type water --headless