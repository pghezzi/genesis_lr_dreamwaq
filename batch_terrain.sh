#!/bin/bash

# Number of terrain dimensions (adjust if needed)
N=7

for ((i=0; i<N; i++)); do
  echo "=============================="
  echo "Starting run with TERRAIN index $i set to 1"
  echo "=============================="

  # Build TERRAIN string like: 0,0,1,0,0,0,0
  terrain=""
  for ((j=0; j<N; j++)); do
    if [ "$j" -eq "$i" ]; then
      terrain+="1"
    else
      terrain+="0"
    fi

    # Add comma except for last element
    if [ "$j" -lt $((N-1)) ]; then
      terrain+=","
    fi
  done

  echo "TERRAIN=$terrain"

  TERRAIN=$terrain python /home/pablo/Documents/HCR_Genesis_LR_CL/legged_gym/scripts/train.py \
    --task=go2_dreamwaq_env --headless
done