#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare -a model_types=("att_tccnet" "conf_tccnet" "conf_att_tccnet")
declare -a modes=("" "spatial" "temporal")

for model_type in "${model_types[@]}"; do
  for mode in "${modes[@]}"; do
    python3 tests/erasure/erasure_tccnet.py --model_type "$model_type" --deactivate "$mode" || exit
  done
done
