#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare -a model_types=("att_tccnet" "conf_tccnet" "conf_att_tccnet")
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")
declare -a modes=("" "spat" "temp")

for model_type in "${model_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for mode in "${modes[@]}"; do
      python3 tests/erasure/save_grads_tccnet.py --model_type "$model_type" --data_folder "$dir" --deactivate "$mode" || exit
    done
  done
done
