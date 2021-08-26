#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/ers/ers_test_tccnet.py"

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("conf_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a modes=("spat" "temp")

# Values: "single" "multi"
declare -a erasures=("multi")

for model in "${models[@]}"; do
  for dir in "${dirs[@]}"; do
    for mode in "${modes[@]}"; do
      for erasure in "${erasures[@]}"; do
        python3 "$path_to_script" --model_type "$model" --data_folder "$dir" --sal_type "$mode" --erasure_type "$erasure" --infer_path || exit
      done
    done
  done
done
