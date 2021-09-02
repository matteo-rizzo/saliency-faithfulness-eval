#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/ers/ers_test_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("conf")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spat" "temp")

# Values: "single" "multi"
declare -a erasures=("multi")

for sal_type in "${sal_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      for erasure in "${erasures[@]}"; do
        python3 "$path_to_script" --sal_type "$sal_type" --data_folder "$dir" --sal_dim "$sal_dim" --erasure_type "$erasure" --infer_path || exit
      done
    done
  done
done
