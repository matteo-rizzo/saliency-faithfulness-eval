#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/ers/ers_analysis_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "single" "multi"
declare -a test_types=("single" "multi")

for sal_type in "${sal_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      for test_type in "${test_types[@]}"; do
        python3 "$path_to_script" --data_folder "$dir" --sal_type "$sal_type" --sal_dim "$sal_dim" --test_type "$test_type" || exit
      done
    done
  done
done
