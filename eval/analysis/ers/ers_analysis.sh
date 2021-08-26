#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/ers/ers_analysis_tccnet.py"

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("att_tccnet" "conf_tccnet" "conf_att_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_types=("spatiotemp" "spat" "temp")

# Values: "single" "multi"
declare -a test_types=("single" "multi")

for model in "${models[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_type in "${sal_types[@]}"; do
      for test_type in "${test_types[@]}"; do
        python3 "$path_to_script" --model_type "$model" --data_folder "$dir" --sal_type "$sal_type" --test_type "$test_type" || exit
      done
    done
  done
done
