#!/bin/bash

# This script runs the MLP encoder test (WP2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/mlp/mlp_test_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split" "fold_0" "fold_1" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "imposed" "learned" "baseline"
declare -a modes=("imposed" "learned" "baseline")

for sal_type in "${sal_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      for mode in "${modes[@]}"; do
        python3 "$path_to_script" --sal_type "$sal_type" --data_folder "$dir" --sal_dim "$sal_dim" --mode "$mode" --infer_path_to_pretrained
      done
    done
  done
done
