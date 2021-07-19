#!/bin/bash

# This script runs the MLP encoder test (WP2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="tests/mlp/train_mlp_tccnet.py"

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("att_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a modes=("spatiotemp" "spat" "temp")

for model in "${models[@]}"; do
  for dir in "${dirs[@]}"; do
    for mode in "${modes[@]}"; do
      python3 "$path_to_script" --model_type "$model" --data_folder "$dir" --sal_type "$mode" --infer_path
      python3 "$path_to_script" --model_type "$model" --data_folder "$dir" --sal_type "$mode" --learn_weights --infer_path
    done
  done
done
