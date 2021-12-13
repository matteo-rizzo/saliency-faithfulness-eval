#!/bin/bash

# This script runs the MLP encoder test (WP2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/rand/rand_labels_test_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

for sal_type in "${sal_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
        python3 "$path_to_script" --sal_type "$sal_type" --data_folder "$dir" --sal_dim "$sal_dim" --infer_path_to_pretrained || exit
    done
  done
done
