#!/bin/bash

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="vis/vis.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a splits=("tcc_split")

for sal_dim in "${sal_dims[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for split in "${splits[@]}"; do
      python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" --data_folder "$split" || exit
      python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" --data_folder "$split" --use_train_set || exit
    done
  done
done
