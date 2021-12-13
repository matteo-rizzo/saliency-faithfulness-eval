#!/bin/bash

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/acc/acc_analysis_tccnet_mlp.py"

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("att" "conf" "conf_att")

for sal_dim in "${sal_dims[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for data_folder in "${data_folders[@]}"; do
      declare path_to_base="/media/matteo/Extreme SSD/models/faithful-attention-eval/${sal_dim}/${sal_type}_tccnet"
      declare path_to_diff="/media/matteo/Extreme SSD/results/faithful-attention-eval/mlp/${sal_dim}/${sal_type}/learned"
      python3 "$path_to_script" --sal_dim "$sal_dim" --sal_type "$sal_type" --data_folder "$data_folder" --path_to_base "$path_to_base" --path_to_diff "$path_to_diff" || exit
    done
  done
done
