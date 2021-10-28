#!/bin/bash

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/acc/acc_analysis_tccnet_acc.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a splits=("tcc_split" "fold_0" "fold_1" "fold_2")

for sal_type in "${sal_types[@]}"; do
  for sal_dim in "${sal_dims[@]}"; do
    for split in "${splits[@]}"; do
      python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" --data_folder "$split" || exit
    done
  done
done
