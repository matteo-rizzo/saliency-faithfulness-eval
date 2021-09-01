#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values (JW1-WP3)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/adv/adv_analysis_tccnet.py"

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("conf_tccnet" "conf_att_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_types=("spatiotemp")

for model in "${models[@]}"; do
  for sal_type in "${sal_types[@]}"; do
      for data_folder in "${data_folders[@]}"; do
        python3 "$path_to_script" --model_type "$model" --sal_type "$sal_type" --data_folder "$data_folder" || exit
      done
  done
done
