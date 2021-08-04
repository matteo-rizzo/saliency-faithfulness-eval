#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values (JW1-WP3)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="tests/adv/train_adv_tccnet.py"

# Values: 0.00005 0.0005 0.005 0.05
declare -a lambdas=(0.00005)

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("conf_att_tccnet" "att_tccnet" "conf_tccnet")

# Values: "spatiotemp" "spat" "temp"
declare -a modes=("spatiotemp")

for lambda in "${lambdas[@]}"; do
  for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
      python3 "$path_to_script" --epochs 1000 --model_type "$model" --adv_lambda "$lambda" --sal_type "$mode" --infer_path || exit
    done
  done
done
