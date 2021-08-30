#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values (JW1-WP3)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/adv/adv_test_tccnet.py"

# Values: 0.00005 0.0005 0.005 0.05
declare -a lambdas=(0.00005 0.0005 0.005 0.05)

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("att_tccnet" "conf_tccnet" "conf_att_tccnet")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_types=("spat" "temp" "spatiotemp")

for lambda in "${lambdas[@]}"; do
  for model in "${models[@]}"; do
    for sal_type in "${sal_types[@]}"; do
      python3 "$path_to_script" --epochs 1000 --model_type "$model" --adv_lambda "$lambda" --sal_type "$sal_type" --infer_path || exit
    done
  done
done
