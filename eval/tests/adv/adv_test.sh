#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values (JW1-WP3)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/tests/adv/adv_test_tccnet.py"

# Values: 0.00005 0.0005 0.005 0.05
declare -a lambdas=(0.00005 0.0005 0.005 0.05)

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp")

for lambda in "${lambdas[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" --adv_lambda "$lambda" --epochs 500 --infer_path || exit
    done
  done
done
