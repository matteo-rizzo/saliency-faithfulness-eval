#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values (JW1-WP3)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare -a lambdas=(0.00005 0.0005 0.005 0.05)
declare -a model_types=("att_tccnet" "conf_tccnet" "conf_att_tccnet")
declare -a modes=("spatiotemp" "spat" "temp")

for lambda in "${lambdas[@]}"; do
  for model_type in "${model_types[@]}"; do
    for mode in "${modes[@]}"; do
      python3 tests/adv/train_adv_tccnet.py --epochs 1000 --model_type "$model_type" --adv_lambda "$lambda" --mode "$mode" || exit
    done
  done
done
