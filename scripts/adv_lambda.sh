#!/bin/bash

# This script runs the training of the adversarial model for each of the selected lambda values

declare -a lambdas=(0.00005 0.0005 0.005 0.05)

for lambda in "${lambdas[@]}"; do
  python3 train/train.py --epochs 1000 --adv_lambda "$lambda" || exit
done
