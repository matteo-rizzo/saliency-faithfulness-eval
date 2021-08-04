#!/bin/bash

# This script runs the training of the base model for the selected number of random seeds

declare num_seeds=5

for i in $(seq 1 $num_seeds); do
  python3 train/train.py --epochs 1000 --random_seed "$i" || exit
done
