#!/bin/bash

for i in {0..5}; do
  python3 train/train.py --epochs 1000 --random_seed "$i" || exit
done
