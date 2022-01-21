#!/bin/bash

# This script runs the intermediate representation erasure tests (SS1-SS2)

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/ers/multi_ers_agg_results_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

for sal_type in "${sal_types[@]}"; do
  for sal_dim in "${sal_dims[@]}"; do
    python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" || exit
  done
done
