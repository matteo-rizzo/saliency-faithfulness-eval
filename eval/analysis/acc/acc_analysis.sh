#!/bin/bash

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/matteo/Projects/faithful-attention-eval/

declare path_to_script="eval/analysis/acc/acc_analysis_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "spatiotemp" "spat" "temp"
declare -a result_types=("acc" "mlp")

for sal_type in "${sal_types[@]}"; do
  for sal_dim in "${sal_dims[@]}"; do
    for result_type in "${result_types[@]}"; do
      python3 "$path_to_script" --sal_type "$sal_type" --sal_dim "$sal_dim" --result_type "$result_type" || exit
    done
  done
done
