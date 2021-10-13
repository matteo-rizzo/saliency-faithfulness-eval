# This script saves to file the gradients w.r.t. to spatial, temporal or spatiotemporal saliency

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/def-conati/marizzo/xai/ers-faithful-attention-eval/

declare path_to_script="eval/tests/ers/save_grads_tccnet.py"

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a dirs=("fold_0" "fold_2")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

for sal_type in "${sal_types[@]}"; do
  for dir in "${dirs[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      python3 "$path_to_script" --sal_type "$sal_type" --data_folder "$dir" --sal_dim "$sal_dim" --infer_path || exit
      python3 "$path_to_script" --sal_type "$sal_type" --data_folder "$dir" --sal_dim "$sal_dim" --infer_path --use_train_set || exit
    done
  done
done
