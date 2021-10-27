pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/def-conati/marizzo/xai/ers-faithful-attention-eval/

declare path_to_script=eval/tests/acc/acc_test_tccnet.py

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spat")

for sal_type in "${sal_types[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --use_uniform --save_pred --save_sal --infer_path || exit
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --use_uniform --save_pred --save_sal --use_train_set --infer_path || exit
    done
  done
done