source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/def-conati/marizzo/xai/faithful-attention-eval/

declare path_to_script=test/test_saliency_tccnet.py

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spat" "temp")

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

for sal_dim in "${sal_dims[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for data_folder in "${data_folders[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" || exit
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --use_train_set || exit
    done
  done
done
