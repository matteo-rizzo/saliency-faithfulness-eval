pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/faithful-attention-eval

declare path_to_script=tests/accuracy/test_tccnet.py

# Values: "att_tccnet" "conf_tccnet" "conf_att_tccnet"
declare -a models=("att_tccnet" "conf_tccnet" "conf_att_tccnet")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_types=("spatiotemp" "spat" "temp")

for model in "${models[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    for sal_type in "${sal_types[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --model_type "$model" --use_train_set --sal_type "$sal_type" --infer_path --save_pred --save_sal || exit
      python3 $path_to_script --data_folder "$data_folder" --model_type "$model" --sal_type "$sal_type" --infer_path || exit
    done
  done
done