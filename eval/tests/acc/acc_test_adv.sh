pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/def-conati/marizzo/xai/acc-faithful-attention-eval/

declare path_to_script=eval/tests/acc/acc_test_tccnet.py

# Values: "att" "conf" "conf_att"
declare -a sal_types=("conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "00005" "0005" "005" "05"
declare -a lambdas=("00005" "0005" "005" "05")

for sal_type in "${sal_types[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      for lambda in "${lambdas[@]}"; do
        declare path_to_pth="trained_models/adv/${sal_dim}/${sal_type}/${data_folder}/${lambda}"
        python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --save_pred --save_sal --path_to_pretrained $path_to_pth || exit
        python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --save_pred --save_sal --path_to_pretrained $path_to_pth --use_train_set || exit
      done
    done
  done
done