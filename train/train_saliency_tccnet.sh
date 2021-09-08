pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/faithful-attention-eval

declare path_to_script=train/train_saliency_tccnet.py

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

for sal_type in "${sal_types[@]}"; do
  for data_folder in "${data_folders[@]}"; do
    for sal_dim in "${sal_dims[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --epochs 1 || exit
    done
  done
done