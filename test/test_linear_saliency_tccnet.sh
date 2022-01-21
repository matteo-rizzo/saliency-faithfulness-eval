source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/def-conati/marizzo/xai/ers-faithful-attention-eval/

declare path_to_script=test/test_linear_saliency_tccnet.py

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "imposed" "learned" "baseline"
declare -a weights_modes=("learned")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split" "fold_0" "fold_1" "fold_2")

for sal_dim in "${sal_dims[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for weights_mode in "${weights_modes[@]}"; do
      for data_folder in "${data_folders[@]}"; do
        declare path_to_pretrained="/media/matteo/Extreme SSD/results/faithful-attention-eval/mlp/${sal_dim}/${sal_type}/${weights_mode}/${data_folder}"
        declare path_to_sw="/media/matteo/Extreme SSD/models/faithful-attention-eval/${sal_dim}/${sal_type}_tccnet/${data_folder}/sal"
        python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --path_to_pretrained "$path_to_pretrained" --path_to_sw "$path_to_sw" --use_train_set|| exit
      done
    done
  done
done
