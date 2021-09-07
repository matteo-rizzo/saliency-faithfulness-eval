pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/faithful-attention-eval

declare path_to_script=train/train_tccnet.py

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split")

for data_folder in "${data_folders[@]}"; do
    python3 $path_to_script --data_folder "$data_folder" --epochs 1 || exit
done