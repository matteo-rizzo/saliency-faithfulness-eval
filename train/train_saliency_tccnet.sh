#!/bin/bash

#PBS -l walltime=72:00:00,select=1:ncpus=12:ompthreads=12:ngpus=1:mem=186gb
#PBS -N conf_f0_st
#PBS -A st-conati-1
#PBS -m abe
#PBS -M matteo.rizzo.phd@gmail.com
#PBS -o output.txt
#PBS -e error.txt

#######################################################

module load python/3.6

cd ../..

pwd
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/scratch/st-conati-1/marizzo/xai/faithful-attention-eval

declare path_to_script=train/train_linear_saliency_tccnet.py

# Values: "spatiotemp" "spat" "temp"
declare -a sal_dims=("spatiotemp" "spat" "temp")

# Values: "att" "conf" "conf_att"
declare -a sal_types=("att" "conf" "conf_att")

# Values: "tcc_split" "fold_0" "fold_1" "fold_2"
declare -a data_folders=("tcc_split" "fold_0" "fold_1" "fold_2")

for sal_dim in "${sal_dims[@]}"; do
  for sal_type in "${sal_types[@]}"; do
    for data_folder in "${data_folders[@]}"; do
      python3 $path_to_script --data_folder "$data_folder" --sal_type "$sal_type" --sal_dim "$sal_dim" --epochs 500 || exit
    done
  done
done
