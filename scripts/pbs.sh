#PBS -l walltime=6:00:00,select=1:ncpus=12:ngpus=1:mem=64gb:gpu_mem=12gb
#PBS -N adv
#PBS -A st-conati-1-gpu
#PBS -m abe
#PBS -M matteo.rizzo.phd@gmail.com
#PBS -o output.txt
#PBS -e error.txt

#######################################################

module load python/3.6

cd "$PBS_O_WORKDIR" || exit

pwd

cd ..

source venv/bin/activate

which python3

export PYTHONPATH=$PYTHONPATH:/scratch/st-conati-1/marizzo/pytorch-fc4
