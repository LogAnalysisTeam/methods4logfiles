#!/bin/bash
#SBATCH --partition=gpulong
#SBATCH --time=72:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --error=../../logs/repHDFS1.%j.err
#SBATCH --out=../../logs/repHDFS1.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

# add the project to Python's path and libraries' path
export PYTHONPATH=$PYTHONPATH:/home/korytmar/methods4logfiles
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/korytmar/anaconda3/lib

# train TCN model and evaluate hyperparameters on random search
cd /home/korytmar/methods4logfiles/src/experiments
python replicate_results.py
