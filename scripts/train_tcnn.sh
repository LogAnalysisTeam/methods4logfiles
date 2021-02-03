#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --error=../../logs/tcnnHDFS1.%j.err
#SBATCH --out=../../logs/tcnnHDFS1.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

# add the project to Python's path
export PYTHONPATH=$PYTHONPATH:/home/korytmar/methods4logfiles

# train TCN model and evaluate hyperparameters on random search
cd /home/korytmar/methods4logfiles/src/models
python train_tcnn.py