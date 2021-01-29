#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --error=../../logs/baselineHDFS1.%j.err
#SBATCH --out=../../logs/baselineHDFS1.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

# add the project to Python's path
export PYTHONPATH=$PYTHONPATH:/home/korytmar/methods4logfiles

# train baseline models (Local Outlier Factor, Isolation Forest) and evaluate hyperparameters on grid search
cd /home/korytmar/methods4logfiles/src/models
python train_baseline_models.py