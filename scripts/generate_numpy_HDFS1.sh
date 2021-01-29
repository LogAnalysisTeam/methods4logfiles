#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --error=../../logs/numpyHDFS1.%j.err
#SBATCH --out=../../logs/numpyHDFS1.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

# add the project to Python's path
export PYTHONPATH=$PYTHONPATH:/home/korytmar/methods4logfiles

# generate NumPy arrays from intermediate HDFS1
cd /home/korytmar/methods4logfiles/src/features
python build_features_hdfs.py $1


