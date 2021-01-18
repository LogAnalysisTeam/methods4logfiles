#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --error=../../logs/fasttextHDFS1.%j.err
#SBATCH --out=../../logs/fasttextHDFS1.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

FASTTEXT=~/./fasttext/fastText-0.9.2/fasttext
DATADIR=../data/interim/HDFS1
OUTPUTDIR=../models/embeddings

# train FastText on all folds
ml GCC
for i in $(seq 1 $1)
do
    ${FASTTEXT} skipgram -input ${DATADIR}/train-data-HDFS1-cv${i}-${1}.log -output ${OUTPUTDIR}/fasttext-skipgram-hdfs1-d100-n3-6-cv${i}-${1} -dim 100 -minn 3 -maxn 6 -minCount 10000 -thread 1
done
