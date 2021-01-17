#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --error=../../logs/fasttextHDFS2.%j.err
#SBATCH --out=../../logs/fasttextHDFS2.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

FASTTEXT=~/./fasttext/fastText-0.9.2/fasttext
DATADIR=../data/interim/HDFS1
OUTPUTDIR=../models/embeddings


# train FastText on all folds
ml GCC
for i in {1..${1}}
do
    echo ${DATADIR}/train-data-HDFS1-cv-${i}-${1}.log
    echo ${OUTPUTDIR}/fasttext-skipgram-hdfs-d100-n3-6-cv-${i}-${1}
    # ${FASTTEXT} skipgram -input ${DATADIR}/train-data-HDFS1-cv-${i}-${1}.log -output ${OUTPUTDIR}/fasttext-skipgram-hdfs-d100-n3-6-cv-${i}-${1} -dim 100 -minn 3 -maxn 6 -minCount 10000 -thread 16
done

# ${FASTTEXT} skipgram -input ${SCRATCH_DIR}/data.log -output ${OUTPUTDIR}/fasttext-skipgram-hdfs-d100-n3-6 -dim 100 -minn 3 -maxn 6 -minCount 10000 -thread 16
