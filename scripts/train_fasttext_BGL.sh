#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --error=../../logs/fasttextBGL.%j.err
#SBATCH --out=../../logs/fasttextBGL.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

FASTTEXT=~/./fasttext/fastText-0.9.2/fasttext
DATADIR=../data/raw/BGL
OUTPUTDIR=../models/embeddings

# train FastText
ml GCC
${FASTTEXT} skipgram -input ${DATADIR}/BGL.log -output ${OUTPUTDIR}/fasttext-skipgram-bgl-d100-n3-6 -dim 100 -minn 3 -maxn 6 -minCount 10000 -thread 16
