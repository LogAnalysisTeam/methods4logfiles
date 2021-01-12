#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --error=../../logs/fasttextHDFS2.%j.err
#SBATCH --out=../../logs/fasttextHDFS2.%j.out

# clear the environment from any previously loaded modules
ml purge > /dev/null 2>&1

FASTTEXT=~/./fasttext/fastText-0.9.2/fasttext
DATADIR=../data/raw/HDFS2
OUTPUTDIR=../models/embeddings

# create a temporal folder
SCRATCH_DIR=/data/temporary/job_${SLURM_JOB_ID}
mkdir ${SCRATCH_DIR}

# cat HDFS data to a single file
cat ${DATADIR}/*.log > ${SCRATCH_DIR}/data.log

# train FastText
ml GCC
${FASTTEXT} skipgram -input ${SCRATCH_DIR}/data.log -output ${OUTPUTDIR}/fasttext-skipgram-hdfs-d100-n3-6 -dim 100 -minn 3 -maxn 6 -minCount 10000 -thread 16

rm -rf ${SCRATCH_DIR}
