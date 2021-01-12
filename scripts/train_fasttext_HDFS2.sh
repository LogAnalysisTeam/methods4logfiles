#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --error=../logs/fasttextHDFS.%j.err
#SBATCH --out=../logs/fasttextHDFS.%j.out

FASTTEXT=~./fasttext/fastText-0.9.2/fasttext
DATADIR=../data/raw/HDFS2
OUTPUTDIR=../models/embeddings

# create a temporal folder
SCRATCH_DIR=/data/temporary/job_${SLURM_JOB_ID}
mkdir ${SCRATCH_DIR}

# cat HDFS data to a single file
cat ${DATADIR}/*.log > ${SCRATCH_DIR}/data.log

# train FastText
ml GCC
${FASTTEXT} skipgram -input ${SCRATCH_DIR}/data.log -output ${OUTPUTDIR}/fasttext-skipgram-hdfs-d100-n1-1 -dim 100 -minn 1 -maxn 1 -minCount 10000 -thread 16

rm -rf ${SCRATCH_DIR}