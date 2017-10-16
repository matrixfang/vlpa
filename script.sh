#!/bin/bash -l

#$ -N "vlpa"
##$ -o ${rootDir}${logDir}vlpa.log
#$ -o sr.log
#$ -j y
#$ -V
#$ -P "roughsur"
#$ -l h_rt=12:00:00
#$ -M wyfang@bu.edu
#$ -m ae

rootDir="/usr3/graduate/wyfang"
workDir="/GitHub/vlpa"
logDir="/GitHub/vlpa/log/"

cd ${rootDir}${workDir}
module load python/2.7.12
python test.py
mv sr.log ${rootDir}${logDir}
