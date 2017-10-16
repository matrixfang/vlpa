#!/bin/bash -l



#$ -N ${name}
#$ -o ${rootDir}${logDir}${name}.log
#$ -j y
#$ -V
#$ -P "roughsur"
#$ -l h_rt=12:00:00
#$ -M wyfang@bu.edu
#$ -m ae

name="vlpa"
rootDir="/usr3/graduate/wyfang"
workDir="/GitHub/vlpa"
logDir="/GitHub/vlpa/log/"

cd ${rootDir}${workDir}
module load python/2.7.12
python test.py
mv ${name}.log ${rootDir}${logDir}
