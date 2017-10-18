#!/bin/bash

taskname="vlpa"
rootDir="/usr3/graduate/wyfang"
workDir="/GitHub/vlpa"
logDir="/GitHub/vlpa/log/"

qsub<<EOF
#!/bin/bash -l

#$ -N ${taskname}
#$ -o ${rootDir}${logDir}${taskname}.log
#$ -j y
#$ -V
#$ -P "roughsur"
#$ -l h_rt=12:00:00
#$ -M wyfang@bu.edu
#$ -m ae

cd ${rootDir}${workDir}
module load python/2.7.13
python main.py
mv ${taskname}.log ${rootDir}${logDir}
EOF
