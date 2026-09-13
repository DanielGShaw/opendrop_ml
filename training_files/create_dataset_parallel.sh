#!/bin/bash
  
#PBS -l ncpus=48
#PBS -l mem=180GB
#PBS -l jobfs=180GB
#PBS -q normal
#PBS -P oe97
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/oe97
#PBS -l wd

#load modules
module load tensorflow/2.15.0

#load venv
source /scratch/oe97/ds1693/envs/gpu3/bin/activate
python3 --version

# print date and time to denote start
date
echo "Generating contours"

mkdir -p contour_dataset_5par/

# Run python file in parallel piping the angle to %, and defining the roughness value in each line
# -P term may need altering if running on HPC systems
# for example: seq 110 0.02 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 /home/563/ds1693/data_gen/create_contours_args4.py /scratch/oe97/ds1693/contour_dataset_4par_ref/ -a % -r $roughness > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.0 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.002 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.004 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.006 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.008 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r 0.01 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.06 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.05 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.04 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.03 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.02 > $PBS_JOBID.log
seq 20 0.05 180 | xargs -n 1 -P $PBS_NCPUS -t -I % python3 create_contours_args6.py ./contour_dataset_5par/ -a %  -r -0.01 > $PBS_JOBID.log
# print date and time to denote end
#date
#echo "Combining to a single file"

#python3 combine.py

# print date and time to denote end
date
