#!/bin/bash
  
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=8GB
#PBS -q gpuvolta
#PBS -P oe97
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/oe97
#PBS -l wd

#clear any already loaded modules
module purge

#load modules
#module load python3/3.8.5  #3.7/4
#module load opencv/4.5.5
#module load cudnn/8.6.0-cuda11  
#module load tensorflow/2.8.0
module load tensorflow/2.15.0

#Add venv to python path
echo PYTHONPATH is:
echo $PYTHONPATH

#load venv
#source /scratch/oe97/ds1693/gpu_2/bin/activate
source /scratch/oe97/ds1693/envs/gpu3/bin/activate
python3 --version

# should be /scratch/oe97/ds1693/envs/gpu_2/lib/python3.11/site-packages:/apps/tensorflow/2.15.0/lib/python3.11/site-packages
echo Add venv to path...
#export PYTHONPATH=/scratch/oe97/ds1693/envs/gpu_2/lib/python3.11/site-packages:$PYTHONPATH
echo PYTHONPATH is
echo $PYTHONPATH

# print date and time to denote start
date

# run the train file
#python3 train_optuna.py > output.txt
#python3 load_optuna.py > output.txt
#python3 finish_optuna.py > output.txt
#python3 check_tf.py > check_output.txt
python3 test_evaluation.py

# print date and time to denote end
date

deactivate
