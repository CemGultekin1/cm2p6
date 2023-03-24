#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=2,6,7,10,15,16,19,24,25,28,33,34,37,42,43,46,51,52,55,60,61,64,69,70,73,78,79,82,87,88,89,91,92,96,97,100,104,106,109,114,115,118,123,124,127,132,133,136,141,142,145,150,151,154,159,160,163,168,169,172,177,178,181,186,187,190,195,196,199,204,205,208,213,214,217,364,366,367,370,373,375,376,379,384,385,388,393,394,397,402,403,406,411,412,415,420,421,424,429,430,433,436,438,439,442,445,447,448,451,456,457,460,465,466
#SBATCH --mem=150GB
#SBATCH --job-name=trainjob
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/clone_loc/cm2p6/jobs/trainjob.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/train.py $ARGS;\
	python3 run/eval.py $ARGS --mode eval;\
	"