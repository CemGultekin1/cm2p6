#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=1-10
#SBATCH --mem=100GB
#SBATCH --job-name=learned_deconv
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/learned_deconv_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/learned_deconv_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/clone_loc/cm2p6/jobs/learned_deconv.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/learned_deconv.py $ARGS;\
	"