#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --array=1
#SBATCH --mem=30GB
#SBATCH --job-name=fgath
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/fgath_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/fgath_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
module purge
singularity exec --overlay /scratch/cg3306/climate/.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/gather_filter_weights.py $SLURM_ARRAY_TASK_ID;\
	"