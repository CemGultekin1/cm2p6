#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --array=1-15
#SBATCH --mem=150GB
#SBATCH --job-name=datagen
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/datagen_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/datagen_%A_%a.err
#SBATCH --cpus-per-task=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/datagen.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python3 run/datagen.py $ARGS;\
	"
echo "$(date)"