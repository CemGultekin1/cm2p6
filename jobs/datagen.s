#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-10
#SBATCH --mem=180GB
#SBATCH --job-name=datagen
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/datagen_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/datagen_%A_%a.err
#SBATCH --cpus-per-task=18
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$JOBS_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/jobs/datagen.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/datagen.py $ARGS;\
	"