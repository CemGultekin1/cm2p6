#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --array=1-3
#SBATCH --mem=72GB
#SBATCH --job-name=scalars
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/scalars_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/scalars_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$JOBS_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/jobs/scalars.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/scalars.py $ARGS;\
	"