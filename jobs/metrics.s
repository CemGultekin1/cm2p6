#!/bin/bash
#SBATCH --time=40:00
#SBATCH --array=1-20
#SBATCH --mem=20GB
#SBATCH --job-name=metrics
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/metrics_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/metrics_%A_%a.err
#SBATCH --cpus-per-task=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
echo "$(date)"
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python metrics/gather_evals1.py fcnn 20 $SLURM_ARRAY_TASK_ID;\
	"
echo "$(date)"