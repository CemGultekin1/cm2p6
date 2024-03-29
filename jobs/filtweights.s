#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --array=2027,2355,2451-2452,2928,2942
#SBATCH --mem=20GB
#SBATCH --job-name=filtweights
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/filtweights_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/filtweights_%A_%a.err
#SBATCH --cpus-per-task=4
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/filtweights.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python3 linear/coarse_graining_operators.py $ARGS;
	"
echo "$(date)"