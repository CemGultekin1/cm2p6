#!/bin/bash
#SBATCH --time=28:00:00
#SBATCH --array=50,52,54,62,150,151,152,154,155,156,158,159,160,162,163,164,166,167,168,170,171,172,174,175,176,180,182,192,193,194,196,209,210,212,225,226,228,230,241,242,244,284,286,297,300
#SBATCH --mem=80GB
#SBATCH --job-name=offline_sweep
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/offline_sweep_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/offline_sweep_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/offline_sweep.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python3 run/train.py $ARGS;\
	python3 run/analysis/eval.py $ARGS --mode eval;\
	"
echo "$(date)"