#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --array=49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268,272,276,280,284,288,292,296,300,304
#SBATCH --mem=80GB
#SBATCH --job-name=offline_sweep2
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/offline_sweep2_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/offline_sweep2_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/offline_sweep2.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python3 run/train.py $ARGS;\
	python3 run/analysis/eval.py $ARGS --mode eval;\
	python3 run/analysis/distributional.py $ARGS --mode eval;\
	"
echo "$(date)"