#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,177,181,185,189,193,197,201,205,209,213,217,221,225,229,233,237,241,245,249,253,257,261,265,269,273,277,281,285,289,293,297,301
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
	python3 run/analysis/eval.py $ARGS --mode eval;\
	python3 run/analysis/distributional.py $ARGS --mode eval;\
	"
echo "$(date)"