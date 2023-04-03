#!/bin/bash
#SBATCH --time=28:00:00
#SBATCH --array=1,2
#SBATCH --mem=150GB
#SBATCH --job-name=trainjob
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/trainjob.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/train.py $ARGS;\
	python3 run/eval.py $ARGS --mode eval;\
	"
echo "$(date)"