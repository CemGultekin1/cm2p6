#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=11,12
#SBATCH --mem=80GB
#SBATCH --job-name=evjob
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/evjob_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/evjob_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/trainjob.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source /ext3/env.sh;\
	python3 run/analysis/eval.py $ARGS --mode eval;\
	python3 run/analysis/distributional.py $ARGS --mode eval;\
	python3 run/analysis/legacy_comparison.py $ARGS --mode eval;\
	python3 run/analysis/legacy_snapshots.py $ARGS --mode eval;\
	"
echo "$(date)"