#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=1-4
#SBATCH --mem=30GB
#SBATCH --job-name=evaljob
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/evaljob_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/evaljob_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
echo "$(date)"
ARGS=$(sed -n "$JOBS_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/jobs/trainjob.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/legacy_comparison.py $ARGS --mode eval;\
	python3 run/legacy_snapshots.py $ARGS --mode eval;
	"
echo "$(date)"