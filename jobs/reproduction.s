#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --mem=10GB
#SBATCH --job-name=reproduction
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/reproduction_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/reproduction_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
echo "$(date)"
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/cm2p6/jobs/reproduction.txt)
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/cm2p6/overlay-15GB-500K.ext3:ro\
	 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "
		source /ext3/env.sh;
		python3 run/train.py $ARGS;
		python3 run/analysis/eval.py $ARGS --mode eval;
		python3 run/analysis/distributional.py $ARGS --mode eval;
		python3 run/analysis/legacy_comparison.py $ARGS --mode eval;
		python3 run/analysis/legacy_snapshots.py $ARGS --mode eval;
	"
echo "$(date)"