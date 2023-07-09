#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --array=17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,178,182,186,190,194,198,202,206,210,214,218,222,226,230,234,238,242,246,250,254,258,262,266,270,274,278,282,286,290,294,298,302
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
	python3 run/train.py $ARGS --reset True;\
	python3 run/analysis/eval.py $ARGS --mode eval;\
	python3 run/analysis/distributional.py $ARGS --mode eval;\
	"
echo "$(date)"