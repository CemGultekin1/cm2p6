#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,22,23,24,25,26,27,28,29,30,31,32,49,51,53,54,55,57,58,59,60,61,62,63,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,111,112,113,114,117,118,120,149,153,157,161,165,169,173,177,181,182,185,186,188,189,190,193,197,198,201,202,205,206,209,213,214,217,218,221,222,225,229,230,233,234,237,238,241,245,246,249,250,252,253,254,257,261,262,265,266,269,270,273,277,278,281,282,285,286,290,293,294,297,298,301,302
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