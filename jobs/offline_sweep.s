#!/bin/bash
#SBATCH --time=28:00:00
#SBATCH --array=51,52,53,54,55,56,57,58,59,60,61,62,63,64,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,180,181,182,184,185,186,188,189,190,192,193,194,196,197,198,200,201,202,204,205,206,208,209,210,212,216,217,218,220,221,222,224,225,226,228,229,230,232,233,234,236,237,238,240,241,242,244,245,246,248,252,256,257,258,260,261,262,264,265,266,268,269,270,272,273,274,276,277,278,280,281,282,284,285,286,288,289,290,292,293,294,296,297,298,300,301,302,304
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
	"
echo "$(date)"