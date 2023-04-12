#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --array=1,5,11,57,79,80,82,88,183,270,273,276,379,380,382,383,385,386,388,389,390,391,392,393,394,395,397,398,400,401,403,404,406,407,409,410,412,413,414,415,416,417,418,419,421,422,424,425,427,428,430,431,433,434,436,437,438,439,440,441,442,443,445,446,448,449,451,452,454,455,457,458,460,461,462,463,464,465,466,467,469,470,472,473,475,476,478,479,481,482,484,485,486,487,488,489,490,491,493,494,496,497,499,500,502,503,505,506,508,509,510,511,512,513,514,515,517,518,520,521,523,524,526,527,529,530,532,533,1050,1053,1056,1059,1062,1065,1068,1071,1074,1077,1080,1083,1086,1089,1092,1095,1098,1101,1104,1107,1110,1113,1116,1119,1122,1125,1128,1131,1134,1137,1140,1143,1146,1149,1152,1155,1158,1161,1164,1167,1170,1173,1176,1179,1182,1185,1188,1191,1194,1197,1200,1203,1375,1379,1382,1385,1400,1403,1406,1409,1422,1424,1430,1433,1446,1448,1451,1454,1457,1470,1471,1475,1494,1495,1499,1502,1505,1518,1520,1523,1526,1529,1542,1544,1547,1550,1553
#SBATCH --mem=150GB
#SBATCH --job-name=trainjob
#SBATCH --output=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/outputs/slurm_logs/trainjob_%A_%a.err
#SBATCH --cpus-per-task=8
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
	python3 run/distributional.py $ARGS --mode eval;\
	python3 run/legacy_comparison.py $ARGS --mode eval;\
	python3 run/legacy_snapshots.py $ARGS --mode eval;\
	"
echo "$(date)"