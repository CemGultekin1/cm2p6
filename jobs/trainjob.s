#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --array=1,2,15,19,78,79,82,87,88,96,104,106,402,403,406,411,412,415,420,421,424,429,430,433,436,438,439,442,445,447,448,451,456,457,460,465,466,469,474,475,478,483,484,487,492,493,496,501,502,505,508,510,511,514,517,519,520,523,528,529,532,537,538,541,546,547,550,555,556,559,564,565,568,573,574,577,580,582,583,586,589,591,592,595,600,601,604,609,610,613,618,619,622,627,628,631,636,637,640,645,646,649,652,654,655,658,661,663,664,667,672,673,676,681,682,685,690,691,694,699,700,703,708,709,712,717,718,721,724,726,727,730,733,735,736,739,744,745,748,753,754,757,762,763,766,771,772,775,780,781,784,789,790,793,796,798,799,802,805,807,808,811,816,817,820,825,826,829,834,835,838,843,844,847,852,853,856,861,862,865,1373,1377,1380,1383,1398,1401,1404,1407,1420,1422,1428,1431,1444,1446,1449,1452,1455,1468,1469,1473,1492,1493,1497,1500,1503,1516,1518,1521,1524,1527,1540,1542,1545,1548,1551
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