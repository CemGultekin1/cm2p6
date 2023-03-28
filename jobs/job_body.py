from utils.paths import ENV_PATH
def job(argsfile,python_file,add_eval:bool = False,**kwargs):
    head = ["#!/bin/bash"]
    intro = [f"#SBATCH --{key.replace('_','-')}={val}" for key,val in kwargs.items()]
    dateline = [f"echo \"$(date)\""]
    bashline = [
        f"ARGS=$(sed -n \"$SLURM_ARRAY_TASK_ID\"p {argsfile})"
    ]
    env = ENV_PATH
    bodystart = ["module purge",\
       f"singularity exec --nv --overlay {env}:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c \"\\"]
    codebody = [
        "source src.sh;",
        f"python3 {python_file} $ARGS;"
    ]
    if add_eval:
        codebody.append(f"python3 run/eval.py $ARGS --mode eval;")
    codebody = ['\t' + cb + '\\' for cb in codebody]
    codebody.append('\t\"')
    return "\n".join(head +  intro + dateline + bashline + bodystart + codebody + dateline)

def create_slurm_job(path,python_file = 'run/train.py',**kwargs):
    argsfile = path.replace('.s','.txt')
    text = job(argsfile,python_file,**kwargs)
    with open(path,'w') as f:
        f.write(text)
