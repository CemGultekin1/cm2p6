
import itertools
import os
from jobs.job_body import create_slurm_job
from utils.paths import JOBS, JOBS_LOGS

JOBNAME = 'learned_deconv'
root = JOBS

NCPU = 18
NSEC = 1
PERCPU = 10
def python_args():
    def givearg(filtering,sigma,depth,section):
        st =  f"--minibatch 1 --prefetch_factor 1 --disp 1 --depth {depth} --disp 100 --sigma {sigma} --section {section} --mode data --num_workers {NCPU} --filtering {filtering}"
        return st
    
    filtering = ['gcm']
    sigmas = [4]#,8,12,16]
    depths = [0]#,5]
    
    
    sections = [f'{i} {NSEC}' for i in range(NSEC)]
    prods = (filtering,sigmas,depths,sections)
    lines = []
    for args in itertools.product(*prods):
        lines.append(givearg(*args))
    njob = len(lines)
    lines = '\n'.join(lines)

    argsfile = JOBNAME + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    return njob
def slurm(njob):
    slurmfile =  os.path.join(JOBS,JOBNAME + '.s')
    out = os.path.join(JOBS_LOGS,JOBNAME+ '_%a_%A.out')
    err = os.path.join(JOBS_LOGS,JOBNAME+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/learned_deconv.py',
        time = "2:00:00",array = f"1-{njob}",\
        mem = f"{int(NCPU*PERCPU)}GB",job_name = JOBNAME,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        ntasks_per_node = "1")

def main():
    njob = python_args()
    slurm(njob)



if __name__=='__main__':
    main()
