
import itertools
import os
from jobs.job_body import create_slurm_job
from constants.paths import JOBS, JOBS_LOGS

JOBNAME = 'datagen'
root = JOBS

NCPU = 3
NSEC = 15
PERCPU = 50
def python_args():
    def givearg(co2,filtering,sigma,depth,section):        
        st =  f"--minibatch 1 --depth {depth} --sigma {sigma} --section {section} --co2 {co2} --mode data --num_workers {NCPU} --filtering {filtering}"
        return st
    
    filtering = ['gaussian','gcm']
    sigmas = [4,8,12,16]
    depths = [0]
    co2s = [True]
    
    sections = [f'{i} {NSEC}' for i in range(NSEC)]
    prods = (co2s,filtering,sigmas,depths,sections)
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
    out = os.path.join(JOBS_LOGS,JOBNAME+ '_%A_%a.out')
    err = os.path.join(JOBS_LOGS,JOBNAME+ '_%A_%a.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/datagen.py',
        time = "48:00:00",array = f"1-{njob}",\
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
