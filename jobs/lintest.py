
import itertools
import os
from jobs.job_body import create_slurm_job
from constants.paths import JOBS, JOBS_LOGS

JOBNAME = __file__.replace('.py','').split('/')[-1]
root = JOBS
NCPU = 4
PERCPU = 30/NCPU
TIME = 48
def python_args():
    def givearg(filtering,depth,sigma,co2):
        st =  f"{sigma} {depth} {co2} 0"
        return st
    
    filtering = ['gcm']#'gaussian']
    sigmas = [4,8,12,16]
    depths = [0,5,55,110,181,330,728]
    co2s = ['False','True']
    
    prods = (filtering,depths,sigmas,co2s)
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
        python_file = 'linear/datagen.py',
        time = f"{TIME}:00:00",array = f"1-{njob}",\
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
