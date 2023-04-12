

import os
from models.search import is_evaluated, is_trained
from utils.arguments  import replace_param
from jobs.job_body import create_slurm_job
from utils.arguments import options
from constants.paths import JOBS, JOBS_LOGS

JOBNAME = 'evaljob'
root = JOBS

NCPU = 8

def check_training_task(args):
    _,modelid = options(args,key = "model")
    return is_trained(modelid)

def generate_eval_tasks():
    argsfile = 'trainjob.txt'
    path = os.path.join(root,argsfile)
    file = open(path,'r')
    argslist = file.readlines()
    file.close()
    jobnums = []
    for i,args in enumerate(argslist):
        args = args.split()
        args = replace_param(args,'mode','eval')
        argslist[i] = ' '.join(args)


    argsfile = JOBNAME + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write('\n'.join(argslist))

    for i,args in enumerate(argslist):
        args = args.split()
        _,modelid = options(args,key = "model")
        if not is_evaluated(modelid) and is_trained(modelid):
            jobnums.append(str(i+1))
    jobarray = ','.join(jobnums)
    slurmfile =  os.path.join(JOBS,JOBNAME + '.s')
    out = os.path.join(JOBS_LOGS,JOBNAME+ '_%A_%a.out')
    err = os.path.join(JOBS_LOGS,JOBNAME+ '_%A_%a.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/eval.py',
        time = "1:00:00",array = jobarray,\
        mem = "30GB",job_name = JOBNAME,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",#gres="gpu:1",
        ntasks_per_node = "1")


def main():
    generate_eval_tasks()

if __name__=='__main__':
    main()
