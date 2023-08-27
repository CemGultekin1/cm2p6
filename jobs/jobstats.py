from collections import defaultdict
from typing import List
from constants.paths import JOBS_LOGS
import os
import numpy as np
from utils.slurm import ArgsReader

def gather_logs(task_name,must_includes = []):
    logs = os.listdir(JOBS_LOGS)
    sublogs = []
    for log in logs:
        all_included = task_name in log
        if not all_included:
            continue
        for mistr in must_includes:
            if mistr not in log:
                all_included = False
                break
        if not all_included:
            continue
        sublogs.append(log)

    return sublogs

def iterate_logs(logs):
    keys = {}
    for key in logs:
        key = key.replace('.err','').replace('.out','')
        if key in keys:
            continue
        keys[key] = 0
        jl = JobLog(key)
        yield jl
        
    
class JobCheck:
    def __init__(self,jl:'JobLog') -> None:
        self.joblog = jl
        self._flag = None
    @property
    def flag(self,):
        if self._flag is None:
            self._flag = self.__call__()
        return self._flag
    def __call__(self,):...
class TerminationCheck(JobCheck):
    def __call__(self,):
        line = self.joblog.errlines[-1].lower()
        if 'err' in line or 'cancelled' in line:
            return False
        return True
                

class JobLog:
    def __init__(self,key,) -> None:
        self.path = os.path.join(JOBS_LOGS,key)
        self.jobid = key.split('_')[-2]
        self.taskid = key.split('_')[-1]
        self._outlines = None
        self._errlines = None
    @property
    def outlines(self,):
        if self._outlines is None:
            with open(self.path+'.out') as f:
                outlines = f.readlines()
            self._outlines = outlines
        return self._outlines
    @property
    def errlines(self,):
        if self._errlines is None:
            with open(self.path+'.err') as f:
                outlines = f.readlines()
            self._errlines = outlines
        return self._errlines

        
def check_by_taskid(taskname,taskid):
    logs = gather_logs(taskname,must_includes=[str(taskid)])
    termjobs = {}
    for joblog in iterate_logs(logs):        
        termcheck = TerminationCheck(joblog)
        termcheck.flag
        termjobs[joblog.path] = termcheck.flag
    one_exists_flag= np.any(list(termjobs.values()))
    return one_exists_flag,termjobs

def check_task(taskname):     
    logs = gather_logs(taskname)
    termjobs = defaultdict(lambda : [])
    for joblog in iterate_logs(logs):        
        termcheck = TerminationCheck(joblog)
        termcheck.flag
        termjobs[joblog.taskid].append(termcheck)
    done_jobs = []
    for taskid,vals in termjobs.items():
        flags = [val.flag for val in vals]        
        if np.any(flags):
            done_jobs.append(int(taskid))
            continue
        
    argsreader = ArgsReader(taskname + '.txt')
    argsreader.read_model_list()
    
    done_jobs = np.array(done_jobs)
    lineflags = np.zeros(len(argsreader.lines),dtype = bool)
    lineflags[done_jobs - 1] = True
    
    inds = np.where(~ lineflags)[0] + 1
    return inds
def to_continuous(partinds:np.ndarray):
    trains = [[],]
    for i in partinds:
        last_train = trains[-1]
        if not bool(last_train):
            last_train.append(i)
            continue
        j= last_train[-1]
        if j == i-1:
            last_train.append(i)
        else:
            trains.append([i])
    if not bool(trains[0]):
        return []
    for i in range(len(trains)):
        a,b = trains[i][0],trains[i][-1]
        trains[i] = [a,b]
    return trains

def array_string(inds):
    trains = to_continuous(inds)
    st = []
    for tr in trains:
        a,b = tr
        if b>a:
            st.append(f'{a}-{b}')
        else:
            st.append(f'{a}')
    return ','.join(st)

def main():
   inds = check_task('filtweights')
   seq_str = array_string(inds)
   print(seq_str)
    
        

if __name__ == '__main__':
    main()