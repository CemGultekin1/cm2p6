from typing import Dict, List
from options.singular import JobMultiplier
from jobs.job_body import create_slurm_job
from utils.slurm import flushed_print
from constants.paths import JOBS, JOBS_LOGS
from data.coords import DEPTHS
from options.slurm  import SlurmJobHeaders,SlurmJobBody
import os
TRAINJOB = 'trainjob'
root = JOBS
NCPU = 8

            
            
def get_base():
    return JobMultiplier(
        num_workers = NCPU,
        disp = 50,
        filtering = 'gaussian',
        batchnorm = tuple([1]*7 + [0]),
        lossfun = ['heteroscedastic','MSE','MVARE'],
    )

class JobGroup:
    job_name:str
    time:str = "10:00:00"
    mem:int = 10
    cpus_per_task:int = 1
    array:List[int] = [1,]
    train_py :str = "run/train.py"
    eval_py : List[str] = [
        f'run/analysis/{x}.py' for x in 'eval distributional legacy_comparison legacy_snapshots'.split()
    ]
    gres :str = "gpu:1"
    def __init__(self) -> None:
        self.jobs = self.gather_jobs()
        self.jobs.adjust_architecture()
        self.jobs = self.jobs.unique()
    @property
    def txt_path(self,):
        txt_file = self.job_name + '.txt'
        return os.path.join(JOBS,txt_file)
    def save(self,):
        lines = [args.line for args in self.jobs.argslist]
        with open(self.txt_path,'w') as f:
            f.write("\n".join(lines))
    def gather_jobs(self,)->JobMultiplier:...
    @property
    def slurm_kwargs(self,)->Dict[str,str]:
        return dict(
            job_name = self.job_name,
            mem = self.mem,
            time = self.time,
            cpus_per_task = self.cpus_per_task,
            array = ','.join([str(x) for x in self.array]),
            gres = self.gres
        )
    @property
    def slurm_path(self,)->str:
        sjh = SlurmJobHeaders(**self.slurm_kwargs)
        return os.path.join(JOBS,sjh.file_name)
    def write_slurm_file(self,):
        sjh = SlurmJobHeaders(**self.slurm_kwargs)
        sjb = SlurmJobBody(read_only=True)
        sjb.add_line(
            f"ARGS=$(sed -n \"$SLURM_ARRAY_TASK_ID\"p {self.txt_path})",out_singularity=True
        )
        sjb.add_line(
            f"python3 {self.train_py} $ARGS"
        )
        for ev in self.eval_py:
            sjb.add_line(
                f"python3 {ev} $ARGS --mode eval"
            )
        sb = "\n".join([str(sjh),str(sjb)])
        
        with open(self.slurm_path,'w') as f:
            f.write(sb)
        model_options = [arg.line for arg in self.jobs.argslist]
        with open(self.txt_path,'w') as f:
            f.write("\n".join(model_options))
        print(f'writing:\t {self.txt_path}\n\t\t{self.slurm_path}')
        # self.jobs.
            
            
class ReproductionJobs(JobGroup):
    job_name :str = 'reproduction'
    def gather_jobs(cls,):
        jm = get_base()
        jm.add(
                filtering = 'gaussian',
                interior = False,
                num_workers = 8,
                min_precision = 0.01,
                clip = 1.,
                scheduler = "MultiStepLR",
                lr = 5e-4,
                batchnorm = tuple([0]*8),
                lossfun = 'heteroscedastic_v2',
                final_activation = 'softplus_with_constant',
                legacy_scalars = True,
                maxepoch = 100,
                domain = 'four_regions',
                gz21 = [True,False],
        )
        return jm
    
class SGDTest(JobGroup):
    mem:int = 60
    array_list :int = [1,2]
    cpus_per_task:int = 8
    time:str = "48:00:00"
    job_name :str = 'sgdtst'
    def gather_jobs(cls,):
        jm = JobMultiplier(
            num_workers = cls.cpus_per_task,
            disp = 50,
            filtering = 'gaussian',
            batchnorm = tuple([1]*7 + [0]),
            lossfun = 'heteroscedastic',
            lr = 1e-2,
            legacy_scalars = True,
            maxepoch = 500,
            optimizer = "SGD",
            momentum = 0.9,
        )
        jm.add(
                interior = False,
                domain = 'four_regions',
        )
        jm.add(
                interior = True,
                domain = 'global',
        )        
        return jm
    
class MainJobs(JobGroup):
    job_name :str = 'mainjobs'
    def gather_jobs(cls,):
        jm = get_base()
        jm.add(
            filtering = 'gaussian',
            interior = False,
            num_workers = 8,
            min_precision = [0.024,0.025],
            domain = 'four_regions',
            lossfun = 'heteroscedastic',
        )
        jm.add(
            filtering = 'gaussian',
            interior = False,
            num_workers = 8,
            domain = 'four_regions',
            lossfun = 'heteroscedastic',
        )
        jm.add(
            filtering = 'gaussian',
            interior = False,
            wet_mask_threshold = 0.5,
            domain = 'global',
            lossfun = 'heteroscedastic',
        )
        jm.add(
            lsrp = 0,     
            depth = 0,
            sigma = [4,8,12,16],
            filtering = 'gcm',
            temperature = False,
            latitude = False,
            domain = ['four_regions','global'],
            seed = list(range(3))
        )
        jm.add(
            lsrp = [0,1],     
            depth = 0,
            sigma = [4,8,12,16],
            filtering = 'gcm',
            temperature = True,
            latitude = [False,True],
            domain = ['four_regions','global'],
            seed = list(range(3))
        )
        jm.add(
            lsrp = [0,1],     
            depth =[int(d) for d in DEPTHS],
            sigma = [4,8,12,16],
            filtering = 'gcm',
            temperature = True,
            latitude = [False,True],
            domain = 'global',
            seed = list(range(3))
        )
        kernel_factors = [float(f)/21. for f in [21,15,11,9,7,5,4,3,2,1]]
        for kf in kernel_factors:
            jm.add(
                lsrp = [0,1],
                depth =0,
                sigma = [4,8,12,16],
                temperature = True,
                latitude = True,
                domain = 'global',
                kernel_factor = kf,
            )
        return jm

def main():
    SGDTest().write_slurm_file()

if __name__ == "__main__":
    main()