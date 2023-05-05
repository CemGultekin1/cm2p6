from dataclasses import dataclass
from constants.paths import ENV_PATH,JOBS_LOGS,CUDA_SINGULARITY
import os

@dataclass
class SlurmJobHeaders:
    time:str = "1:00:00"
    array:str = "1"
    mem:str = "8GB"
    job_name:str = "none"
    output:str = JOBS_LOGS
    error:str = JOBS_LOGS
    nodes:str = 1
    ntasks_per_node:str = 1
    cpus_per_task :str = 1    
    gres:str = "none"
    def __post_init__(self,):
        self.output = os.path.join(self.output,self.job_name+"_%A_%a.out")
        self.error = os.path.join(self.error,self.job_name+"_%A_%a.err")
        
        keys = tuple(self.__dict__.keys())
        for key in keys:
            self.__dict__[key] = str(self.__dict__[key])
        if "GB" not in self.mem:
            self.mem+="GB"
    def __repr__(self,):        
        stt = [(key,val) for key,val in self.__dict__.items() if key != "gres"]
        if self.gres != "none":
            stt.append(("gres",self.gres))
        st = "\n".join([f"#SBATCH --{key.replace('_','-')}={val}" for key,val in stt])
        return "#!/bin/bash\n" + st
    @property
    def file_name(self,)->str:
        return self.job_name + '.s'
class SlurmJobBody:
    def __init__(self,read_only:bool = False,) -> None:
        rw = "ro" if read_only else "rw"
        self.environment = f"module purge\nsingularity exec --nv --overlay {ENV_PATH}:{rw}\\\n\t {CUDA_SINGULARITY} /bin/bash -c \"\n"
        self.body = ["\t\tsource /ext3/env.sh;"]
        self.date = "echo \"$(date)\""
        self.pre_body = [self.date]
    def add_line(self,line:str,out_singularity:bool = False):
        if not out_singularity:
            if line[-1]!=";":
                line += ";"
            self.body.append(line)
            return
        self.pre_body.append(line)
        
    
    def __repr__(self,):
        return "\n".join(self.pre_body)+"\n"+self.environment + "\n\t\t".join(self.body) +"\n\t\"\n" + self.date

    