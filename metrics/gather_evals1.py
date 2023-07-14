import itertools
import json
import os
import sys
from typing import List
from metrics.geomean import VariableWetMaskedMetrics, WetMaskCollector, WetMaskedMetrics
from models.nets.cnn import kernels2spread
from plots.metrics_ import metrics_dataset
from constants.paths import JOBS, EVALS, all_eval_path
from metrics.modmet import  ModelResultsCollection
from utils.slurm import ArgsReader, PartitionedArgsReader, flushed_print
from utils.xarray_oper import skipna_mean,merge_by_attrs
import xarray as xr
from utils.arguments import  options
import numpy as np
from datetime import date
from options.params import MODEL_PARAMS
def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None,None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid,line
def turn_to_lsrp_models(lines):
    lsrplines = []
    for i in range(len(lines)):
        line = lines[i]
        lsrp_flag,_,lsrpline = get_lsrp_modelid(line.split())
        if lsrp_flag:
            lsrplines.append(lsrpline)
    lsrplines = np.unique(lsrplines).tolist()
    return lsrplines 

def append_statistics(sn:xr.Dataset,):#coordvals):
    modelev = metrics_dataset(sn.sel(lat = slice(-85,85)),dim = [])
    # print(modelev)
    # raise Exception
    modelev = skipna_mean(modelev,dim = ['lat','lon'])
    # for c,v in coordvals.items():
    #     if c not in modelev.coords:
    #         modelev = modelev.expand_dims(dim = {c:v})
    for key in 'Su Sv Stemp'.split():
        r2key = f"{key}_r2"
        msekey = f"{key}_mse"
        sc2key = f"{key}_sc2"
        if r2key not in modelev.data_vars:
            continue
        modelev[r2key] = 1 - modelev[msekey]/modelev[sc2key]
    return modelev
    # print(modelev.Su_r2.values.item(),1- modelev.Su_mse.values.item()/modelev.Su_sc2.values.item())
    # return modelev
def merge_and_save(stats):
    xr.merge(list(stats.values())).to_netcdf(all_eval_path(),mode = 'w')

def kernel_size_fun(kernels):    
    ks = kernels2spread(kernels)*2 + 1
    return ks
def lsrp_gather():
    argsreader = ArgsReader('lsrpjob.txt')
    mrc = ModelResultsCollection()
    wc   = WetMaskCollector()
    for i,line in enumerate(argsreader.iterate_lines()):
        print(line)
        wmm = VariableWetMaskedMetrics(line.split(),wc,stencils = [1,3,5,7,9,11,15,21])
        if not wmm.file_exists():
            continue
        wmm.load() 
        wmm.latlon_reduct()        
        mrc.add_metrics(wmm)
    ds = mrc.merged_dataset()    
    filename = all_eval_path().replace('.nc','20230712_linear.nc')
    ds.to_netcdf(filename,mode = 'w')
def cnn_gather():
    argrr = PartitionedArgsReader('offline_sweep2',)
    models = os.path.join(JOBS,'offline_sweep2.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    mrc = ModelResultsCollection()
    wc   = WetMaskCollector()
    lines = np.array(lines)
    for i,line in enumerate(lines):
        wmm = VariableWetMaskedMetrics(line.split(),wc,stencils = [1,3,5,7,9,11,15,21])
        if not wmm.file_exists():
            continue
        wmm.load()        
        wmm.latlon_reduct()
        wmm.filtering_name_fix()
        print(f'{i}/{len(lines)}')
        wmm.past_coords_to_metric(('filtering',))
        mrc.add_metrics(wmm)
    ds = mrc.merged_dataset()
    filename = all_eval_path().replace('.nc','20230714.nc')
    ds.to_netcdf(filename,mode = 'w')





class ModelMetricsGathering(PartitionedArgsReader):
    def __init__(self, model_list_file_name: str, part_id: int,\
                    num_parts: int,model:str = MODEL_PARAMS['model']['choices'][0],\
                        stencil_variation = [1,3,5,7,9,11,15,21]):
        super().__init__(model_list_file_name, part_id, num_parts)
        self.model_results = ModelResultsCollection()
        self.wet_masks  = WetMaskCollector()
        self.stencil_variation = stencil_variation
        self.model = model
    def gather_evals(self,):
        for i,(index,line) in enumerate(self.iterate_lines()):
            flushed_print(f'\t {i}/{len(self)} - {index}')# - {line}')
            # continue
            wmm = VariableWetMaskedMetrics(line.split(),self.wet_masks,stencils = self.stencil_variation)
            if not wmm.file_exists():
                continue
            wmm.load()
            wmm.latlon_reduct()
            wmm.filtering_name_fix()
            if self.model != 'linear':
                wmm.past_coords_to_metric(('filtering',))
            self.model_results.add_metrics(wmm)    
    def extension(self,path:str,remove :bool= False,add: bool = False):
        ext = '.nc'
        if ext not in path and add:
            return f'{path}{ext}'
        if ext  in path and remove:
            return path.replace(ext,'')
        return path
    def target_file_name(self,index = -1):
        if index< 0:
            index = self.part_id
        return f'{self.merged_target_name}-{index}-{self.num_parts}'
    def target_path(self,index = -1):
        return os.path.join(EVALS,self.target_file_name(index = index))
    @property
    def merged_target_name(self,):
        return f'{self.extension(all_eval_path(),remove = True)}-{date.today()}-{self.model}'
    def write(self,x:xr.Dataset,path:str):
        path = self.extension(path,add = True)
        # turn_to_writable_attrs(x)
        flushed_print(f'writing to {path}')
        if bool(x.attrs):
            with open(path.replace('.nc','.json'),'w') as outfile:
                json.dump(x.attrs,outfile)
        x.attrs = {}
        x.to_netcdf(path,mode = 'w')
        flushed_print(f'\t\t\t ...success')
    def read(self,path:str)->xr.Dataset:
        path = self.extension(path,add = True)
        flushed_print(f'reading from {path}...')        
        ds = xr.open_dataset(path,)
        if os.path.exists(path.replace('.nc','.json')):
            with open(path.replace('.nc','.json'),'r') as infile:
                attrs = json.load(infile)
            ds.attrs = attrs
        flushed_print(f'\t\t\t ...success')
        return ds
        
    @property
    def merged_path(self,):
        return os.path.join(EVALS,self.merged_target_name)
    def merge_write_this_partition(self,):
        ds = self.model_results.merged_dataset()
        target_path = self.target_path()
        self.write(ds,target_path)
    def iterate_sister_partitions(self,):
        for i in range(1,self.num_parts + 1):            
            path = self.target_path(index = i)
            yield self.extension(path,add = True)
    def all_paths_created(self,):
        for path in self.iterate_sister_partitions():
            if not os.path.exists(path):
                return False
        return True        

    def merge_write_all_partitions(self,):
        if not self.all_paths_created():
            return
        datasets = []
        for path in self.iterate_sister_partitions():
            datasets.append(self.read(path).load())

        merged_datasets =merge_by_attrs(datasets).compute()
        self.write(merged_datasets,self.merged_path)
        flushed_print(f'\t\t... success.')
        flag = self.clean_sister_partitions()
        if not flag:
            flushed_print(f'Sister files couldn\'t be deleted!')
            paths = list(self.iterate_sister_partitions())
            flushed_print( '\n'.join(paths))
    def clean_sister_partitions(self,):   
        flag = self.all_paths_created()
        if not flag:
            return False     
        for path in self.iterate_sister_partitions():
            if os.path.exists(path):
                flushed_print(f'\t\tdeleting {path}...')
                try:                    
                    os.remove(path)
                except:
                    return False
            else:
                return False
        return True
       
        
def main():
    args = sys.argv[1:]
    model_type = args[0]
    num_parts = int(args[1])
    part_id = int(args[2])
    
    
    model_list_file_names = dict((
        (MODEL_PARAMS['model']['choices'][-1],('lsrpjob.txt','linear')),
        (MODEL_PARAMS['model']['choices'][0],('offline_sweep3.txt','fcnn')),
        )
    )
    model_list,model = model_list_file_names[model_type]
    mmg = ModelMetricsGathering(model_list,part_id,num_parts,model = model,stencil_variation=[1,])
    mmg.gather_evals()
    mmg.merge_write_this_partition()
    mmg.merge_write_all_partitions()
if __name__=='__main__':
    main()