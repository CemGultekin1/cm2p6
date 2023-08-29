import itertools
import json
import os
import sys
from typing import List
from metrics.geomean import VariableWetMaskedMetrics, WetMaskCollector
from constants.paths import EVALS, all_eval_path
from metrics.modmet import  ModelResultsCollection
from utils.slurm import  PartitionedArgsReader
import logging
from utils.xarray_oper import merge_by_attrs
import xarray as xr
from datetime import date
from options.params import MODEL_PARAMS
logging.basicConfig(level=logging.INFO)
class ModelMetricsGathering(PartitionedArgsReader):
    def __init__(self, model_list_file_name: str, part_id: int,\
                    num_parts: int,model:str = MODEL_PARAMS['model']['choices'][0],\
                        ocean_interior_variation = [1,3,5,7,9,11,15,21],date :str = date.today()):
        super().__init__(model_list_file_name, part_id, num_parts,)
        self.model_results = ModelResultsCollection()
        self.wet_masks  = WetMaskCollector()
        self.ocean_interior_variation = ocean_interior_variation
        self.model = model
        self.date = date
    def read_dataset(self,):
        path = self.target_path()
        ds = self.read(path)
        return ds
    def gather_evals(self,):        
        for i,(index,line) in enumerate(self.iterate_lines()):
            if i != 0:
                continue
            logging.info(f'\t {i}/{len(self)} - {index}')# - {line}')
            # continue
            wmm = VariableWetMaskedMetrics(line.split(),self.wet_masks,ocean_interior = self.ocean_interior_variation)
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
        path = path.replace(ext,'')
        
        if  add:
            return f'{path}{ext}'
        else:
            return path
    def target_file_name(self,index = -1):
        if index< 0:
            index = self.part_id
        return self.extension(f'{self.merged_target_name}-{index}-{self.num_parts}',add = True)
    def target_path(self,index = -1):
        return os.path.join(EVALS,self.target_file_name(index = index))
    @property
    def merged_target_name(self,):
        return self.extension(f'{self.extension(all_eval_path(),remove = True)}-{self.date}-{self.model}',add = True)
    def write(self,x:xr.Dataset,path:str):
        path = self.extension(path,add = True)
        logging.info(f'writing to {path}')
        if bool(x.attrs):
            with open(self.to_json_path(path),'w') as outfile:
                json.dump(x.attrs,outfile)
        x.attrs = {}
        x.to_netcdf(path,mode = 'w')
        logging.info(f'\t\t\t ...success')
    def to_json_path(self,path:str):
        return path.replace('.nc','.json')
    def read(self,path:str)->xr.Dataset:
        path = self.extension(path,add = True)
        logging.info(f'reading from {path}')        
        ds = xr.open_dataset(path,)
        json_path = self.to_json_path(path)
        if os.path.exists(json_path):
            with open(json_path,'r') as infile:
                attrs = json.load(infile)
            
            ds.attrs = attrs
        key = 'training_depth'
        if key in ds.attrs:
            logging.info(f'{key} - {ds.attrs[key]}')
        
        logging.info(f'\t\t\t ...success')
        return ds
        
    @property
    def merged_path(self,):
        return os.path.join(EVALS,self.merged_target_name)
    def merge_write_this_partition(self,):
        ds = self.model_results.merged_dataset()
        logging.info(ds)
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
        logging.info(f'{len(datasets)} datasets were read')
        merged_datasets =merge_by_attrs(datasets,).compute()#compat = 'override'
        self.write(merged_datasets,self.merged_path)
        logging.info(f'\t\t... success.')
        # flag = self.clean_sister_partitions()
        # if not flag:
        #     logging.info(f'Sister files couldn\'t be deleted!')
        #     paths = list(self.iterate_sister_partitions())
        #     logging.info( '\n'.join(paths))
    def clean_sister_partitions(self,):   
        flag = self.all_paths_created()
        if not flag:
            return False     
        for path in self.iterate_sister_partitions():
            json_path = self.to_json_path(path)
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(path):
                logging.info(f'\t\tdeleting {path}...')
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
    
    from utils.slurm import basic_config_logging
    basic_config_logging()
    logging.info(MODEL_PARAMS['model']['choices'])
    model_list_file_names = dict((
        (MODEL_PARAMS['model']['choices'][-1],('lsrpjob.txt','linear')),
        (MODEL_PARAMS['model']['choices'][0],('offline_sweep2.txt','fcnn')),
        )
    )
    model_list,model = model_list_file_names[model_type]
    mmg = ModelMetricsGathering(model_list,\
                part_id,num_parts,\
                model = model,\
                date = '2023-08-27',\
                ocean_interior_variation=[21])
    logging.info(f'mmg.target_path() = {mmg.target_path()}')
    logging.info(f'mmg.target_file_name() = {mmg.target_file_name()}')
    logging.info(f'mmg.merged_target_name = {mmg.merged_target_name}')
    

    # ds = mmg.read_dataset()
    # ds = ds.sel(depth = 0, co2 = 0,filtering = 'gcm',ocean_interior = 21,sigma = 4,stencil = 21,minibatch = 4)
    # logging.info(ds.Su_r2.values)
    # attrs = ds.attrs
    # for key,val in attrs.items():
    #     logging.info(key,val)
    # logging.info(ds)
    # return
    mmg.gather_evals()
    # mmg.merge_write_this_partition()
    # mmg.merge_write_all_partitions()
if __name__=='__main__':
    main()