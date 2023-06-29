from typing import Dict, Generator, List, Tuple
import xarray as xr
# import matplotlib.pyplot as plt
import itertools
import os
from dataclasses import dataclass
@dataclass
class DatasetPartition:
    filename:str = ''
    root:str = ''
    co2:bool = False
    coarse_graining_factor:int = 1
    beneath_surface:bool = False
    filtering:str = 'gaussian'
    index:int = 0
    total:int = 1
    
    @staticmethod
    def from_filename(filename:str,root:str):
        try:
            feats = filename.split('_')
            flist = [filename,root]
            flist.append('co2' in feats)
            if 'coarse' in filename:
                flist.append(int(feats[1]))
                flist.append('beneath' in feats)
                for filtering in 'gaussian gcm'.split():
                    if filtering in filename:
                        flist.append(filtering)
                        break
                if feats[-2].isdigit():
                    flist.append(int(feats[-2]))
                    flist.append(int(feats[-1].split('.')[0]))
        except:
            print(filename)
            raise Exception
        return DatasetPartition(*flist)
    def __hash__(self,):
        return hash(tuple(self.__dict__.values()))
    def __repr__(self) -> str:
        return str(list(f'{k} = {v}' for k,v in self.__dict__.items() if k!='root'))
    def satisfy_flag(self,**filter_dict)->bool:
        for key,val in filter_dict.items():
            if key in self.__dict__:
                if val != self.__dict__[key]:
                    return False
            else:
                return False
        return True
    @property
    def path(self,):
        return os.path.join(self.root,self.filename)
    def get_dataset(self,):
        return xr.open_zarr(self.path)
    def exclude_features(self,excf:List[str]):
        dc = {k:v for k,v in self.__dict__.items() if k not in excf}
        return DatasetPartition(*dc)
class DatasetPartitionProduct:
    def __init__(self,**product_dict) -> None:
        dp = DatasetPartition()
        multip_values = []
        multip_keys = []
        for key in dp.__dict__.keys():
            if key not in product_dict:
                continue
            vals = product_dict[key]
            multip_values.append(vals)
            multip_keys.append(key)
        self.multip_values = multip_values
        self.multip_keys = multip_keys
    def iterate_products(self,):
        for pdtp in itertools.product(*self.multip_values):            
            yield {mk:pd for mk,pd in zip(self.multip_keys,pdtp)}
class DatasetPartitionsDictionary:
    def __init__(self,root:str = '', files :List[str] = []) -> None:
        if not bool(files):
            files = os.listdir(root)
            
        self.search_dict :Dict[int,DatasetPartition] ={}
        self.files = files
        for file in files:
            dp = DatasetPartition.from_filename(file,root)
            self.search_dict[hash(dp)] = dp
    @staticmethod
    def init_from_dictionary(search_dict):
        dpd = DatasetPartitionsDictionary.__new__(DatasetPartitionsDictionary,)
        dpd.__dict__['search_dict'] = search_dict
        return dpd
    def is_in_dictionary(self,**search_entry):
        dp = DatasetPartition(**search_entry)
        hsdp = hash(dp)
        return hsdp in self.search_dict
    def __repr__(self) -> str:
        stt = []
        for hdp,dp in self.search_dict.items():
            stt.append(f'{hdp}:\t\t{dp}')
        return '\n'.join(stt)
    def iterate_filtered(self,**filter_dict):
        for hdp,dp in self.search_dict.items():
            if dp.satisfy_flag(**filter_dict):
                yield hdp,dp
    def subdictionary(self,**filter_dict):
        subdict = {}
        for hdp,dp in self.iterate_filtered(**filter_dict):
            subdict[hdp] = dp
        return DatasetPartitionsDictionary.init_from_dictionary(subdict)
    def run_production(self,**product_dict):
        dpp = DatasetPartitionProduct(**product_dict)
        for pdict in dpp.iterate_products():
            subdict = self.subdictionary(**pdict)
            yield subdict,DatasetPartition(**pdict)
    def iterate_datasets(self,):
        for _,dp in self.search_dict.items():
            yield dp,dp.get_dataset()
    def collect_parameters(self,*pnames):
        val_dict = {pn:[] for pn in pnames}
        for pd,ds in self.iterate_datasets():
            for pname in pnames:
                val_dict[pname].append(ds[pname].values)
        return tuple(val_dict.values())
    @property
    def filenames(self,)->List[str]:
        fn = []
        for dp in self.search_dict.values():
            fn.append(dp.filename)
        return fn
class DatasetFiltering(DatasetPartitionsDictionary):
    def __init__(self, root: str = '', files: List[str] = [],excluded_features :List[str] = ['index','total','filename','root']) -> None:
        super().__init__(root, files)
        self.excluded_features = excluded_features
    def filter_out_existing(self, generator:Generator[Tuple[DatasetPartitionsDictionary,DatasetPartition],None,None]):
        for subdict,dp in generator:
            newdp = dp.exclude_features(self.excluded_features)
            newdp_hash =  hash(newdp)
            if newdp_hash in self.search_dict:
                continue
            yield subdict,dp
def main():
    root = '/scratch/cg3306/climate/outputs/data/'
    dpd = DatasetPartitionsDictionary(root)
    
    root = '/scratch/zanna/data/cm2.6/coarse_datasets/'
    existing = DatasetFiltering(root)
    production_dictionary = dict(
        co2 = [True,],
        coarse_graining_factor = [4,8,12,16],
        beneath_surface = [False],
        filtering = ['gaussian','gcm']
    )
    generator = dpd.run_production(**production_dictionary)
    generator = existing.filter_out_existing(generator)
    for sdpd,dsp in generator:        
        time_values, = sdpd.collect_parameters('time')
        num_time_values = [len(tv) for tv in time_values]
        total_time_points = sum(num_time_values)
        print(f'{dsp}\t\t\t: {total_time_points}')


if __name__ == '__main__':
    main()
    
