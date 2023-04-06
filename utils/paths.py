import os

file1 = open('paths.txt', 'r')
lines = file1.readlines()
file1.close()

ROOT = lines[0]
ENV_PATH = '/scratch/cg3306/climate/.ext3'
CM2P6_PATH = '/scratch/zanna/data/cm2.6'
OUTPUTS_PATH = '/scratch/cg3306/climate/outputs'
AS15415_PATH = '/scratch/as15415/Data/CM26_Surface_UVT.zarr'

REPO_NAME = 'cm2p6'
REPO = os.path.join(ROOT,REPO_NAME)


class FINE_CM2P6_PATH_Class:
    one_pct_co2 = '1pct_co2'
    zero_co2 = ''
    surface = 'surface'
    deep = 'beneath_surface'
    pzarr = '.zarr'
    delimeter = '_'
    changes_in_path = {
        'sequence' : ('surf','co2'),
        hash((True,False)) :AS15415_PATH
    }
    def lookup_changes(cls,**kwargs):
        keys = tuple([kwargs[key] for key in cls.changes_in_path['sequence']])
        if hash(keys) not in cls.changes_in_path:
            return None
        else:
            return cls.changes_in_path[hash(keys)]
    def __call__(cls,surf:bool,co2:bool)->str:
        val = cls.lookup_changes(surf = surf,co2 = co2)
        if val is not None:
            return val
        filename = []
        surfstr = cls.surface if surf else cls.deep
        filename.append(surfstr)
        if co2:
            filename.append(cls.one_pct_co2)
        filename = cls.delimeter.join(filename)
        filename = filename + cls.pzarr
        return os.path.join(CM2P6_PATH,filename)

FINE_CM2P6_PATH = FINE_CM2P6_PATH_Class()

COARSE_CM2P6_PATH = os.path.join(CM2P6_PATH,'coarse_datasets')


class FolderTracker:
    def __init__(self) -> None:
        self.folders_list = []
    def join(self,*args):
        path = os.path.join(*args)
        if '.' not in path:
            self.folders_list.append(path)
        return path
    def makedirs(self,):
        for folder in self.folders_list:
            if not os.path.exists(folder):
                os.makedirs(folder)
folder_tracker = FolderTracker()

GRID_INFO = folder_tracker.join(CM2P6_PATH,'GFDL_CM2_6_grid.nc')
JOBS = folder_tracker.join(REPO,'jobs')

MODELIDS_JSON = folder_tracker.join(REPO,'modelids.json')
BACKUP_MODELIDS_JSON = folder_tracker.join(REPO,'backup_modelids.json')

JOBS_LOGS = folder_tracker.join(OUTPUTS_PATH,'slurm_logs')
EVALS = folder_tracker.join(OUTPUTS_PATH,'evals')
LEGACY = folder_tracker.join(OUTPUTS_PATH,'legacy')
TIME_LAPSE = folder_tracker.join(OUTPUTS_PATH,'time_lapse')
VIEWS = folder_tracker.join(OUTPUTS_PATH,'views')
SCALARS = folder_tracker.join(OUTPUTS_PATH,'scalars')
PLOTS = folder_tracker.join(OUTPUTS_PATH,'plots')
TEMPORARY_DATA = folder_tracker.join(OUTPUTS_PATH,'data')
FILTER_WEIGHTS = folder_tracker.join(OUTPUTS_PATH,'filter_weights')
ONLINE_MODELS = folder_tracker.join(OUTPUTS_PATH,'online_models')
MODELS = folder_tracker.join(OUTPUTS_PATH,'models')
TRAINING_LOGS = folder_tracker.join(OUTPUTS_PATH,'training_logs')

VIEW_PLOTS = folder_tracker.join(PLOTS,'views')
TIME_LAPSE_PLOTS = folder_tracker.join(PLOTS,'time_lapse')
R2_PLOTS = folder_tracker.join(PLOTS,'r2')
LEGACY_PLOTS = folder_tracker.join(PLOTS,'legacy')

MODELS_JSON = os.path.join(OUTPUTS_PATH,'models_info.json')
DATA_JSON = os.path.join(OUTPUTS_PATH,'data_info.json')


def get_view_path(modelid):
    return os.path.join(VIEWS,modelid + '.nc')
def get_eval_path(modelid):
    return os.path.join(EVALS,modelid + '.nc')
def modelsdict_path():
    return MODELS_JSON
def statedict_path(modelid,legacy:bool = False):
    if not legacy:
        return os.path.join(MODELS,f"{modelid}.pth")
    else:
        modelid = "GZ21"
        return os.path.join(OUTPUTS_PATH,"GZ21.pth")
def model_logs_json_path(modelid):
    return os.path.join(TRAINING_LOGS,f"{modelid}.json")

def all_eval_path():
    return os.path.join(EVALS,'all.nc')