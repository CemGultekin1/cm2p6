import os
ROOT = '/scratch/cg3306/climate'
ENV_PATH = '/scratch/cg3306/climate/.ext3'
CM2P6_PATH = '/scratch/zanna/data/cm2.6'
REPO_NAME = 'cm2p6'
REPO = os.path.join(ROOT,REPO_NAME)

OUTPUTS_PATH = os.path.join(ROOT,'outputs')
AS15415_PATH = '/scratch/as15415/Data/CM26_Surface_UVT.zarr'
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

MODELS_JSON = os.path.join(OUTPUTS_PATH,'models_info.json')
DATA_JSON = os.path.join(OUTPUTS_PATH,'data_info.json')

# folder_tracker.makedirs()

# for dir in [MODELS,TRAINING_LOGS,OUTPUTS_PATH,EVALS,VIEWS,JOBS_LOGS]:
#     if not os.path.exists(dir):
#         os.makedirs(dir)

def get_view_path(modelid):
    return os.path.join(VIEWS,modelid + '.nc')
def get_eval_path(modelid):
    return os.path.join(EVALS,modelid + '.nc')
def modelsdict_path():
    return MODELS_JSON
def statedict_path(modelid):
    return os.path.join(MODELS,f"{modelid}.pth")
def model_logs_json_path(modelid):
    return os.path.join(TRAINING_LOGS,f"{modelid}.json")

def search_compressed_lsrp_paths(sigma:int,):
    fns = os.listdir(LSRP)
    fns = [fn for fn in fns if f'compressed_conv_weights_{sigma}' in fn]
    spns = []
    for fn in fns:
        lstprt = fn.split('_')[-1]
        span_ = int(lstprt.split('.')[0])
        spns.append(span_)
    return fns,spns
def convolutional_lsrp_weights_path(sigma:int,span :int = -1):
    if span < 0:
        return os.path.join(LSRP,f"conv_weights_{sigma}.nc")
    else:
        return os.path.join(LSRP,f'compressed_conv_weights_{sigma}_{span}.nc')

def all_eval_path():
    return os.path.join(EVALS,'all.nc')
def inverse_coarse_graining_weights_path(sigma:int):
    return os.path.join(LSRP,f'inv_weights_{sigma}.nc')
def coarse_graining_projection_weights_path(sigma:int):
    return os.path.join(LSRP,f'proj_weights_{sigma}.nc')

def average_lowhres_fields_path(sigma:int,isdeep):
    if isdeep:
        return os.path.join(LSRP,f'average_lowhres_{sigma}_3D.nc')
    else:
        return os.path.join(LSRP,f'average_lowhres_{sigma}_surface.nc')

def average_highres_fields_path(isdeep):
    if isdeep:
        return os.path.join(LSRP,f'average_highres_3D.nc')
    else:
        return os.path.join(LSRP,f'average_highres_surface.nc')