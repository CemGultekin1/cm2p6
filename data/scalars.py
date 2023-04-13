import xarray as xr
import os
from utils.arguments import options
from constants.paths import SCALARS
from utils.arguments import is_legacy_run,replace_param
def get_scalar_path(args):
    _,scalarid = options(args,key = 'scalars')
    file = os.path.join(SCALARS,scalarid)
    return file
def load_scalars(args:list,):
    scalarargs,_ = options(args,key = 'scalars')
    if scalarargs.legacy_scalars:
        _args = replace_param(args.copy(),'legacy_scalars','False')
        path = get_scalar_path(_args)
        ds = xr.load_dataset(path).load()
        ds['u_scale'] = ds.u_scale*0 + 0.1
        ds['v_scale'] = ds.v_scale*0 + 0.1
        ds['Su_scale'] = ds.Su_scale*0 + 1e-7
        ds['Sv_scale'] = ds.Sv_scale*0 + 1e-7
        return ds
    path = get_scalar_path(args)
    if os.path.exists(path):
        ds = xr.load_dataset(path).load()
    else:
        ds = None
        print('no scalars found!')
    if is_legacy_run(args):
        ds['u_scale'] = ds.u_scale*0 + 0.1
        ds['v_scale'] = ds.v_scale*0 + 0.1
        ds['Su_scale'] = ds.Su_scale*0 + 1e-7
        ds['Sv_scale'] = ds.Sv_scale*0 + 1e-7
    return ds
def save_scalar(args,ds):
    ds.to_netcdf(get_scalar_path(args))
