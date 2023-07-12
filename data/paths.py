import os

from utils.arguments import options
from constants.paths import GRID_INFO,FINE_CM2P6_PATH,COARSE_CM2P6_PATH,TEMPORARY_DATA,FILTER_WEIGHTS

def get_filter_weights_location(args,preliminary:bool = False,utgrid = 'u',svd0213=False):
    svd_tag = '' if not svd0213 else '_svd0213'
    assert utgrid in 'u t'.split()
    utgrid_tag = '' if utgrid == 'u' else '_t'
    prms,_ = options(args,key = "run")
    if not os.path.exists(FILTER_WEIGHTS):
        os.makedirs(FILTER_WEIGHTS)
    filtering = prms.filtering.replace('-','_')
    sigma = prms.sigma
    surf_str = 'surface' if prms.depth < 1e-3 else 'beneath_surface'
    if preliminary:
        a,b = prms.section
        filename = f'{filtering}_{surf_str}_{sigma}_{a}_{b}{utgrid_tag}{svd_tag}.nc'
    else:
        filename = f'{filtering}_{surf_str}_{sigma}{utgrid_tag}{svd_tag}.nc'
    return os.path.join(FILTER_WEIGHTS,filename).replace('.nc','_.nc')

def get_learned_deconvolution_location(args,preliminary:bool = False,):
    prms,_ = options(args,key = "run")
    if not os.path.exists(FILTER_WEIGHTS):
        os.makedirs(FILTER_WEIGHTS)
    filtering = prms.filtering.replace('-','_')
    sigma = prms.sigma
    surf_str = 'surface' if prms.depth < 1e-3 else 'beneath_surface'
    if preliminary:
        a,b = prms.section
        filename = f'l2fit_{filtering}_{surf_str}_{sigma}_{a}_{b}.nc'
    else:
        filename = f'l2fit_{filtering}_{surf_str}_{sigma}.nc'
    return os.path.join(FILTER_WEIGHTS,filename)

def get_filename(sigma,depth,co2,filtering,locdir = COARSE_CM2P6_PATH):
    if sigma > 1:
        co2_str = '1pct_co2' if co2 else ''
        surf_str = 'surface' if depth < 1e-3 else 'beneath_surface'
        filename = f'coarse_{sigma}_{surf_str}_{co2_str}.zarr'
        filename = filename.replace('_.zarr','.zarr')
        path = os.path.join(locdir,filename)
        if filtering is not None:
            path = path.replace('.zarr',f'_{filtering}.zarr')
    else:
        path = FINE_CM2P6_PATH(depth < 1e-3,co2)
    return path

def get_high_res_grid_location():
    return GRID_INFO

def get_high_res_data_location(args):
    prms,_ = options(args,key = "data")
    return get_filename(1,prms.depth,prms.co2,prms.filtering)

def get_low_res_data_location(args,silent :bool = False):
    prms,_ = options(args,key = "run")
    
    filename = get_filename(prms.sigma,prms.depth,prms.co2,prms.filtering)
    # if prms.lsrp == 1:
    #     f0 = filename.replace('.zarr','_.zarr').split('/')[-1]
    #     f1 = filename.split('/')[-1]
    #     print('-'*64)
    #     print(f'{f0} = {f1}.replace(".zarr","_.zarr")')
    #     print('-'*64)
    #     filename = filename.replace('.zarr','_.zarr')
    if prms.spacing == 'long_flat':
        filename = filename.replace('.zarr','_flat.zarr')
    if not silent:
        print(filename)
    return filename

def get_preliminary_low_res_data_location(args):
    prms,_ = options(args,key = "run")
    a,b = prms.section
    filename = get_filename(prms.sigma,prms.depth,prms.co2,prms.filtering,locdir = TEMPORARY_DATA)
    if prms.spacing == 'long_flat':
        filename = filename.replace('.zarr','_flat.zarr')
    filename = filename.replace('.zarr',f'_{a}_{b}.zarr')
    return filename

def get_data_address(args):
    drs = os.listdir(COARSE_CM2P6_PATH)
    def searchfor(drs,tokens):
        for token in tokens:
            drs = [f for f in drs if token in f]
        return drs
    def filterout(drs,tokens):
        for token in tokens:
            drs = [f for f in drs if token not in f]
        return drs

    prms,_ = options(args,key = "data")
    includestr = '.zarr'
    excludestr = 'coarse'
    if prms.co2:
        includestr += ' CO2'
    else:
        excludestr += ' CO2'

    if prms.depth <1e-3 :
        includestr += ' surf'
    else:
        assert prms.depth > 0
        includestr += ' 3D'
    if len(excludestr)>0:
        excludestr = excludestr[1:]
    include = includestr.split()
    exclude = excludestr.split()
    drs =searchfor(drs,include)
    drs = filterout(drs,exclude)
    if len(drs) == 1:
        return os.path.join(COARSE_CM2P6_PATH,drs[0])
    else:
        return None