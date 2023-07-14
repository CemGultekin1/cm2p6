import os
from data.exceptions import RequestDoesntExist
from run.analysis.eval import lsrp_pred
import torch
from data.load import get_data
from models.load import load_model
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from run.helpers import PrecisionToStandardDeviation
from constants.paths import VIEWS
from utils.xarray_oper import fromtensor, fromtorchdict, fromtorchdict2tensor
import xarray as xr



def main():
    # args = sys.argv[1:]
    from utils.slurm import read_args
    args = read_args(2,)
    from utils.arguments import replace_params
    args = replace_params(args,'mode','view','num_workers','1')
    # args = replace_params(args,'gz21','True','legacy_scalars','True','mode','view','num_workers','1')
    prec2std = PrecisionToStandardDeviation(args)
    
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    # modelid,net = load_old_model(1)
    device = get_device()
    net.to(device)



    
    runargs,_ = options(args,key = "run")
    lsrp_flag = runargs.lsrp
    lsrpid = f'lsrp_{lsrp_flag}'
    assert runargs.mode == "view"
    
    multidatargs = populate_data_options(args,non_static_params=[],domain = 'global',interior = False,wet_mask_threshold = 0.5)
    allstats = []
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        nt_limit = 5
        
        for fields,forcings,field_mask,forcing_mask,field_coords,forcing_coords in test_generator:
            time,depth,co2 = field_coords['time'].item(),field_coords['depth'].item(),field_coords['co2'].item()
            # raise Exception
            print(time,depth,co2)
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'time':[time],'depth':[depth]})
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            # mean = fromtorchdict2tensor(forcings,**kwargs).type(torch.float32)
            
            with torch.set_grad_enabled(False):
                mean,precision =  net.forward(fields_tensor.to(device))
                mean = mean.to("cpu")
                precision = precision.to("cpu")
            std = prec2std(precision)
            # outfields = fromtorchdict2tensor(forcings).type(torch.float32)
            # mask = fromtorchdict2tensor(forcing_mask).type(torch.float32)

            # yhat = mean.numpy()[0]
            # y = outfields.numpy()[0]
            # m = mask.numpy()[0] < 0.5
            # yhat[m] = np.nan
            # y[m] = np.nan
            # prst = lambda y: print(np.mean(y[y==y]),np.std(y[y==y]))
            # prst(y),prst(yhat),prst(fields_tensor.numpy())
            # nchan = yhat.shape[0]
            # import matplotlib.pyplot as plt
            # fig,axs = plt.subplots(nchan,2,figsize = (2*5,nchan*6))
            # for chani in range(nchan):
            #     ax = axs[chani,0]
            #     ax.imshow(y[chani,::-1])
            #     ax = axs[chani,1]
            #     ax.imshow(yhat[chani,::-1])
            # fig.savefig('view_intervention.png')
            # return



            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            predicted_std = fromtensor(std,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
            true_fields = fromtorchdict(fields,field_coords,field_mask,denormalize = True,**kwargs)

            if lsrp_flag:
                (predicted_forcings,lsrp_forcings),true_forcings = lsrp_pred(predicted_forcings,true_forcings)
            def rename_dict(suffix):
                renames = {}
                for key in true_forcings.data_vars.keys():
                    renames[key] = f'{key}_{suffix}'
                return renames

            err_forcings = true_forcings - predicted_forcings
            
            err_forcings = err_forcings.rename(rename_dict('err'))
            predicted_std = predicted_std.rename(rename_dict('std'))
            true_forcings = true_forcings.rename(rename_dict('true'))
            
            predictions_ = xr.merge([predicted_forcings,err_forcings,true_forcings,true_fields,predicted_std])
            if lsrp_flag:
                err_forcings = true_forcings - lsrp_forcings
                err_forcings = err_forcings.rename(rename_dict('err'))
                lsrp_predictions_ = xr.merge([lsrp_forcings,err_forcings,true_forcings,true_fields])
                allstats.append({lsrpid:lsrp_predictions_})
            allstats.append({modelid:predictions_})

            nt+=1
            if nt == nt_limit:
                break
        
    evs = {modelid:xr.merge([alst[modelid] for alst in allstats])}
    if lsrp_flag:
        evs[lsrpid] = xr.merge([alst[lsrpid] for alst in allstats])
    for key in evs:
        filename = os.path.join(VIEWS,key+'.nc')
        print(f'filename:\t{filename}')
        evs[key].sel(lon = slice(-180,180),).to_netcdf(filename,mode = 'w')

            

            






if __name__=='__main__':
    main()
