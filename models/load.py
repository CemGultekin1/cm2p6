import json
import os
from models.bank import init_architecture
from models.lossfuns import MSE, heteroscedasticGaussianLoss,MVARE
from models.nets.cnn import LCNN, DoubleCNN, DoubleLCNNWrapper
from params import replace_param
import torch
from utils.arguments import options
from utils.parallel import get_device
from utils.paths import model_logs_json_path, modelsdict_path, statedict_path

def update_statedict(state_dict_,net_,optimizer_,scheduler_,last_model = True):
    if state_dict_ is None:
        state_dict_ = {}
    if last_model:
        state_dict_["last_model"] = net_.state_dict()
    else:
        state_dict_["best_model"] = net_.state_dict()
    state_dict_["optimizer"] = optimizer_.state_dict()
    state_dict_["scheduler"] = scheduler_.state_dict()
    return state_dict_



def get_statedict(modelid):
    statedictfile =  statedict_path(modelid)
    logfile = model_logs_json_path(modelid)
    device = get_device()
    state_dict = None
    logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
    if os.path.exists(statedictfile):
        print(f"model {modelid} state_dict has been found")
        state_dict = torch.load(statedictfile,map_location=torch.device(device))
        if os.path.exists(logfile):
            with open(logfile) as f:
                logs = json.load(f)
    else:
        print(f"model {modelid} state_dict has not been found")
        
        
    return state_dict,logs

def load_modelsdict():
    file = modelsdict_path()
    if os.path.exists(file):
        with open(file) as f:
            modelsdict = json.load(f)
    else:
        modelsdict = {}
    return modelsdict

def load_double_lcnn(state_dict,archargs):
    if len(state_dict) != 2:    
        net = LCNN(archargs.widths, archargs.kernels,True,False, 0)
        net.load_state_dict(state_dict)
        return net
    net = DoubleLCNNWrapper(archargs.widths, archargs.kernels,True,False, 0)
    net1 = LCNN(archargs.widths, archargs.kernels,True,False, 0)
    net.load_state_dict(state_dict['mean'])
    net1.load_state_dict(state_dict['var'])
    net.add_var_part(net1)
    return net
def load_old_model(model_id:int):
    file_location = f'/scratch/cg3306/climate/runs/G-{model_id}/best-model'
    args = '--filtering gaussian --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --batchnorm 1 1 1 1 1 1 1 0'.split()
    archargs,_ = options(args,key = "arch")
    
    state_dict = torch.load(file_location,map_location=torch.device(get_device()))
    net = load_double_lcnn(state_dict,archargs)
    return f'G-{model_id}',net
def get_conditional_mean_state_dict(args):
    new_args = replace_param(args.copy(),'lossfun','MSE')
    new_args = replace_param(new_args,'model','fcnn')
    print(' '.join(new_args))
    _,state_dict,_,_,_,_,_,_ = load_model(new_args)
    return state_dict
def load_model(args):
    archargs,_ = options(args,key = "arch")
    net = init_architecture(archargs)
    modelargs,modelid = options(args,key = "model")
    state_dict,logs = get_statedict(modelid)
    if modelargs.lossfun == "heteroscedastic":
        criterion = heteroscedasticGaussianLoss
    elif modelargs.lossfun == "MSE":
        criterion = MSE
    elif modelargs.lossfun == "MVARE":
        criterion = MVARE
        assert isinstance(net,DoubleCNN)
        state_dict1 = get_conditional_mean_state_dict(args)
        assert state_dict1 is not None
        net.cnn1.load_state_dict(state_dict1["best_model"])
    runargs,_ = options(args,key = "run")
    optimizer = torch.optim.Adam(net.parameters(), lr=runargs.lr)

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2)
    rerun_flag = runargs.reset_model and runargs.mode == 'train'
    if state_dict is not None and not rerun_flag:
        # net.load_state_dict(state_dict["last_model"])
        if runargs.mode == "train":
            net.load_state_dict(state_dict["last_model"])
        else:
            net.load_state_dict(state_dict["best_model"])
        print(f"Loaded an existing model")
        if "optimizer" in state_dict and not runargs.reset_optimizer:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and not runargs.reset_optimizer:
            scheduler.load_state_dict(state_dict["scheduler"])
    else:
        if state_dict is not None:
            print(f"Model was not found")
        elif rerun_flag:
            print(f"Model is re-initiated for rerun")
    if runargs.relog:
        logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
    if len(logs["epoch"])>0:
        epoch = logs["epoch"][-1]
    else:
        epoch = 0
    runargs.epoch = epoch
    return modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs
