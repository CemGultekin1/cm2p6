from argparse import ArgumentTypeError

SCALAR_PARAMS = {
    "sigma" : {"type": int, "choices" : (4,8,12,16)},
    "legacy_scalars" : {"type": bool, "default":False},
}


DATA_PARAMS = {
    "domain" : {"type": str, "choices" : ["four_regions","global","custom"],},
    "temperature" : {"type": bool, "default":False},
    "latitude" : {"type": bool, "default":False},
    "lsrp" :  {"type": int, "default":0},
    "interior" :  {"type": bool, "default":False},
    "wet_mask_threshold" :  {"type": float, "default":0.5},
    "filtering" :  {"type": str, "choices" : ["gcm","gaussian","greedy_gaussian"]},
    "depth" : {"type": float, "default" : 0.},
    "co2" : {"type":bool,"default":False},
    "spacing" : {"type":str,"choices":["asis","long_flat"]}
}




TRAIN_PARAMS = {
    "lr" : {"type": float, "default" : 1e-2},
    "lossfun" : {"type":str, "choices":["heteroscedastic","MSE","MVARE","heteroscedastic_v2"]},
    "weight_decay" : {"type":float, "default": 0.},
    "clip" : {"type":float, "default": -1},
    "scheduler" : {"type":str,"choices":["ReduceLROnPlateau","MultiStepLR"]},
    "optimizer":{"type":str,"choices":["Adam","SGD"]},
    "momentum":{"type":float,"default":0.},
}


EVAL_PARAMS = {
    "modelid" : {"type":str,"default" : ""},
    "dataids" : {"type":str, "nargs":'+',"default" : ""}
}


ARCH_PARAMS = {
    "kernels" : {"type": int, "nargs":'+', "default" : (5,5,3,3,3,3,3,3)},
    "widths" : {"type": int,  "nargs":'+',"default" : (2,128,64,32,32,32,32,32,4)},
    "skipconn" : {"type":int,"nargs":'+',"default":tuple([0]*8)},
    "batchnorm" : {"type":int,"nargs":'+',"default":tuple([1]*8)},
    "seed" : {"type":int,"default":0},
    "model" : {"type":str, "choices":["fcnn","dfcnn","lsrp:0"]},
    "min_precision" : {"type":float, "default":0.},
    "final_activation" : {"type":str, "choices":["softplus","square","softplus_with_constant"]},
    "gz21" : {"type":bool,"default":False},
    "direct_address" : {"type":str,"default":"none"},
}


RUN_PARAMS = {
    "minibatch" : {"type": int, "default" : 1},
    "num_workers" : {"type": int, "default" : 0},
    "prefetch_factor" : {"type": int, "default": 1},
    "maxepoch" : {"type": int, "default" : 500},
    "persistent_workers" : {"type":bool,"default":True},
    "reset":{"type":bool,"default":False},
    "disp" :  {"type":int,"default":-1},
    "mode" : {"type": str, "choices" : ["train","eval","data","scalars","view"],},
    "section" : {"type":int, "nargs": 2, "default":(0,1)},
    "sanity": {"type":bool, "default":False},
}



PARAMS = dict(TRAIN_PARAMS,**DATA_PARAMS,**RUN_PARAMS,**ARCH_PARAMS,**SCALAR_PARAMS)


DATA_PARAMS = dict(DATA_PARAMS,**SCALAR_PARAMS)
TRAIN_PARAMS = dict(TRAIN_PARAMS,**DATA_PARAMS)
MODEL_PARAMS = dict(TRAIN_PARAMS,**ARCH_PARAMS)
RUN_PARAMS = dict(MODEL_PARAMS,**RUN_PARAMS)



USUAL_PARAMS = {
    "depth" : (0,5,55,110,181,330,728),
    "sigma" : (4,8,12,16),
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')




for d in (DATA_PARAMS,ARCH_PARAMS,RUN_PARAMS,SCALAR_PARAMS,TRAIN_PARAMS,MODEL_PARAMS,PARAMS):#
    for key in d:
        if "choices" in d[key]:
            d[key]["default"] = d[key]["choices"][0]
        if d[key]["type"]==bool:
            d[key]["dest"] =key
            d[key]["type"] = str2bool



