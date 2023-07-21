import os
import matplotlib.pyplot as plt
from constants.paths import TRAINING_LOGS
from utils.slurm import read_args
from utils.arguments import options
import json
import numpy as np
def main():
    # linenum = 62
    title_keys = 'sigma domain temperature filtering depth'.split()
    lines0 = [18,36,56,64,122,123,124,126,127,128,130,131,132,134,135,136,138,139,140,142,143,144,146,147,148,150,151,152,154,155,156,158,159,160,162,163,164,166,167,168,170,171,172,174,175,176,179,184,192,195,196,210,212,216,226,227,228,232,240,242,243,244,248,256,258,259,260,264,272,274,276,280,284,288,289,292,296,300,304]
    lines1 = [18,20,50,52,56,64,178,180,184,192,194,196,200,204,208,210,212,216,220,224,226,228,232,236,240,242,244,248,256,258,260,264,268,272,274,276,280,284,288,292,296,300,304]
    lines = np.unique(lines0 + lines1)
    for linenum in lines:
        args = read_args(linenum,filename = 'offline_sweep2.txt')
        args,modelid = options(args,key = "model")
        # if not (args.filtering == 'gcm' and args.lossfun == 'heteroscedastic'):
        #     continue
            
        print(linenum)
        logfile = os.path.join(TRAINING_LOGS,modelid + '.json')
        if not os.path.exists(logfile):
            continue
        with open(logfile,'r') as f:
            logs = json.load(f)
            
        loss_types = 'train val'.split()
        loss_keys = [k for k in logs.keys() if bool([l for l in loss_types if l in k])]
        lr_key = [k for k in logs.keys() if 'lr' in k][0]
        loss_values = {
            k : logs[k] for k in loss_keys
        }
        
        for k in loss_values:
            if isinstance(loss_values[k][0],list):
                loss_values[k] = [sum(lki)/len(lki) for lki in loss_values[k]]
        for k in loss_values:
            plt.plot(loss_values[k],label = k)
        fin_lr = logs[lr_key][-1]
        title = ', '.join([str(args.__dict__[k]) for k in title_keys] + [f'lr:{fin_lr}'])
        plt.title(title)
        plt.legend()
        plt.savefig(f'training-{linenum}.png')
        plt.close()
if __name__ == '__main__':
    main()