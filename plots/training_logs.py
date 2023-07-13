import os
import matplotlib.pyplot as plt
from constants.paths import TRAINING_LOGS
from utils.slurm import read_args
from utils.arguments import options
import json
def main():
    linenum = 62
    title_keys = 'sigma domain temperature filtering'.split()
    for linenum in [18,20,50,52,56,64,178,180,184,192,194,196,200,204,208,210,212,216,220,224,226,228,232,236,240,242,244,248,256,258,260,264,268,272,274,276,280,284,288,292,296,300,304]:
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
            # print(k,len(loss_values[k]),)
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