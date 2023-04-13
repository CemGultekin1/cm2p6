from utils.parallel import get_device
from models.save import save_statedict
from models.load import load_model,update_statedict
from utils.parallel import get_device
from data.load import get_data
import time
import torch
import numpy as np
import sys
from torch.nn.utils import clip_grad_norm_

from utils.slurm import flushed_print


class Timer:
    def __init__(self,):
        self.times = {}
    def start(self,label):
        if label not in self.times:
            self.times[label] = []
        self.times[label].append(time.time())
    def end(self,label):
        assert label in self.times
        t1 = self.times[label][-1]
        self.times[label][-1] = time.time() - t1
    def __repr__(self) -> str:
        keys = [f"\t{lbl} : {np.mean(vals[-30:-1])}" for lbl, vals in self.times.items()]
        return "\n".join(keys)
    def reset(self,):
        self.times = {}

def main():
    args = sys.argv[1:]
    # from utils.slurm import read_args
    # from utils.arguments import replace_params
    # args = read_args(2)
    # args =replace_params(args,'num_workers','3','disp','1')

    modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs=load_model(args)
    clip = runargs.clip
    flushed_print('torch.cuda.is_available():\t',torch.cuda.is_available())
    flushed_print('runargs:\t',runargs)
    training_generator,val_generator=get_data(args,half_spread = net.spread,torch_flag = True,data_loaders = True,groups = ('train','validation'))
    
    device=get_device()
    net.to(device)
    print(f"using device: {device}")
    flushed_print("epochs started")
    timer = Timer()
    # print(runargs.epoch,runargs.maxepoch)
    # return
    for epoch in range(runargs.epoch,runargs.maxepoch):
        logs['train-loss'].append([])
        tt=0
        net.train()
        timer.start('data')
        for infields,outfields,mask in training_generator:
            if not torch.any(mask>0):
                continue
            # infields = infields[0].numpy()
            # outfields = outfields[0].numpy()
            # mask = mask[0].numpy()
            # import matplotlib.pyplot as plt
            # def plot_method(field,name):
            #     nchan = field.shape[0]
            #     fig,axs = plt.subplots(nchan,1,figsize = (10,10*nchan))
            #     for i in range(nchan):
            #         print(name,i,np.mean(np.abs(field[i])))
            #         ff = field[i]
            #         ff = ff[::-1]
            #         axs[i].imshow(ff,cmap = 'bwr')
            #     fig.savefig(name)
            #     plt.close()
            # plot_method(np.log10(np.abs(infields)),'log10_infields.png')
            # plot_method(np.log10(np.abs(outfields)),'log10_outfields.png')
            # plot_method(infields,'infields.png')
            # plot_method(outfields,'outfields.png')
            # plot_method(mask,'mask.png')
            # return
            infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
            timer.end('data')
            timer.start('model')
            outputs = net.forward(infields)

            # with torch.set_grad_enabled(False):
            #     net.eval()
            #     outputs = net.forward(infields)
            # mean,cond_var = outputs
            # yhat1 = cond_var.to("cpu").numpy()[0]
            # yhat = mean.to("cpu").numpy()[0]
            # y = outfields.to("cpu").numpy()[0]
            # m = mask.to("cpu").numpy()[0] < 0.5
            # # yhat[m] = np.nan
            # # y[m] = np.nan
            # # yhat1[m] = np.nan
            
            # nchan = yhat.shape[0]
            # import matplotlib.pyplot as plt
            # fig,axs = plt.subplots(nchan,3,figsize = (3*5,nchan*6))
            # kwargs = dict(cmap = 'bwr')
            # for chani in range(nchan):
            #     for j,zh in zip(range(3),[y,yhat,yhat1]):
            #         ax = axs[chani,j]
            #         pos = ax.imshow(zh[chani,::-1],**kwargs)
            #         fig.colorbar(pos,ax = ax)
            # fig.savefig('train_intervention.png')
            # return

            

            
            loss = criterion(outputs, outfields, mask)
            logs['train-loss'][-1].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if clip>0:
                clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            timer.end('model')


            tt+=1
            if runargs.disp > 0 and tt%runargs.disp==0:
                flushed_print('\t\t\t train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),\
                        '\t ±',\
                        str(np.std(np.array(logs['train-loss'][-1]))))
                flushed_print(timer)
            timer.start('data')

        timer.reset()
        with torch.set_grad_enabled(False):
            net.eval()
            val_loss=0.
            num_val=0
            for infields,outfields,mask in val_generator:
                if not torch.any(mask>0):
                    continue
                infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
                outputs = net.forward(infields)
                loss = criterion(outputs, outfields, mask)
                val_loss+=loss.item()
                num_val+=1
        logs['val-loss'].append(val_loss/num_val)
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step(logs['val-loss'][-1])



        if len(logs['epoch'])>0:
            logs['epoch'].append(epoch)
        else:
            logs['epoch'].append(0)

        flushed_print('#epoch ',str(logs['epoch'][-1]),' ',\
                    ' val-loss: ',str(logs['val-loss'][-1]),\
                    ' train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),\
                    ' lr: ',str(logs['lr'][-1]))

        state_dict = update_statedict(state_dict,net,optimizer,scheduler,last_model = True)
        if np.amin(logs['val-loss']) == logs['val-loss'][-1]:
            state_dict = update_statedict(state_dict,net,optimizer,scheduler,last_model = False)
        save_statedict(modelid,state_dict,logs)
        if logs['lr'][-1]<1e-7:
            break


if __name__=='__main__':
    main()
