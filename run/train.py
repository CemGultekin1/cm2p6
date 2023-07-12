import itertools
from models.nets.cnn import CNN
from utils.parallel import get_device
from models.save import save_statedict
from models.load import load_model,update_statedict
from utils.parallel import get_device
from data.load import get_data
import torch
import numpy as np
import sys

from utils.slurm import flushed_print
from run.helpers import Timer

def dummy_gpu_fill(infields:torch.Tensor,net:CNN):
    nchan = infields.shape[1]
    shp = (1,nchan,1000,1000)
    x = torch.zeros(shp,).to(infields.device,dtype = torch.float32)
    net(x)
    
def main():
    args = sys.argv[1:]
    # from utils.slurm import read_args
    # from utils.arguments import replace_params
    # args = read_args(17,filename = 'offline_sweep.txt')
    # args = replace_params(args,'num_workers','3','disp','1','reset',\
    #             'True','minibatch','2','depth','110','domain','global')

    modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs = load_model(args)
    print(net)
    flushed_print('torch.cuda.is_available():\t',torch.cuda.is_available())
    flushed_print('runargs:\t',runargs)
    training_generator,val_generator=get_data(args,half_spread = net.spread,torch_flag = True,data_loaders = True,groups = ('train','validation'))
    
    device=get_device()
    net.to(device)
    print(f"using device: {device}")
    flushed_print("epochs started")
    timer = Timer()

    for epoch in range(runargs.epoch,runargs.maxepoch):
        logs['train-loss'].append([])
        tt=0
        net.train()
        timer.start('data')
        for i,(infields,outfields,mask) in enumerate(training_generator):
            if not torch.any(mask>0):
                continue
            net.zero_grad()
            infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
            timer.end('data')
            timer.start('model')
            outputs = net.forward(infields)
            loss = criterion(outputs, outfields, mask)
            
            # import matplotlib.pyplot as plt
            # def plot_samples(vec,name:str):
            #     vec = vec.cpu().numpy()
            #     for i,j in itertools.product(*[range(k) for k in vec.shape[:2]]):
            #         vecij = vec[i,j]
            #         plt.imshow(vecij[::-1,:])
            #         plt.savefig(f'{name}-{i}-{j}.png')
            #         plt.close()
            # plot_samples(infields,'infields')
            # plot_samples(outfields,'outfields')
            # plot_samples(mask,'mask')
            # return
                    
            # mean,prec = mean*mask,prec*mask
            # train_interrupt = dict(
            #     input = infields,
            #     # mean = outputs[0],
            #     # prec = outputs[1],
            #     true_result = outfields,
            #     mask = mask,
            #     loss = loss.detach().item(),
            #     **net.state_dict()
            # )
            # torch.save(train_interrupt,f'temp/train_interrupt_{i}.pth')
            # if i==0:
            #     raise Exception
            
            
            logs['train-loss'][-1].append(loss.item())
    
            loss.backward()
            # if runargs.clip>0:
            #     clip_grad_norm_(net.parameters(), runargs.clip)
            optimizer.step()
            timer.end('model')


            tt+=1
            if runargs.disp > 0 and tt%runargs.disp==0:##np.mean(np.array(logs['train-loss'][-1]))),\
                flushed_print('\t\t\t train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),\
                        '\t ±',\
                        str(np.std(np.array(logs['train-loss'][-1]))))
                # flushed_print(timer)

            timer.start('data')
            if runargs.domain == 'four_regions' or runargs.sigma > 8:
                net.eval()
                dummy_gpu_fill(infields,net)
                net.train()
        timer.reset()
        with torch.set_grad_enabled(False):
            net.eval()
            val_loss=0.
            num_val = 0
            for infields,outfields,mask in val_generator:
                if not torch.any(mask>0):
                    continue
                infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
                outputs = net.forward(infields)
                loss = criterion(outputs, outfields, mask)
                val_loss+=loss.item()
                num_val+=1
                # if num_val == 24:
                #     break

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
