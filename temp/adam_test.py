from utils.parallel import get_device
from models.save import save_statedict
from models.load import options,init_architecture,heteroscedasticGaussianLossV2
from utils.parallel import get_device
from data.load import get_data_
import time
import torch
import numpy as np
import sys
from models.nets.gz21 import FullyCNN,SoftPlusTransform
from torch.nn.utils import clip_grad_norm_

from utils.slurm import flushed_print



def load_model(args):
    # archargs,_ = options(args,key = "arch")
    net = FullyCNN()#**archargs.__dict__)
    net.final_transformation = SoftPlusTransform()
    criterion = heteroscedasticGaussianLossV2
    
    runargs,_ = options(args,key = "run")
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    return net,criterion,optimizer,runargs

def main():
    # args = sys.argv[1:]
    from utils.slurm import read_args
    from utils.arguments import replace_params
    args = read_args(3)
    args =replace_params(args,'num_workers','1','disp','1','reset','True','interior','False')

    net,criterion,optimizer,runargs=load_model(args)

    training_generator,=get_data_(args,half_spread = net.spread,torch_flag = True,data_loaders = True,groups = ('train',))
    device=get_device()
    net.to(device)
    net.train()
    for i,(infields,outfields,mask) in enumerate(training_generator):
        infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
        net.zero_grad()
        outputs = net.forward(infields)
        loss = criterion(outputs, outfields, mask)
        
        train_interrupt = dict(
            input = infields,
            output = torch.cat(outputs,dim = 1),
            true_result = outfields,
            mask = mask,
            loss = loss.detach().item(),
            **net.state_dict()
        )
        torch.save(train_interrupt,f'temp/train_interrupt_{i}.pth')
        if i==24:
            raise Exception        
        loss.backward()
        if runargs.clip>0:
            clip_grad_norm_(net.parameters(), runargs.clip)
        optimizer.step()
        flushed_print(loss.item())


if __name__=='__main__':
    main()
