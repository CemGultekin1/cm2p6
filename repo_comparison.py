from models.load import load_model
import torch
import matplotlib.pyplot as plt

def main():
    interrupt_loc0 = '/scratch/cg3306/climate/subgrid/gz21/train_interrupt.pth'
    tf = torch.load(interrupt_loc0, map_location=torch.device('cpu'))
    
    interrupt_loc1 = 'train_interrupt.pth'
    tf1 = torch.load(interrupt_loc1, map_location=torch.device('cpu'))
    
    key = 'true_result'
    for key in 'input output true_result'.split():
        print(tf[key].shape)
        print(tf1[key].shape)
        err = torch.mean(torch.abs(tf[key][0,:2] - tf1[key][0,:2])).item()
        print(f'{key} discrepency = {err}')
        fig,axs = plt.subplots(4,2,figsize = (30,15))
        for i in range(4):
            neg = axs[i,0].imshow(tf[key][i,0].detach().numpy())
            neg = axs[i,1].imshow(tf1[key][i,0].detach().numpy())
        fig.savefig(f'{key}.png')
        plt.close()
    
    # fig.colorbar(neg,cax = axs[0])
    
    return
    print(list(tf.keys()))
    from utils.slurm import read_args
    args = read_args(2)

    modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs=load_model(args)

    for i in range(8):
        w0 = net.nn_layers[2*i].weight
        w1 = tf[f'{2*i}.weight']
        
        b0 = net.nn_layers[2*i].bias
        b1 = tf[f'{2*i}.bias']
        
        werr = torch.sum(torch.abs(w0 - w1)).item()
        berr = torch.sum(torch.abs(b0 - b1)).item()
        print(f'{i} - werr, berr = {werr,berr}')
    
if __name__ == '__main__':
    main()