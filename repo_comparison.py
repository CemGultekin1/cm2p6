from models.load import load_model
import torch
import matplotlib.pyplot as plt

def main():
    t = 12
    interrupt_loc0 = f'/scratch/cg3306/climate/subgrid/gz21/train_interrupt_{t}_.pth'
    tf = torch.load(interrupt_loc0, map_location=torch.device('cpu'))
    
    # interrupt_loc1 = f'/scratch/cg3306/climate/subgrid/gz21/train_interrupt_{t}.pth'
    interrupt_loc1 = f'train_interrupt_{t}.pth'
    
    tf1 = torch.load(interrupt_loc1, map_location=torch.device('cpu'))
    print(list(tf.keys()))
    print(list(tf1.keys()))
    key = 'true_result'
    for key in 'input output true_result mask'.split():
        print(tf[key].shape)
        print(tf1[key].shape)
        err = torch.mean(torch.abs(tf[key] - tf1[key]),dim = (0,2,3)).detach().numpy()
        print(f'{key} discrepency = {err}')
        nrow = tf[key].shape[0]
        fig,axs = plt.subplots(nrow,2,figsize = (30,15))
        for i in range(nrow):
            neg = axs[i,0].imshow(tf[key][i,0].detach().numpy())
            neg = axs[i,1].imshow(tf1[key][i,0].detach().numpy())
        fig.savefig(f'{key}_{t}.png')
        plt.close()
    print(f"tf['loss']-tf1['loss'] = {tf['loss']-tf1['loss']}")
    return

    for i in range(8):
        w0 = tf1[f'{2*i}.weight']
        w1 = tf[f'{2*i}.weight']
        
        b0 = tf1[f'{2*i}.bias']
        b1 = tf[f'{2*i}.bias']
        
        werr = torch.sum(torch.abs(w0 - w1)).item()
        berr = torch.sum(torch.abs(b0 - b1)).item()
        print(f'{i} - werr, berr = {werr,berr}')
    
if __name__ == '__main__':
    main()