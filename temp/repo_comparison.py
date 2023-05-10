from models.load import load_model
import torch
import matplotlib.pyplot as plt
import os 
import numpy as np
def main():
    t = 0
    interrupt_loc0 =  f'temp/train_interrupt_{t}.pth'
    tf = torch.load(interrupt_loc0, map_location=torch.device('cpu'))
    imgs_folder = 'temp/imgs'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
    # interrupt_loc1 = f'/scratch/cg3306/climate/subgrid/gz21/train_interrupt_{t}.pth'
    interrupt_loc1 = f'temp/train_interrupt_{t}.pth'
    
    tf1 = torch.load(interrupt_loc1, map_location=torch.device('cpu'))
    # print(list(tf.keys()))
    # print(list(tf1.keys()))
    key = 'true_result'
    for key in 'input mean prec true_result mask'.split():
        
        
        # print(f"{key} = {tf[key].shape}")
        # print(f"{key} = {tf1[key].shape}")
        # print()
        # continue
        mask = tf1['mask'].detach().numpy()
        mask[mask == 0] = np.nan
        # err = torch.mean(torch.abs(tf[key]*mask - tf1[key]*mask),dim = (0,2,3)).detach().numpy()
        # print(f'{key} discrepency = {err}')
        nrow = 4#tf[key].shape[0]
        fig,axs = plt.subplots(nrow,2,figsize = (30,15))
        for j in range(min(tf[key].shape[1],tf[key].shape[1])):
            for i in range(nrow):
                a = tf[key].detach().numpy()
                # b = tf1[key].detach().numpy()
                if key != 'input':
                    a = a*mask
                    # b = b*mask
                print(f'{key}\t:{a.shape}')
                # diff = np.log10(np.abs(b-a))
                neg = axs[i,j].imshow(a[i,j,::-1])
                # neg = axs[i,j].imshow(b[i,j,::-1])
                # neg = axs.imshow(diff[j,::-1])
                # fig.colorbar(neg,ax = axs)
            axs[i,j].set_title(interrupt_loc0)
            # axs[1,j].set_title(interrupt_loc1)
            # axs[0,2].set_title('diff')
            # axs.set_title('diff')
        fig.savefig(f'{imgs_folder}/{key}_{t}.png')
        plt.close()
    print(f"tf['loss']-tf1['loss'] = {tf['loss']-tf1['loss']}")
    # return

    for i in range(21):
        if f'{i}.weight' not in tf1:
            continue
        w0 = tf1[f'{i}.weight']
        w1 = tf[f'{i}.weight']
        
        b0 = tf1[f'{i}.bias']
        b1 = tf[f'{i}.bias']
        
        werr = torch.sum(torch.abs(w0 - w1)).item()
        berr = torch.sum(torch.abs(b0 - b1)).item()
        print(f'{i} - werr, berr = {werr,berr}')
    
if __name__ == '__main__':
    main()