from models.load import load_model
import torch
import matplotlib.pyplot as plt
import numpy as np
def main():
    t = 0
    interrupt_loc1 = f'train_interrupt_{t}.pth'
    
    tf1 = torch.load(interrupt_loc1, map_location=torch.device('cpu'))
    # print(list(tf1.keys()))
    # maskedout = tf1['output'][:,0]*tf1['mask'][:,0]
    # maskedout = maskedout.reshape([maskedout.shape[0],-1])    
    # for md in maskedout:
    #     vals = np.sort(md.detach().numpy())
    #     dvals = vals[1:] - vals[:-1]
    #     plt.semilogy(dvals,'*')
    # plt.savefig('dvals.png')
    # return
    key = 'true_result'
    for key in 'input output true_result mask'.split():
        var_ = tf1[key]
        nrow = var_.shape[0]
        fig,axs = plt.subplots(nrow,1,figsize = (30,15))
        for i in range(nrow):
            if nrow > 1:
                ax = axs[i]
            else:
                ax = axs
            neg = ax.imshow(var_[i,0].detach().numpy())
        fig.savefig(f'{key}_{t}.png')
        plt.close()
    
if __name__ == '__main__':
    main()