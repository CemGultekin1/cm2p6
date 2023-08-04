import logging
from data.load import load_xr_dataset
from linear.coarse_graining_inversion import CoarseGrainingInverter
import matplotlib.pyplot as plt
import numpy as np

def main():
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    sigma = 4
    args = f'--sigma {sigma} --depth 0 --filtering gcm'.split()
    cds,_ = load_xr_dataset(args,high_res = False)
    fds,_ = load_xr_dataset(args,high_res = True)

    cu = cds.u.isel(time = 0)
    fu = fds.u.isel(time = 0)
    
    cginv = CoarseGrainingInverter(filtering = 'gcm',depth = 0,sigma = sigma)
    cginv.load()
    # logging.info(f'cginv.mat.shape= {cginv.mat.shape}')
    # logging.info(f'fu.shape= {fu.shape}')
    ccu = cginv.forward_model(fu)
    ccu = ccu.rename({
        'ulat':'lat','ulon':'lon'
    })
    
    fig,axs = plt.subplots(ncols = 3,figsize = (30,10))
    err = np.abs(ccu - cu)
    cu.fillna(0).plot(ax = axs[0])
    ccu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig('coarse_graining_example.png')
    plt.close()
    

if __name__ == '__main__':
    main()