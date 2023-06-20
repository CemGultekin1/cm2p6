import os
import sys
from typing import List, Tuple
from data.exceptions import RequestDoesntExist
from models.nets.cnn import CNN
import torch
from data.load import get_data
from models.load import load_model
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
import numpy as np
from constants.paths import SALIENCY
from utils.xarray import fromtorchdict2tensor
import xarray as xr
import itertools
class MovingWindowFeeder:
    def __init__(self,input_vec:torch.Tensor,mask:torch.Tensor,span:int) -> None:
        self.input_vec = input_vec
        self.mask = mask
        self.shape = input_vec.shape
        self.span = span
        lat,lon = [(span,input_vec.shape[i]-span,2*span + 1) for i in (2,3)]
        # lat,lon = [(span,input_vec.shape[i]-span,) for i in (2,3)]
        self.location_counter = itertools.product(range(*lat),range(*lon))
    def __iter__(self,):
        return self
    def __next__(self,):
        full_water_found = False
        while not full_water_found:
            i,j = self.location_counter.__next__()
            full_water_found = self.mask[0,0,i-self.span,j-self.span] == 1
                
        slc0,slc1 = [slice(k-self.span,k+self.span + 1) for k in (i,j)]
        inputwin = self.input_vec[:,:,slc0,slc1]
        return inputwin,(i,j)
class MovingWindowConsumer:
    def __init__(self,mwf:MovingWindowFeeder,) -> None:
        self.mowife = mwf
        self.shape = mwf.shape
        self.span = mwf.span
    def go_through_windows(self,):
        for win,ij in self.mowife:
            self.update(win,ij)
    def update(self,win:torch.Tensor,ij:Tuple[int,int]):...
       
class CNNSaliencyOnWindow:
    def __init__(self,net:CNN,) -> None:
        net.eval()
        self.net = net
    def __call__(self,inputvec:torch.Tensor,):
        inputvec.requires_grad = True
        
        inputvec.grad = None
        outputs,_ = self.net(inputvec)
        
        shp = list(inputvec.shape)
        shp[1] = outputs.shape[1]
        grad_energy = torch.zeros(*shp,dtype = torch.float32)
        
        for i in range(outputs.shape[1]):
            if i > 0:
                inputvec.grad = None
                outputs,_ = self.net(inputvec)
            outputs[0,i,0,0].backward()
            grad_energy[0,i] = torch.mean(inputvec.grad**2,dim = 1,keepdim = True)
        return grad_energy.numpy()
    
class CNNSaliencyRadius(CNNSaliencyOnWindow):
    @staticmethod
    def energy_decay(img:np.ndarray)->Tuple[list,list]:
        n = img.shape[0]
        energies = []
        keys = []
        tot = np.sum(img)
        for i,j in itertools.product(*[range(n)]*2):
            d = np.maximum(np.abs(i - n//2),np.abs(j - n//2))
            energies.append(img[i,j])
            # energies.append(img[i,j])
            keys.append(d)
        
        npkeys = np.array(keys)
        npenergies = np.array(energies)
        ranked_energy = []
        for i in range(n//2):
            relen = np.sum(npenergies[npkeys >= i])/tot
            ranked_energy.append(np.log10(relen))
        keys = np.arange(n//2).tolist()
        # import matplotlib.pyplot as  plt
        # plt.semilogy(npkeys,npenergies,'*')
        # plt.savefig('dummy.png')
        # raise Exception
        return keys,ranked_energy
            
    def __call__(self, inputvec: torch.Tensor):
        grad_win = super().__call__(inputvec)
        grad_win = grad_win[0]
        energies = []
        keys = []
        for i in range(grad_win.shape[0]):
            keys_,energies_ = self.energy_decay(grad_win[i])
            energies.append(energies_)
            keys.append(keys_)
        return tuple((k,e) for k,e in zip(keys,energies))
class KeyedAdaptiveHistogram:
    def __init__(self,num_samples:int = 16,nbins = 64) -> None:
        self.num_samples = num_samples
        self.samples = []
        self.keys = []
        self.counter = 0
        self.nbins = nbins
        self.histogram = None
        self.xmin = None
        self.xmax = None  
    def put_to_histogram(self,keys_:List[int],samples_:List[int],):
        dx = (self.xmax - self.xmin)/self.nbins
        for key,sample in zip(keys_,samples_):
            loc = np.floor((sample - self.xmin)/dx).astype(int)
            if loc < 0:
                continue
            if loc > self.nbins - 1:
                continue
            # loc = np.minimum(self.nbins - 1,loc)
            # loc = np.maximum(0,loc)
            i = self.keys.index(key)
            self.histogram[loc,i] += 1
    def process_sample(self,keys_:List[int],samples_:List[int],):
        if self.counter >= self.num_samples:
            self.put_to_histogram(keys_,samples_)
            self.counter += len(keys_)
            return
        self.keys.extend(keys_)
        self.samples.extend(samples_)
        self.counter += len(keys_)
        if self.counter >= self.num_samples and self.histogram  is None:
            self.xmin = np.amin(self.samples)
            self.xmax = np.amax(self.samples)
            self.keys = np.unique(self.keys).tolist()
            nkeys = len(self.keys)
            self.histogram = np.zeros((self.nbins,nkeys))
    def to_xarray(self,varname:str,coord_name:str = 'radius'):        
        edges  = np.linspace(self.xmin,self.xmax,self.nbins + 1)
        midpts = (edges[1:] + edges[:-1])/2
        coords = {coord_name:np.arange(self.histogram.shape[1]),\
            'midpts':midpts,}#np.power(10.,midpts),}
        cdf = np.cumsum(self.histogram,axis = 0)
        cdf = cdf/cdf[-1:,:]
        data_vars = {
            varname : (['midpts',coord_name],cdf),
        }
        return xr.Dataset(
            data_vars,coords
        )[varname]
        
class MultiAdaptiveHistograms:
    def __init__(self,num_samples:int = 16,nbins = 64,size = 3) -> None:
        self.size = size
        self.ahists  = [KeyedAdaptiveHistogram(num_samples=num_samples,nbins=nbins) for _ in range(size)]
    def process_samples(self,keysample_pairs):
        for ks,ahist in zip(keysample_pairs,self.ahists):
            ahist.process_sample(*ks)
    @property
    def extremums(self,):
        exts = np.zeros((self.size,2))
        for i,ahi in enumerate(self.ahists):
            exts[i] = ahi.xmin,ahi.xmax
        return exts
    @property
    def base_histogram(self,):
        if np.any([ahist.histogram is None for ahist in self.ahists]):
            return None
        hist =  np.stack([ahist.histogram for ahist in self.ahists],axis = 0)
        return hist
    @property
    def histogram(self,):
        hist = self.base_histogram
        if hist is None:
            return None
        axes =list(range(len(hist.shape)))
        reduc_axes = tuple(axes[1:])
        return hist/np.sum(hist,axis = reduc_axes,keepdims = True)

    @property
    def key_conditional_histogram(self,):
        hist = self.base_histogram
        if hist is None:
            return None
        return hist/np.sum(hist,axis = 1,keepdims = True)
    def save_histogram(self,filename:str):
        self_histogram = self.base_histogram
        if self_histogram  is None:
            return
        svdict = dict(histograms = self_histogram,
                      extremums = self.extremums)
        np.savez(filename,**svdict)
    def load_from_file(self,filename:str):
        svdict = np.load(filename)
        histograms = svdict['histograms']
        extremums = svdict['extremums']
        self.size = histograms.shape[0]
        nbins = histograms.shape[1]
        self.ahists  = [KeyedAdaptiveHistogram(nbins=nbins,) for _ in range(self.size)]
        for ahi,hist,ext in zip(self.ahists,histograms,extremums):
            ahi.histogram = hist
            ahi.xmin,ahi.xmax = ext
        

class CNNSaliencyHistogramFeeder(MovingWindowConsumer,):
    def __init__(self,net:CNN,inputvec:torch.Tensor,mask:torch.Tensor,adaptive_histogram:MultiAdaptiveHistograms) -> None:
        self.cnn_saliency = CNNSaliencyRadius(net)        
        mwf = MovingWindowFeeder(inputvec,mask,net.spread)        
        super().__init__(mwf)
        self.adaptive_histogram = adaptive_histogram
    def update(self, win: torch.Tensor, ij: Tuple[int, int]):
        energy = self.cnn_saliency(win)
        self.adaptive_histogram.process_samples(energy)
    
def save(modelid,predicted_forcings,predicted_std):
    filename = os.path.join(SALIENCY,modelid+'.nc')
    if not os.path.exists(SALIENCY):
        os.makedirs(SALIENCY)
    def add_tag_to_names(ds:xr.Dataset,tag:str):
        rename_dict = {
            key: f'{key}_{tag}' for key in ds.data_vars.keys()
        }
        return ds.rename(rename_dict)
        # return ds
    predicted_forcings = add_tag_to_names(predicted_forcings,'_mean')
    predicted_std =  add_tag_to_names(predicted_std,'_std')
    stats = xr.merge([predicted_forcings,predicted_std])
    stats.to_netcdf(filename)
def plot_saliency_histogram(adaptive_histogram:MultiAdaptiveHistograms,tag:str):
    if adaptive_histogram.histogram is None:
        return
    import matplotlib.pyplot as plt
    ds = adaptive_histogram.ahists[0].to_xarray('Su')
    # chist = adaptive_histogram.key_conditional_histogram
    # hists = [hist,chist]
    # ncols = hist.shape[0]
    fig,axs = plt.subplots(1,1)#,figsize= (6*ncols,6))
    # for (i,hist),j in itertools.product(zip(range(2),hists),range(ncols)):
    #     ax = axs[i,j]        
    # axs.imshow(hist[0]>0)
    np.log10(ds['Su']).plot(ax = axs)
    fig.savefig(f'histogram_{tag}.png')
    plt.close()
def main():
    args = sys.argv[1:]
    
    # from utils.slurm import read_args    
    # args = read_args(4,filename = 'saliency.txt')
    from utils.arguments import replace_params
    args = replace_params(args,'num_workers','1','disp','1','reset','False','minibatch','1','mode','eval')
    
    
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    net.to(device)



    
    runargs,_ = options(args,key = "run")
    lsrp_flag = runargs.lsrp
    lsrpid = f'lsrp_{lsrp_flag}'
    assert runargs.mode == "eval"
    
    multidatargs = populate_data_options(args,non_static_params=[],domain = 'global',interior = False,wet_mask_threshold = 0.5)
    adaptive_histogram = MultiAdaptiveHistograms(num_samples=1024,nbins = 512)
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('all',),shuffle = True)
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        # nt_limit = 16
        for fields,_,forcing_mask,field_coords,forcing_coords in test_generator:
            time,depth,co2 = field_coords['time'].item(),field_coords['depth'].item(),field_coords['co2'].item()

            print(time,depth,co2)
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'time':[time],'depth':[depth]},\
                masking = False)
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            fmtensor = fromtorchdict2tensor(forcing_mask).type(torch.float32)
            CNNSaliencyHistogramFeeder(net,fields_tensor,fmtensor,adaptive_histogram).go_through_windows()
           
            pop_keys = ['abs_lat', 'abs_lat_scale', 'sign_lat', 'sign_lat_scale']
            for k in pop_keys:
                if k in fields:
                    fields.pop(k)

            nt+=1
            if nt % 20:
                adaptive_histogram.save_histogram(os.path.join(SALIENCY,modelid))
                # plot_saliency_histogram(adaptive_histogram,str(nt))
        adaptive_histogram.save_histogram(os.path.join(SALIENCY,modelid))
        
    

            

            






if __name__=='__main__':
    main()
