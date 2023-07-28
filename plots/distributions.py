from copy import deepcopy
import itertools
import os
from typing import List
from run.collect_distributions import KeyedHistograms
from constants.paths import OUTPUTS_PATH
import matplotlib.pyplot as plt
import matplotlib
import  numpy as np
def args2dict(args):
    args = args.split()
    keys = args[::2]
    vals = args[1::2]
    return dict(tuple((key.replace('--',''),val) for key,val in zip(keys,vals)))
def is_dictionary_match(query,d):
    for key in query:
        if key not in d:
            return False
        if d[key] != query[key]:
            return False
    return True
class KeyFinder:
    def __init__(self,argsdictionary) -> None:
        self.dictionary = argsdictionary
        self.vals_dictionary = {}
    def find_all_values(self,):
        for key in self.dictionary.keys():
            kdic = args2dict(key)
            for key_,val_ in kdic.items():
                if key_ not in self.vals_dictionary:
                    self.vals_dictionary[key_] = []
                self.vals_dictionary[key_].append(val_)
        for key_ in self.vals_dictionary.keys():
            self.vals_dictionary[key_] = np.unique(self.vals_dictionary[key_])
    def iterate_slices(self,*leavekeys):
        self.find_all_values()
        dictionary = deepcopy(self.vals_dictionary)
        for lk in leavekeys:
            if lk in dictionary:
                dictionary.pop(lk)
        
        keys = list(dictionary.keys())
        for vals in itertools.product(*dictionary.values()):
            keyvals = dict(tuple(zip(keys,vals)))
            kf = self.request(**keyvals)
            yield keyvals,kf
        
    def request(self,**query):
        subdictionary = {}
        for key,val in self.iter_query_fits(**query):
            subdictionary[key] = val
        return KeyFinder(subdictionary)
            
    def iter_query_fits(self,**query):
        query = {key:str(val) for key,val in query.items()}
        for key,val in self.dictionary.items():
            searchdict = args2dict(key)
            if not is_dictionary_match(query,searchdict):
                continue
            yield key,val
        return None
    def get_element(self,**query):
        for key,val in self.iter_query_fits(**query):
            return val
        return None
def remove_vowels(st):
    st0 = st
    vwls = 'a e o i u _'.split()
    for x in vwls:
        st = st.replace(x,'')
    if len(st) < 3:
        st = st0
    return st.lower()
def to_file_name(keydict):
    sts = []
    for key,val in keydict.items():
        key = remove_vowels(key)
        val = remove_vowels(val)
        sts.extend([key,val])
    return '_'.join(sts)

def plot_for_sigma(sigma):
    path = os.path.join(OUTPUTS_PATH,'distributions_of_all')
    khist = KeyedHistograms.load_from_npz(path)
    kf = KeyFinder(khist.histdict)
    kf = kf.request(sigma = sigma, )
    folder = '/scratch/cg3306/climate/cm2p6/paper_images/adist'
    if not os.path.exists(folder):
        os.makedirs(folder)
    units_dict = dict(
        u = 'm$/$s',
        v = 'm$/$s',
        temp = 'Celsius ($^{\circ}$C)',
        Su = 'm$^2/$s$^4$',
        Sv = 'm$^2/$s$^4$',
        Stemp ='m$^2/$s$^4$'
    )
    matplotlib.rcParams.update({'font.size': 17})
    for key,kf1 in kf.iterate_slices('co2',):        
        x0 = kf1.get_element(co2 = False)
        x1 = kf1.get_element(co2 = True)
        
        absolute =  key['field'] != 'temp'
        if absolute:
            x0 = x0.to_absolutes()
            x1 = x1.to_absolutes()
        fig,ax = plt.subplots(1,1,figsize = (5,5))
        kwargs = dict(
            linewidth = 2
        )
        kwargs_var = (
            dict(
                label = '+0% CO$_2$',
                color = 'blue'
            ),
            dict(
                label = '+1% CO$_2$',
                color = 'red'
            )
        )
        if not absolute:
            plotfun = ax.semilogy
        else:
            plotfun = ax.loglog
        medges = np.empty(0)
        for x,kwargs1 in zip([x0,x1],kwargs_var):
            # cutvals = x.cut_density(1e-6)
            plotfun(x.medges,x.histogram/np.sum(x.histogram),**kwargs,**kwargs1)
            medges = np.concatenate([medges, x.medges])
        ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
        # ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
        vmin,vmax = np.amin(medges),np.amax(medges)
        sc = np.amax([np.abs(vmin),np.abs(vmax)])*0.05
        vmin = vmin - sc
        vmax = vmax + sc
        ax.set_xlim(vmin,vmax)
        ax.legend()
        unit = units_dict[key['field']]
        ax.set_xlabel(unit)
        
        filename = to_file_name(key)
        path = os.path.join(folder,filename + '.png')
        plt.subplots_adjust(bottom=0.15, right=0.98, top=0.95, left= 0.15)
        plt.savefig(path)        
        plt.close()
        print(path)
        # raise Exception
def main():
    for sigma in [8,12,16,]:
        plot_for_sigma(sigma)
    
    
    
    


if __name__ == '__main__':
    main()