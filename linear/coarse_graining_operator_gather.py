from collections import defaultdict
import logging
from linear.lincol import  CollectParts, SparseVecCollection
from constants.paths import FILTER_WEIGHTS
import os
from utils.slurm import flushed_print
from jobs.jobstats import check_by_taskid

def get_available_parts():
    foldername = os.path.join(FILTER_WEIGHTS)

    file_list = os.listdir(foldername)
    file_list = [fl for fl in file_list if '-parts' in fl]
    return file_list

def parse_line(line):
    x = list(map(int,line.split()))
    k = 'parti,partm,sigma,depth,ncpu'.split(',')
    return dict(tuple(zip(k,x)))
def path_and_bucket_hash(line):
    parsed = parse_line(line)
    npz_name = f'parallel-part-{parsed["parti"]}-{parsed["partm"]}.npz'
    foldername = f'gcm-dpth-{parsed["depth"]}-sgm-{parsed["sigma"]}-parts'
    return foldername,npz_name
def are_buckets_full(path_buckets):
    bucket_flags = {}
    for key,vals in path_buckets.items():
        bucket_flags[key] = True
        for val in vals:
            path = os.path.join(FILTER_WEIGHTS,key,val)
            if not os.path.exists(path):
                bucket_flags[key] = False
            break
    return bucket_flags

def main():
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    from utils.slurm import ArgsReader
    fed = ArgsReader('filtweights.txt')
    fed.read_model_list()
    path_buckets = defaultdict(lambda : [])
    for i,line in enumerate(fed.lines):#fed.lines[:512]):
        hsh,path = path_and_bucket_hash(line)
        path_buckets[hsh].append(path)
    bucket_flags = are_buckets_full(path_buckets)
    for key,val in bucket_flags.items():
        print(key,'\t\t\t bucket is full: ',val)
    full_buckets = [key for key,val in bucket_flags.items() if val]
    
    for foldername,paths in path_buckets.items():
        if foldername not in full_buckets:
            continue
        head = foldername.replace('-parts','')
        logging.info(f'FOLDERNAME = {foldername}')
        paths = [os.path.join(FILTER_WEIGHTS,foldername,path) for path in paths]
        CollectParts.unite_all(FILTER_WEIGHTS,head,paths)
            
if __name__ == '__main__':
    main()
    

