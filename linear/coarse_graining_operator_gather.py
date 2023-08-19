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
# def get_associated_file(line_index,line):
#     parsed = parse_line(line)
#     foldername = f'gcm-dpth-{parsed["depth"]}-sgm-{parsed["sigma"]}-parts'
#     root = os.path.join(FILTER_WEIGHTS,foldername)
#     if not os.path.exists(root):
#         return None
#     npz_name = f'parallel-part-{parsed["parti"]}-{parsed["partm"]}.npz'
#     path = os.path.join(root,npz_name)
#     if not os.path.exists(path):
#         return None
#     flag,_ = check_by_taskid('filtweights',line_index)
#     if flag:
#         return path
#     else:
#         return None
def main():
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    from utils.slurm import ArgsReader
    fed = ArgsReader('filtweights.txt')
    fed.read_model_list()
    path_buckets = defaultdict(lambda : [])
    for i,line in enumerate(fed.lines[:512]):
        hsh,path = path_and_bucket_hash(line)
        path_buckets[hsh].append(path)
    for key,val in path_buckets.items():
        print(f'{key}: {len(val)}')
    for foldername,paths in path_buckets.items():
        head = foldername.replace('-parts','')
        paths = [os.path.join(FILTER_WEIGHTS,foldername,path) for path in paths]
        CollectParts.unite_all(FILTER_WEIGHTS,head,paths)
            
if __name__ == '__main__':
    main()
    

