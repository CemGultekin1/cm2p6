import os
import shutil
def main():
    root = '/scratch/zanna/data/cm2.6/coarse_datasets'
    files = os.listdir(root)
    tobedeleted = []
    gaussians = []
    for file in files:
        if 'gaussian' in file:
            gaussians.append(file)
            continue
        if 'gcm' in file:
            continue
        tobedeleted.append(file)
    for file in tobedeleted:
        path = os.path.join(root,file)
        print(f'shutil.rmtree({file})')# shutil.rmtree(path)
    sigmas = {s:[] for s in [4,8,12,16]}
    for file in gaussians:
        sigma = int(file.split('_')[1])
        sigmas[sigma].append(file)
    rename_files = []
    for files in sigmas.values():
        if len(files) == 2: # all are greedy gaussian
            files1 = [(f,f.replace('gaussian','greedy_gaussian')) for f in files]
        else:
            files1 = [(f,f.replace('gaussian','greedy_gaussian')) for f in files if '_.zarr' not in f]
            files1 += [(f,f.replace('_.zarr','.zarr')) for f in files if '_.zarr'  in f]
        rename_files.extend(files1)
    for a,b in rename_files:
        f0 = os.path.join(root,a)
        f1 = os.path.join(root,b)
        os.rename(f0,f1)
        # print(f'os.rename({f0},\n\t,{f1})\n\n')
    return
    folders = os.listdir(root)
    folders = [f for f in folders if '_.zarr' in f]
    folders1 = [f.replace('_.zarr','_gcm.zarr') for f in folders]
    for f0,f1 in zip(folders,folders1):
        f0 = os.path.join(root,f0)
        f1 = os.path.join(root,f1)
        os.rename(f0,f1)
        
if __name__ == '__main__':
    main()