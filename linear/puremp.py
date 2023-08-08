import multiprocessing as mp

def init_worker(mps, fps, cut):
    global memorizedPaths, filepaths, cutoff
    global DG

    path = f'file-{mp.current_process()}.txt'
    with open(path,'w') as f:
        f.write(f'hi!')
        
    # print("process initializing", mp.current_process())
    memorizedPaths, filepaths, cutoff = mps, fps, cut
    DG = 1##nx.read_gml("KeggComplete.gml", relabel = True)

def work(item):
    # _all_simple_paths_graph(DG, cutoff, item, memorizedPaths, filepaths)
    print(f'#{mp.current_process()}\t\twork = {item}')

def _all_simple_paths_graph(DG, cutoff, item, memorizedPaths, filepaths):
    pass # print "doing " + str(item)

if __name__ == "__main__":
    pool = mp.Pool(3)
    _ = list(pool.map(work, (0,1,2,3),))