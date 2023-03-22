from models.bank import init_architecture
from models.load import get_statedict, load_model
from models.save import save_statedict,statedict_path,model_logs_json_path
from utils.arguments import options
from params import replace_param
import os
def main():
    args = '--lsrp 0 --depth 0 --sigma 4 --filtering gcm --temperature False --latitude False --domain four_regions --seed 0 --num_workers 16 --disp 50 --batchnorm 1 1 1 1 1 1 1 0 --lossfun MSE --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4'.split()
    
    new_args = replace_param(args.copy(),'model','dfcnn')
    _,modelid1 = options(new_args,key = "model")
    
    
    archargs,_ = options(new_args,key = "arch")
    
    
    _,modelid = options(args,key = "model")
    state_dict,logs = get_statedict(modelid)
    print(modelid1,modelid)
    # save_statedict(modelid1,state_dict,logs)
    statedictfile,logfile = statedict_path(modelid1),model_logs_json_path(modelid1)
    assert not os.path.exists(statedictfile)
    print(list(logs.keys()))
    # net = init_architecture(archargs)


    
if __name__ == '__main__':
    main()