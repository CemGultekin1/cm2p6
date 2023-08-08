import itertools
from data.coords import DEPTHS


def main():
    filename = __file__.replace('.py','') + '.txt'
    f = open(filename, 'w')
    '''
    parti = int(args[0])
    partm = int(args[1])
    sigma = int(args[2])
    depth = int(args[3])
    ncpu = int(args[4])
    '''
    sigmas = [4,8,12,16]    
    numparts = 40
    numcpu = 20
    depths = [int(d) for d in DEPTHS]
    depths = [d for d in depths if d!= 35]
    f.writelines(
        [
            f'{part} {numparts} {sigma} {depth} {numcpu}\n' for sigma,depth,part in itertools.product(sigmas,depths,range(numparts))
        ]
    )
    f.close()

if __name__ == '__main__':
    main()