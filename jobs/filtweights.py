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
    num_parts = 128
    num_cpu = 4
    depths = [int(d) for d in DEPTHS]
    depths = [d for d in depths if d!= 35 and d > 0]
    f.writelines(
        [
            f'{part} {num_parts} {sigma} {depth} {num_cpu}\n' for sigma,depth,part in itertools.product(sigmas,depths,range(num_parts))
        ]
    )
    f.close()

if __name__ == '__main__':
    main()