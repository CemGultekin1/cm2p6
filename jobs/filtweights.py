import itertools
from constants.paths import JOBS


def main():
    filename = __file__.replace('.py','') + '.txt'
    f = open(filename, 'w')
    sigmas = [4,8,12,16]    
    numparts = 40
    numcpu = 20
    f.writelines(
        [
            f'{part} {numparts} {sigma} {numcpu}\n' for sigma,part in itertools.product(sigmas,range(numparts))
        ]
    )
    f.close()

if __name__ == '__main__':
    main()