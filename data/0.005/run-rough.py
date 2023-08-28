from lotus import run
from changef90 import new_f90
import os
import sys


def run_bumps(lam):
    new_f90(lam)
    run(1024, f'{cwd}/{1/lam:.0f}')

def run_cont(lam):
    new_f90(lam)
    run(1024, f'{cwd}/lotus-data', f'/scratch/jmom1n15/BumpStab/data/0.005/{1/lam:.0f}/{1/lam:.0f}/')

if __name__ == "__main__":
    cwd = os.getcwd()
    run_cont(1/int(sys.argv[1]))
    
