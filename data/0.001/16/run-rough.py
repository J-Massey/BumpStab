#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from lotus import run
from changef90 import new_f90
from pathlib import Path
import numpy as np

def extract_k():
    with open("lotus.f90","r") as fileSource:
        fileLines = fileSource.readlines()
    txt = fileLines[24]
    return float([s for s in txt.split(' ')][-2][:-1])


def run_bumps(lam):
    new_f90(lam)
    run(1024, f'{cwd}/{1/lam:.0f}')

def run_cont(lam):
    new_f90(lam)
    run(1024, f'{cwd}/{1/lam:.0f}', '/ssdfs/users/jmom1n15/thicc-swimmer/full-body-bumps/256/')

if __name__ == "__main__":
    cwd = Path.cwd()
    run_bumps(1/16)
    