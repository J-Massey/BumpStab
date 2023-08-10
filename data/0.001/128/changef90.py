#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def new_f90(lam: float):
    with open("lotus.f90","r") as fileSource:
        fileLines = fileSource.readlines()

    fileLines[19] = f"    real, parameter    :: A = 0.1*L, St_d = 0.3, k_x={1/lam:.1f}, k_z={1/lam:.1f}, h_roughness=0.001\n"

    
    if lam >= 1/16:
        fileLines[49] = f"        z = {2*lam/4}\n"
    elif lam >= 1/32:
        fileLines[49] = f"        z = {6*lam/4}\n"
    else:
        fileLines[49] = f"        z = {1/64}\n"

    with open("lotus.f90","w") as fileOutput:
        fileOutput.writelines(fileLines)

