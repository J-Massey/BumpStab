import os
import time
from tkinter import Tcl
import numpy as np
from lotusvis.flow_field import ReadIn


def fluid_snap(sim_dir: str):
    os.system(f"mkdir -p {os.path.abspath(sim_dir + '/..')}/uvp")
    fsim = ReadIn(sim_dir, 'fluAv', 4096, ext="vti")
    fsnap = fsim.snaps(save=True, part=False, save_path=f"{os.path.abspath(sim_dir + '/..')}/uvp")
    np.save(f"{os.path.abspath(sim_dir + '/..')}/uvp/flow.npy", fsnap)
    return fsnap

def body_snap(sim_dir: str):
    os.system(f"mkdir -p {os.path.abspath(sim_dir + '/..')}/uvp")
    bsim = ReadIn(sim_dir, 'bodAv', 4096, ext="vti")
    bsnap = bsim.snaps(save=True, part=False, save_path=f"{os.path.abspath(sim_dir + '/..')}/uvp")
    np.save(f"{os.path.abspath(sim_dir + '/..')}/uvp/body.npy", bsnap)
    return bsnap.shape


if __name__ == "__main__":
    sim_dir = f"{os.getcwd()}/data/0.001/128/128-save"
    print(fluid_snap(sim_dir).shape)


# print(f"Total files deleted: {delete_count}")