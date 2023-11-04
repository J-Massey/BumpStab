import os
import time
from tkinter import Tcl
from tqdm import tqdm
import sys

from lotusvis.flow_field import ReadIn


def fluid_snap(sim_dir, fn, count):
    fsim = ReadIn(sim_dir, fln, L, ext="vti")
    fsim.u_low_memory_saver(fn, count, f"{fnroot}/uvp")
    fsim.v_low_memory_saver(fn, count, f"{fnroot}/uvp")
    fsim.p_low_memory_saver(fn, count, f"{fnroot}/uvp")

def body_snap(sim_dir, fn, count):
    bsim = ReadIn(sim_dir, bln, L, ext="vti")
    bsim.save_sdf_low_memory(fn, count, f"{fnroot}/uvp")


def main(directory_to_watch):
    bcount=0
    fcount = 0
    delete_count = 0

    datpdir = os.path.join(directory_to_watch, "datp")
    dpdfs = [fp for fp in os.listdir(datpdir)]
    dpdfs = Tcl().call("lsort", "-dict", dpdfs)
    for fn in tqdm(dpdfs, total=len(dpdfs)):
        if fn.startswith(bln):
            path = os.path.join(datpdir, fn)
            body_snap(directory_to_watch, path, bcount)
            os.remove(path)
            bcount += 1
        elif fn.startswith(fln):
            path = os.path.join(datpdir, fn)
            fluid_snap(directory_to_watch, path, fcount)
            os.remove(path)
            fcount += 1
    for root, _, files in os.walk(directory_to_watch):
        for file in files:
            if (file.startswith(fln) or file.startswith(bln)) and \
            (not file.endswith(".pvtr") and not file.endswith(".vtr") and not file.endswith("vtr.pvd")):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                delete_count += 1


    print(f"Total files deleted: {delete_count}")


if __name__ == "__main__":
    case = "0.001/0"
    fln = "fluAv"; bln = "bodAv"
    L = 4096 
    fnroot = f"{os.getcwd()}/{case}"
    directory_to_watch = f"{fnroot}/lotus-data"
    print("Reading from:", directory_to_watch)
    main(directory_to_watch)

