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
    

# bcount=0
# fcount = 0
# delete_count = 0
# # while True:
# for root, _, files in os.walk(directory_to_watch):
#     # Process files
#     for file in files:
#         if root.endswith("datp"):
#             # Sort the files
#             dpdfs = [fp for fp in os.listdir(root)]
#             dpdfs = Tcl().call("lsort", "-dict", dpdfs)
#             for fn in dpdfs:
#                 if fn.startswith("bodyF"):
#                     path = os.path.join(root, fn)
#                     body_snap(directory_to_watch, path, bcount)
#                     os.remove(path)
#                     bcount += 1
#                 elif fn.startswith("fluid"):
#                     path = os.path.join(root, fn)
#                     fluid_snap(directory_to_watch, path, fcount)
#                     os.remove(path)
#                     fcount += 1
# for root, _, files in os.walk(directory_to_watch):
#     for file in files:
#         if (file.startswith("fluid") or file.startswith("bodyF")) and \
#         (not file.endswith(".pvtr") and not file.endswith(".vtr") and not file.endswith("vtr.pvd")):
#             file_path = os.path.join(root, file)
#             os.remove(file_path)
#             delete_count += 1


# print(f"Total files deleted: {delete_count}")