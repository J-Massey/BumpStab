from SVD_save import SaveSVD
from DMD_save import SaveDMD
from resolvent_analysis import ResolventAnalysis
import sys
import numpy as np
from plot_field import plot_field, gif_gen

def vis_unwarp(case):
    os.system(f"mkdir -p figures/{case}-warp")
    flucs = np.load(f"{os.getcwd()}/data/{case}/data/body_flucs.npy")
    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
    flucs.resize(3, nx, ny, nt)
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    for n in range(0, nt, 5):
        plot_field(flucs[1, :, :, n].T, pxs, pys, f"figures/{case}-warp/{n}.png", lim=[-0.5, 0.5], _cmap="seismic")
    gif_gen(f"figures/{case}-warp", f"figures/{case}_warp.gif", 8)


def main(case, dom):
    svd_save = SaveSVD(f"{os.getcwd()}/data/{case}/data", dom)
    svd_save.save_flucs()
    svd_save.save_svd()
    resolvent = SaveDMD(f"{os.getcwd()}/data/{case}/data", dom)
    resolvent.save_fbDMD(r=100)
    ra = ResolventAnalysis(f"{os.getcwd()}/data/{case}/data", dom, omega_span=np.logspace(np.log10(0.1), np.log10(150*2*np.pi), 1000))
    ra.save_gain()
    ra.save_omega_peaks()


# Sample usage
if __name__ == "__main__":
    import os
    # case = sys.argv[1]
    case = "test/up"
    # doms = ["body", "wake"]
    # for dom in doms:
    #     main(case, dom)
    vis_unwarp(case)