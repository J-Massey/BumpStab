import os
import numpy as np
from resolvent.DMD_save import SaveDMD

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')

case = "test/up"
cases=["test/up", "0.001/64", "0.001/128"]
cases=["test/up"]
r=14
for case in cases:
    Phi, Lambda, b = SaveDMD(f"data/{case}/data", "body").brunton_dmd(r)

    nx, ny, nt = np.load(f"data/{case}/data/body_nxyt.npy")
    dir = f"figures/phase-info/{case}-DMD"
    os.system(f"mkdir -p {dir}")


    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)

    lim = [-0.25, 0.25]
    Phi.resize(3, nx, ny, r)

    for n in range(r):
        fig, ax = plt.subplots(figsize=(5, 3))
        qi = np.angle(Phi[2, :, :, n]).T
        print(np.max(qi), np.min(qi))
        lim = np.pi
        levels = np.linspace(-lim, lim, 44)
        _cmap = sns.color_palette("seismic", as_cmap=True)

        cs = ax.contourf(
            pxs,
            pys,
            qi,
            levels=levels,
            vmin=-lim,
            vmax=lim,
            # norm=norm,
            cmap=_cmap,
            extend="both",
            # alpha=0.7,
        )
        ax.set_aspect(1)
        plt.savefig(f"{dir}/{n}.png", dpi=300)
        plt.close()
