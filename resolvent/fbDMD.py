import os
import numpy as np
from pydmd import FbDMD

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
# cases=["test/up"]
r=14
# dmds = []
fig, axs = plt.subplots(figsize=(5, 3))

for idx, case in enumerate(cases):
    # snapshots = np.load(f"data/{case}/data/body_flucs_p.npy")
    nx, ny, nt = np.load(f"data/{case}/data/body_nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)

    # fbdmd = FbDMD(svd_rank=r)
    # fbdmd.fit(snapshots)
    # dmds.append(fbdmd)
    fbdmd = dmds[idx]
    Phi = fbdmd.modes

    # Phi.resize(nx, ny, Phi.shape[1])
    # print(Phi.shape)

    # # Phi, Lambda, b = SaveDMD(f"data/{case}/data", "body").brunton_dmd(r)

    dir = f"figures/phase-info/{case}-DMD"
    os.system(f"mkdir -p {dir}")


    for axid, n in enumerate([8,9]):
        qi = np.angle(fbdmd.modes.T[n].reshape(nx, ny)).T
        lim = np.pi
        levels = np.linspace(-lim, lim, 44)
        _cmap = sns.color_palette("husl", as_cmap=True)

        axs[axid].annotate(
            f"$f^*={(np.imag(np.log(fbdmd.eigs[n])/0.005) )/ ( np.pi ):.2f}$"
                    , xy=(0.175, 0.475), xycoords='axes fraction',
        )

        cs = axs[axid].contourf(
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
        axs[axid].set_aspect(1)
    plt.savefig(f"figures/phase-info/p_angle_{n}.png", dpi=300)
    plt.close()


# Now plot the eigenvalues
colours = sns.color_palette("colorblind", 7)
order = [2, 4, 1]
labs = [r"$\lambda = 1/0$", r"$\lambda = 1/64$", r"$\lambda = 1/128$"]


def plot_eigs():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")

    # plot unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", linewidth=0.5)

    for cidx, dmd in enumerate(dmds):
        ax.scatter(dmd.eigs.real, dmd.eigs.imag, s=10, color=colours[order[cidx]], label=labs[cidx])

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_aspect(1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    plt.savefig(f"figures/phase-info/eigs.png", dpi=600)
    plt.close()

plot_eigs()


def plot_Lambda():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\Re (\lambda)$")
    ax.set_xlim(0.1, 200)
    markerstyles = ["o", "s", "^"]
    for idx, dmd in enumerate(dmds):
        Lambda = np.log(dmd.eigs)/0.005
        frequencies = (np.imag(Lambda) )/ ( np.pi )
        ax.semilogx(abs(frequencies), Lambda.real, label=labs[idx], linewidth=0.6, color=colours[order[idx]], marker=markerstyles[idx], markersize=3, ls='none', markerfacecolor="none", markeredgewidth=0.6) 

    # put legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("figures/phase-info/Lambda.png", dpi=500)
    plt.close()

plot_Lambda()