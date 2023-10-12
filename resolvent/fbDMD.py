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
colours = sns.color_palette("colorblind", 7)
order = [2, 4, 1]
labs = [r"$\lambda = 1/0$", r"$\lambda = 1/64$", r"$\lambda = 1/128$"]

case = "test/up"
cases=["test/up", "0.001/64", "0.001/128"]
# cases=["test/up"]
r=14
# dmds = []
fig, ax = plt.subplots(figsize=(5, 3))
ax.annotate(
    f"$f^*={2.00:.2f}$"
            , xy=(0.175, 0.475), xycoords='axes fraction',
)

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

    if idx:
        n=9
    else:
        n=8

    # for axid, n in enumerate([8,9]):
    qi = np.angle(fbdmd.modes.T[n].reshape(nx, ny)).T/np.pi
    lims = [0.2, 0.4]
    levels = np.linspace(lims[0], lims[1], 6)
    # levels = np.append(np.linspace(-lims[1], -lims[0], 6), levels)
    _cmap = sns.color_palette("husl", as_cmap=True)


    co = ax.contour(
        pxs,
        pys,
        qi,
        levels=levels,
        vmin=lims[0],
        vmax=lims[1],
        colors=[colours[order[idx]]],
        linewidths=0.25,
        label=labs[idx],
        # alpha=0.85,
    )
ax.clabel(co, inline=True, fontsize=6, fmt="%.2f", colors='grey')
ax.set_aspect(1)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend()
plt.savefig(f"figures/phase-info/phase.png", dpi=800)
plt.close()



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