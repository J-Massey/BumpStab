import os
import numpy as np
from pydmd import FbDMD
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
colours = sns.color_palette("colorblind", 7)

def naca_warp(x):
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274

    xp = min(max(x, 0.0), 1.0)
    
    return (a * xp + b * xp**2 + c * xp**3 + d * xp**4 + e * xp**5 + 
            f * xp**6 + g * xp**7 + h * xp**8 + i * xp**9 + j * xp**10)

def plot_eigs():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")

    # plot unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", linewidth=0.5)

    eigs = np.exp(np.load(f"data/0.001/0/unmasked/body_Lambda.npy")*0.005)
    ax.scatter(eigs.real, eigs.imag, s=2, color=colours[2])

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_aspect(1)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    plt.savefig(f"figures/eigs.pdf")
    plt.savefig(f"figures/eigs.png", dpi=600)
    plt.close()


def load(case="0"):
    snapshots = np.load(f"data/0.001/{case}/unmasked/body_unwarped.npy")
    return snapshots


def preprocess(snapshots):
    snapshots = snapshots[:-1]
    snapshots = snapshots - np.mean(snapshots, axis=3, keepdims=True)
    # snapshots = snapshots / np.std(snapshots, axis=3, keepdims=True)
    return snapshots


def mask_snaps(snapshots):
    _, nx, ny, nt = snapshots.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    for idx in tqdm(range(nx), desc="Masking"):
        bmask = np.logical_and(pys <= naca_warp(pxs[idx]) - 4/4096 , pys >= -naca_warp(pxs[idx])+4/4096)
        snapshots[:, idx, bmask, :] = 0
    return snapshots

if __name__ == "__main__":
    snapshots = load()
    _, nx, ny, nt = snapshots.shape
    snap_flucs = mask_snaps(snapshots)
    snap_flucs = preprocess(snap_flucs)


    r=400
    fbdmd = FbDMD(svd_rank=r)
    fbdmd.fit(snap_flucs.reshape(2*nx*ny, nt))
    # dmds.append(fbdmd)
    Phi = fbdmd.modes
    np.save(f"data/0.001/0/unmasked/body_V_r.npy", Phi)

    A_tilde = fbdmd.operator._Atilde
    rho, W = np.linalg.eig(A_tilde)
    # Find the eigenfunction from spectral expansion
    Lambda = np.log(rho) / 0.005
    np.save(f"data/0.001/0/unmasked/body_Lambda.npy", Lambda)
    plot_eigs()
    
# -- Plot the modes -- #
# Phi.resize(nx, ny, Phi.shape[1])

# for idx, case in enumerate(cases):
#     fbdmd = dmds[idx]
#     dir = f"figures/DMD/{case}"

#     for axid, n in tqdm(enumerate(range(r)), desc="Plotting modes"):
#         fig, ax = plt.subplots(figsize=(6, 3))
#         qi = np.real(fbdmd.modes.T[n].reshape(nx, ny))
#         y_mask = np.logical_and(pys >= -0.1, pys <= 0.1)
#         qi = qi[:, y_mask]
#         qi = gaussian_filter1d(qi, sigma=1, axis=0)
#         # print(qi.max()/np.pi, qi.min()/np.pi)
#         lims = [-0.01, 0.01]
#         norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
#         _cmap = sns.color_palette("seismic", as_cmap=True)

#         ax.plot(pxs, [naca_warp(x) for x in pxs], color="k", linewidth=0.7)
#         ax.plot(pxs, [-naca_warp(x) for x in pxs], color="k", linewidth=0.7)

#         ax.set_title(f"$f^*={fbdmd.frequency[axid]/0.005:.2f}$")
#         co = ax.imshow(
#             qi.T,
#             extent=[0, 1, -0.1, 0.1],
#             cmap=_cmap,
#             norm=norm,
#         )

#         ax.set_aspect(1)
#         ax.set_xlabel(r"$x$")
#         ax.set_ylabel(r"$y$")
#         plt.savefig(f"{dir}/mode_{n}.png", dpi=600)
#         plt.close()





# def plot_Lambda():
#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.set_xlabel(r"$f^*$")
#     ax.set_ylabel(r"$\Re (\lambda)$")
#     ax.set_xlim(0.1, 200)
#     markerstyles = ["o", "s", "^"]
#     for idx, dmd in enumerate(dmds):
#         Lambda = np.log(dmd.eigs)/0.005
#         frequencies = (np.imag(Lambda) )/ ( np.pi )
#         ax.semilogx(abs(frequencies), Lambda.real, label=labs[idx], linewidth=0.6, color=colours[order[idx]], marker=markerstyles[idx], markersize=3, ls='none', markerfacecolor="none", markeredgewidth=0.6) 

#     # put legend outside plot
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.savefig(f"{dir}/Lambda.png", dpi=500)
#     plt.close()

# plot_Lambda()