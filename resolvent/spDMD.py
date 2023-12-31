import os
import numpy as np
from pydmd import SpDMD
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
    ax.set_xlabel(r"$\Re(\varsigma)$")
    ax.set_ylabel(r"$\Im(\varsigma)$")

    theta = np.linspace(0, 2*np.pi, 200)
    # ax.fill(np.cos(theta), np.sin(theta), color="green", alpha=0.15)
    # ax.fill(0.8*np.cos(theta), 0.8*np.sin(theta), color="white")
    ax.plot(np.cos(theta), np.sin(theta), color="k", linewidth=0.4)


    eigs = np.exp(np.load(f"data/0.001/0/unmasked/sp_Lambda.npy")*0.005)
    ax.scatter(eigs.real, eigs.imag, s=1, color=colours[2])
    # ax.scatter(eigs[36:].real, eigs[36:].imag, s=1, color='purple')
    ax.set_aspect(1)

    plt.savefig(f"figures/eigs.pdf")
    plt.savefig(f"figures/eigs.png", dpi=600)
    plt.close()

def plot_unmapped(qi, nx, ny, nt):
    pxs = np.linspace(0, 1, nx)
    for n in tqdm(range(nt), desc="Plotting modes", total=nt):
        fig, ax = plt.subplots(figsize=(6, 3))
        q = np.sqrt(qi[0, :, :, n]**2 + qi[1, :, :, n]**2)
        lims = [0, .1]
        norm = TwoSlopeNorm(vmin=lims[0], vcenter=(lims[1]-lims[0])/2, vmax=lims[1])
        _cmap = sns.color_palette("icefire", as_cmap=True)

        # ax.plot(pxs, [naca_warp(x) for x in pxs], color="k", linewidth=0.7)
        # ax.plot(pxs, [-naca_warp(x) for x in pxs], color="k", linewidth=0.7)

        co = ax.imshow(
            q.T,
            extent=[0, 1, 0, 0.25],
            cmap=_cmap,
            norm=norm,
            origin="lower",
        )

        ax.set_aspect(1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.savefig(f"figures/mapping/unmapped/{n}.png", dpi=600)
        plt.close()


def plot_modes(dmd, Phi, nx, ny, r):
    Phi = Phi.reshape(2, nx, ny, r)
    pxs = np.linspace(0, 1, nx)
    for axid, n in tqdm(enumerate(range(r)), desc="Plotting modes"):
        fig, ax = plt.subplots(figsize=(6, 3))
        qi = np.real(Phi[:, :, :, n])
        qi = np.sqrt(qi[0]**2 + qi[1]**2)
        lims = [0, 0.001]
        norm = TwoSlopeNorm(vmin=lims[0], vcenter=(lims[1]-lims[0])/2, vmax=lims[1])
        _cmap = sns.color_palette("icefire", as_cmap=True)

        ax.set_title(f"$f^*={dmd.frequency[axid]/0.005:.2f}$")
        co = ax.imshow(
            qi.T,
            extent=[0, 1, 0, 0.25],
            cmap=_cmap,
            norm=norm,
            origin="lower",
        )

        # ax.set_aspect(1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.savefig(f"figures/DMD/sp_mode_{n}.png", dpi=600)
        plt.close()


def load(case="0"):
    u = np.load(f"data/0.001/{case}/unmasked/s_profile.npy")
    v = np.load(f"data/0.001/{case}/unmasked/n_profile.npy")
    snapshots = np.stack((u, v), axis=0)
    snapshots = np.einsum("ijkl->iklj", snapshots)
    return snapshots


def preprocess(snapshots):
    snapshots = snapshots - np.mean(snapshots, axis=3, keepdims=True)
    return snapshots


def run_spDMD():
    snapshots = load()
    _, nx, ny, nt = snapshots.shape
    np.save(f"data/0.001/0/unmasked/nxyt.npy", np.array([nx, ny, nt]))
    snap_flucs = preprocess(snapshots)
    # plot_unmapped(snap_flucs[:, :, :, :200], nx, ny, nt)
    flat_snaps = snap_flucs.reshape(2*nx*ny, nt)

    r=100
    spdmd = SpDMD(svd_rank=r, gamma=300, rho=1.0e4)
    spdmd.fit(flat_snaps)
    # dmds.append(spDMD)
    Phi = spdmd.modes
    plot_modes(spdmd, Phi, nx, ny, r)
    np.save(f"data/0.001/0/unmasked/sp_V_r.npy", Phi)

    A_tilde = spdmd.operator._Atilde
    rho, W = np.linalg.eig(A_tilde)
    # Find the eigenfunction from spectral expansion
    Lambda = np.log(rho) / 0.005
    np.save(f"data/0.001/0/unmasked/sp_Lambda.npy", Lambda)


if __name__ == "__main__":
    # run_spDMD()
    plot_eigs()