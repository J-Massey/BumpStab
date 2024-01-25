import os
import numpy as np

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
    warp = (a * xp + b * xp**2 + c * xp**3 + d * xp**4 + e * xp**5 + 
            f * xp**6 + g * xp**7 + h * xp**8 + i * xp**9 + j * xp**10)
    
    return warp


def fwarp(t: float, pxs):
    if isinstance(pxs, float):
        x = pxs
        xp = min(max(x, 0.0), 1.0)
        return -0.5*(0.28 * xp**2 - 0.13 * xp + 0.05) * np.sin(2*np.pi*(t - (1.42* xp)))
    else:
        return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def plot_mapped_unmapped(qim, qi):
    _, nx, ny, nt = qim.shape
    n=0
    pxs = np.linspace(0, 1, nx)
    pysm = np.linspace(-0.35, 0.35, ny)
    y_maskm = np.logical_and(pysm <= 0.15, pysm >= -0.15)
    _, nx, ny, nt = qi.shape
    pysu = np.linspace(-0.25, 0.25, ny)
    y_masku = np.logical_and(pysu <= 0.15, pysu >= -0.15)
    for n in range(0, 205, 5):
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        ax[0].text(-0.12, 0.98, r"(a)", transform=ax[0].transAxes)
        ax[1].text(-0.12, 0.98, r"(b)", transform=ax[1].transAxes)

        # ax[0].set_xlabel(r"$x$")
        ax[1].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$y$")
        ax[1].set_ylabel(r"$n$")
        q_mapped = np.sqrt(qim[0, :, :, n]**2 + qim[1, :, :, n]**2)
        q_unmapped = np.sqrt(qi[0, :, :, n]**2 + qi[1, :, :, n]**2)
        lims = [0, 1.4]
        norm = TwoSlopeNorm(vmin=lims[0], vcenter=(lims[1]-lims[0])/2, vmax=lims[1])
        _cmap = sns.color_palette("icefire_r", as_cmap=True)

        co1 = ax[0].imshow(
            q_mapped[:, y_maskm].T,
            extent=[0, 1, -0.15, 0.15],
            cmap=_cmap,
            norm=norm,
            origin="lower",
        )
        ax[0].set_aspect(1)
        ax[0].fill_between(pxs, [naca_warp(x)-fwarp(n/200, x) for x in pxs], [-naca_warp(x)-fwarp(n/200, x) for x in pxs], color="white")

        co2 = ax[1].imshow(
            q_unmapped.T,
            extent=[0, 1, 0, 0.25],
            cmap=_cmap,
            norm=norm,
            origin="lower",
        )

        # Add space for arrow
        fig.subplots_adjust(top=0.85)

        # Add colorbar
        cax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
        cb = plt.colorbar(co1, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
        cb.ax.xaxis.tick_top()  # Move ticks to top
        cb.ax.xaxis.set_label_position('top')  # Move label to top
        cb.set_label(r"$\left|\vec{u}\right|$", labelpad=-40, rotation=0)

        # plt.savefig(f"figures/mapping/unmap-gif/unmapping.pdf")
        plt.savefig(f"figures/mapping/unmap-gif/{n}.png", dpi=600)
        plt.close()


def plot_mapped_unmapped_stat(qim, qi):
    _, nx, ny, nt = qim.shape
    pxs = np.linspace(0, 1, nx)
    pysm = np.linspace(-0.35, 0.35, ny)
    y_maskm = np.logical_and(pysm <= 0.15, pysm >= -0.15)
    _, nx, ny, nt = qi.shape
    pysu = np.linspace(-0.25, 0.25, ny)
    y_masku = np.logical_and(pysu <= 0.15, pysu >= -0.15)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax[0].text(-0.12, 0.98, r"(a)", transform=ax[0].transAxes)
    ax[1].text(-0.12, 0.98, r"(b)", transform=ax[1].transAxes)

    # ax[0].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    ax[1].set_ylabel(r"$n$")
    q_mapped = np.sqrt(qim[0, :, :, 0]**2 + qim[1, :, :, 0]**2)
    q_unmapped = np.sqrt(qi[0, :, :, 0]**2 + qi[1, :, :, 0]**2)
    lims = [0, 1.4]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=(lims[1]-lims[0])/2, vmax=lims[1])
    _cmap = sns.color_palette("icefire_r", as_cmap=True)

    co1 = ax[0].imshow(
        q_mapped.T,
        extent=[-0.35, 2, -0.35, 0.35],
        cmap=_cmap,
        # norm=norm,
        origin="lower",
    )
    ax[0].set_aspect(1)
    # ax[0].fill_between(pxs, [naca_warp(x)-fwarp(0, x) for x in pxs], [-naca_warp(x)-fwarp(0, x) for x in pxs], color="white")

    # co2 = ax[1].imshow(
    #     q_unmapped.T,
    #     extent=[0, 1, 0, 0.25],
    #     cmap=_cmap,
    #     norm=norm,
    #     origin="lower",
    # )

    # # Add space for arrow
    # fig.subplots_adjust(top=0.85)

    # # Add colorbar
    # cax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
    # cb = plt.colorbar(co1, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
    # cb.ax.xaxis.tick_top()  # Move ticks to top
    # cb.ax.xaxis.set_label_position('top')  # Move label to top
    # cb.set_label(r"$\left|\vec{u}\right|$", labelpad=-40, rotation=0)

    plt.savefig(f"figures/mapping/unmapping_stat.pdf")
    # plt.savefig(f"figures/mapping/unmapping_stat.png", dpi=600)
    plt.close()


def save_qim():
    u = np.load(f"data/stationary/u.npy", mmap_mode="r")
    v = np.load(f"data/stationary/v.npy", mmap_mode="r")
    qim = np.stack((u, v), axis=0)
    qim  = np.einsum("ijkl->ilkj", qim)
    np.save(f"data/stationary/uv.npy", qim)

def save_body():
    uv = np.load(f"data/stationary/uv.npy", mmap_mode="r")
    _, nx, ny, nt = uv.shape
    pxs = np.linspace(-.35, 2, nx)
    b_mask = np.logical_and(pxs <= 1, pxs >= 0)
    body = uv[:, b_mask, :, :]
    np.save(f"data/stationary/body.npy", body)


if __name__ == "__main__":
    # qim = np.load(f"data/0.001/0/unmasked/body.npy")
    # s = np.load(f"data/0.001/0/unmasked/s_profile.npy")
    # n = np.load(f"data/0.001/0/unmasked/n_profile.npy")
    # qi = np.stack((s, n), axis=0)
    # qi = np.einsum("ijkl->iklj", qi)
    plot_mapped_unmapped(qim, qi)

    # save_qim()
    # save_body()
    
    # qim = np.load(f"data/stationary/uv.npy", mmap_mode="r")
    # s = np.load(f"data/stationary/s_profile.npy")
    # n = np.load(f"data/stationary/n_profile.npy")
    # qi = np.stack((s, n), axis=0)
    # qi = np.einsum("ijkl->iklj", qi)
    # plot_mapped_unmapped_stat(qim, qi)