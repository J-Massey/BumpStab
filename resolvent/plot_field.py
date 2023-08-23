import numpy as np

import os
from tkinter import Tcl

# import imageio
# from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


def plot_field(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    if lim is None:
        limb = round(np.max(qi), 2)
        lim = [-limb, limb]
    else:
        pass
    fig, ax = plt.subplots(figsize=(5, 3))
    levels = np.linspace(lim[0], lim[1], 45)
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        qi,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
        # alpha=0.7,
    )
    # cbar on top of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.2)
    fig.add_axes(cax)
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=levels[::11])
    clevels = np.mean(lim) - lim[0] * np.array(
        [-3 / 4, -1 / 2, -1 / 4, 1 / 4, 1 / 2, 3 / 4]
    )
    # Find which clevel corresponds to the closest level
    # clevel = np.argmin(np.abs(levels[:, None]-clevels[None, :]), axis=0)
    # co = ax.contour(
    #     pxs,
    #     pys,
    #     qi,
    #     levels=clevels,
    #     vmin=lim[0],
    #     vmax=lim[1],
    #     colors="black",
    #     linewidths=0.25,
    #     # alpha=0.85,
    # )
    # ax.clabel(co, inline=True, fontsize=6, fmt="%.2f")
    ax.set_aspect(1)
    plt.savefig(path, dpi=700)
    plt.close()


def plot_field_cont(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    if lim is None:
        limb = round(np.max(qi), 2)
        lim = [-limb, limb]
    else:
        pass
    fig, ax = plt.subplots(figsize=(5, 3))
    levels = np.linspace(lim[0], lim[1], 45)
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        qi,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
        # alpha=0.7,
    )
    # cbar on top of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.2)
    fig.add_axes(cax)
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=levels[::11])
    clevels = np.mean(lim) - lim[0] * np.array(
        [-3 / 4, -1 / 2, -1 / 4, 1 / 4, 1 / 2, 3 / 4]
    )
    # Find which clevel corresponds to the closest level
    # clevel = np.argmin(np.abs(levels[:, None]-clevels[None, :]), axis=0)
    # co = ax.contour(
    #     pxs,
    #     pys,
    #     qi,
    #     levels=clevels,
    #     vmin=lim[0],
    #     vmax=lim[1],
    #     colors="black",
    #     linewidths=0.25,
    #     # alpha=0.85,
    # )
    # ax.clabel(co, inline=True, fontsize=6, fmt="%.2f")
    ax.set_aspect(1)
    plt.savefig(path, dpi=700)
    plt.close()


def fns(dirn):
    fns = [fn for fn in os.listdir(dirn) if fn.endswith(f".png")]
    fns = Tcl().call("lsort", "-dict", fns)
    return fns


def gif_gen(path, nom, gif_length):
    images = [
        imageio.v3.imread(os.path.join(path, filename), plugin="pillow", mode="RGBA")
        for filename in tqdm(fns(path))
    ]

    # Calculate duration in milliseconds for the entire GIF length
    dur = (gif_length * 1000) / len(images)

    print(f"Writing GIF... with fps = {1000*(1/dur):.2f}")

    # Write the GIF
    imageio.v3.imwrite(
        nom,
        images,
        duration=dur,  # Use duration directly instead of save_kwargs
        loop=0,
        disposal=2,
    )

# Sample usage to visualise and test a case
if __name__ == "__main__":
    import os
    case = "0.001/128"
    os.system(f"mkdir -p figures/{case}-warp")
    q = np.load(f"{os.getcwd()}/data/{case}/data/uvp.npy")
    _, nx, ny, nt = q.shape
    pxs = np.linspace(-0.35, 2, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    # plot_field(q[1, :, :, 0].T, pxs, pys, f"figures/{case}_warped.pdf", lim=[-0.5, 0.5], _cmap="seismic")

    flucs = np.load(f"{os.getcwd()}/data/{case}/data/body_flucs.npy")
    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
    flucs.resize(3, nx, ny, nt)
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    for n in range(0, nt):
        plot_field(q[1, :, :, n].T, pxs, pys, f"figures/{case}-warp/{n}.png", lim=[-0.5, 0.5], _cmap="seismic")
    gif_gen(f"{case}-unwarp", f"figures/{case}_warp.gif", 12)
