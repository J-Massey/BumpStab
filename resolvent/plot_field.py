import math
import sys
import numpy as np

import os
from tkinter import Tcl

import imageio
from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from scipy.ndimage import binary_dilation


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{txfonts}')


def round_to_sig_figs(x, sig_figs=2):
    if x == 0:
        return 0
    return round(x, sig_figs - int(math.floor(math.log10(abs(x)))) - 1)


def plot_field(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    if lim is None:
        limb = round_to_sig_figs((qi.max()+abs(qi.min()))/2, sig_figs=2)
        lim = [-limb, limb]
    else:
        pass
    fig, ax = plt.subplots(figsize=(5, 3))
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        np.ma.masked_array(qi, mask=mask_data(*qi.shape)),
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
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    ax.set_aspect(1)
    plt.savefig(path, dpi=700)
    plt.close()


def mask_data(nx, ny):
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
    
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    X, Y = np.meshgrid(pxs, pys)
    Z_warp_top = np.array([naca_warp(x) for x in pxs])
    Z_warp_bottom = np.array([-naca_warp(x) for x in pxs])
    mask = (Y <= Z_warp_top) & (Y >= Z_warp_bottom)
    mask_extended = binary_dilation(mask, iterations=4)
    return mask_extended

def plot_field_cont(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    if lim is None:
        limb = np.format_float_positional((qi.max()+abs(qi.min()))/2, precision=2, unique=False, fractional=False, trim='k')
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
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    clevels = -lim[0] * np.array(
        [-3 / 4, -1 / 2, -1 / 4, 1 / 4, 1 / 2, 3 / 4]
    )
    # Find which clevel corresponds to the closest level
    # clevel = np.argmin(np.abs(levels[:, None]-clevels[None, :]), axis=0)
    co = ax.contour(
        pxs,
        pys,
        qi,
        levels=clevels,
        vmin=lim[0],
        vmax=lim[1],
        colors="black",
        linewidths=0.25,
        # alpha=0.85,
    )
    ax.clabel(co, inline=True, fontsize=6, fmt="%.2f")
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

def vis_pressure(case):
    dir = f"figures/{case}-pressure"
    os.system(f"mkdir -p {dir}")
    flucs = np.load(f"{os.getcwd()}/data/{case}/data/body_flucs.npy")
    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
    flucs.resize(3, nx, ny, nt)
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)

    print(flucs[2, :, :, 0].max(), flucs[2, :, :, 0].min())
    for n in range(0, nt, 5):
        plot_field_cont(flucs[2, :, :, n].T, pxs, pys, f"{dir}/{n}.png", lim=[-0.1, 0.1], _cmap="seismic")
    gif_gen(f"figures/{case}-pressure", f"figures/{case}_pressure.gif", 8)


# # Sample usage to visualise and test a case
# if __name__ == "__main__":
#     import os
#     case = sys.argv[1]
#     vis_pressure(case)
case="0.001/16"
case="test/up"
flow = np.load(f"{os.getcwd()}/data/{case}/data/uvp.npy")
_, nx, ny, nt = flow.shape
pxs = np.linspace(-0.35, 2, nx)
pys = np.linspace(-0.35, 0.35, ny)

vorticity_fluc = -np.gradient(flow[0, :, :, :], axis=1)+np.gradient(flow[1, :, :, :], axis=0)


# ReCoVor plot
lim=[-0.1, .1]
_cmap="seismic"
qi = np.ones_like(vorticity_fluc[:, :, ::200].mean(axis=2).T)
path = f"figures/smooth.pdf"

if lim is None:
    limb = round_to_sig_figs((qi.max()+abs(qi.min()))/2, sig_figs=2)
    lim = [-limb, limb]
else:
    pass

fig, ax = plt.subplots(figsize=(5, 3))
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette(_cmap, as_cmap=True)
cs = ax.contourf(
    pxs,
    pys,
    np.ma.masked_array(qi, mask=mask_data(*qi.shape).T),
    levels=levels,
    vmin=lim[0],
    vmax=lim[1],
    # norm=norm,
    cmap=_cmap,
    extend="both",
    # alpha=0.7,
)

# clevels = -lim[0] * np.array([-3 / 4, -1 / 2, -1 / 4, 1 / 4, 1 / 2, 3 / 4])
# co = ax.contour(
#     pxs,
#     pys,
#     qi,
#     levels=clevels,
#     vmin=lim[0],
#     vmax=lim[1],
#     colors="black",
#     linewidths=0.25,
# )
# ax.clabel(co, inline=True, fontsize=6, fmt="%.2f")

# cbar on top of the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="7%", pad=0.2)
fig.add_axes(cax)
cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
cb.set_label(r"$\omega_z$", labelpad=-30, rotation=0)
ax.set_aspect(1)
plt.savefig(path, dpi=700)
plt.close()
