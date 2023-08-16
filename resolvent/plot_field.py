import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


def plot_field(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    if lim is None:
        lim = [np.min(qi), np.max(qi)]
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

    clevels = np.mean(lim)-lim[0]*np.array([-3/4, -1/2, -1/4, 1/4, 1/2, 3/4])
    
    # Find which clevel corresponds to the closest level
    # clevel = np.argmin(np.abs(levels[:, None]-clevels[None, :]), axis=0)

    co = ax.contour(
        pxs,
        pys,
        qi,
        levels=clevels,
        vmin=lim[0],
        vmax=lim[1],
        colors='black',
        linewidths=0.25,
        # alpha=0.85,
    )
    ax.clabel(co, inline=True, fontsize=6, fmt='%.2f')

    ax.set_aspect(1)
    plt.savefig(path, dpi=700)
    plt.close()