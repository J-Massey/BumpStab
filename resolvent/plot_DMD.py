import numpy as np

from plot_field import plot_field
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch
import os

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{txfonts}')

colours = sns.color_palette("colorblind", 7)

cases = ["0.001/16", "0.001/128", "test/up"]
labels = [r"$\lambda = 1/16$", r"$\lambda = 1/128$", "Smooth"]

def plot_sigma():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Cumulative modal energy")
    ax.set_ylim(0, 1)

    for idx, case in enumerate(cases):
        with np.load(f"{os.getcwd()}/data/{case}/data/body_svd.npz") as data:
            # Ub = data["Ub"]
            # Sigmab = data["Sigmab"]
            # VTb = data["VTb"]
            # Uf = data["Uf"]
            Sigmaf = data["Sigmaf"]
            # VTf = data["VTf"]
        ax.semilogx(np.arange(1, len(Sigmaf) + 1), np.cumsum(Sigmaf/Sigmaf.sum()), label=labels[idx], linewidth=0.6, color=colours[idx])
    ax.axhline(np.cumsum(Sigmaf/Sigmaf.sum())[99], color="k", linewidth=0.6, linestyle="-.", alpha=0.5, label=f"${np.cumsum(Sigmaf/Sigmaf.sum())[99]*100:.1f}\%$")
    ax.axvline(100, color="k", linewidth=0.6, linestyle="-.", alpha=0.5)
    ax.legend()
    plt.savefig("figures/sigma.pdf", dpi=500)
    plt.close()


case="0.001/16"
for case in cases:
    dir = f"figures/{case}-DMD"
    os.system(f"mkdir -p {dir}")
    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    vr = np.load(f"{os.getcwd()}/data/{case}/data/body_V_r.npy")

    vr.resize(3, nx, ny, 100)

    fig, ax = plt.subplots(figsize=(5, 3))
    lim = [-0.001, 0.001]
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)

    cs = ax.contourf(
        pxs,
        pys,
        vr[2, :, :, 0].real.T,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
        # alpha=0.7,
    )
    ax.set_aspect(1)
    plt.savefig(f"{dir}/{0}.pdf", dpi=500)
    plt.close()

    for n in range(20):
        plot_field((vr[2, :, :, n].T).real, pxs, pys, f"{dir}/{n}.png", lim=[-0.01,0.01], _cmap="seismic")

