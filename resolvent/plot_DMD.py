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
dir = f"figures/{case}-DMD"
os.system(f"mkdir -p {dir}")
nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
pxs = np.linspace(0, 1, nx)
pys = np.linspace(-0.25, 0.25, ny)
vr = np.load(f"{os.getcwd()}/data/{case}/data/body_V_r.npy")

vr.resize(3, nx, ny, 100)

from scipy.ndimage import binary_dilation

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

X, Y = np.meshgrid(pxs, pys)
Z_warp_top = np.array([naca_warp(x) for x in pxs])
Z_warp_bottom = np.array([-naca_warp(x) for x in pxs])
mask = (Y <= Z_warp_top) & (Y >= Z_warp_bottom)
mask_extended = binary_dilation(mask, iterations=4)
masked = np.ma.masked_array((vr[2, :, :, 0].T).real, mask=mask_extended)

fig, ax = plt.subplots(figsize=(5, 3))
lim = [-0.001, 0.001]
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

cs = ax.contourf(
    pxs,
    pys,
    masked,
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

