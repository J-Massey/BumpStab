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
import cv2

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')

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
            Sigmaf = data["Sigmaf"]

        ax.semilogx(np.arange(1, len(Sigmaf) + 1), np.cumsum(Sigmaf/Sigmaf.sum()), label=labels[idx], linewidth=0.6, color=colours[idx])

    ax.axhline(np.cumsum(Sigmaf/Sigmaf.sum())[99], color="k", linewidth=0.6, linestyle="-.", alpha=0.5, label=f"${np.cumsum(Sigmaf/Sigmaf.sum())[99]*100:.1f}\%$")
    ax.axvline(100, color="k", linewidth=0.6, linestyle="-.", alpha=0.5)
    ax.legend()
    plt.savefig("figures/sigma.pdf", dpi=500)
    plt.close()

# plot_sigma()

def plot_Lambda():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\Re \lambda$")
    ax.set_xlim(0.1, 200)
    markerstyles = ["o", "s", "^"]
    for idx, case in enumerate(cases):
        Lambda = np.load(f"{os.getcwd()}/data/{case}/data/body_Lambda.npy")
        frequencies = (np.imag(Lambda) )/ ( np.pi )
        ax.semilogx(abs(frequencies), Lambda.real, label=labels[idx], linewidth=0.6, color=colours[idx], marker=markerstyles[idx], markersize=3, ls='none', markerfacecolor="none", markeredgewidth=0.6) 

    ax.legend()
    plt.savefig("figures/Lambda.png", dpi=500)
    plt.close()

plot_Lambda()
# frequencies = np.imag(Lambda) / (2 * np.pi * dt)



cases=["test/up", "0.001/64", "0.001/128"]
case = "test/up"

for case in cases:

    with np.load(f"{os.getcwd()}/data/{case}/data/body_svd.npz") as data:
        Sigmaf = data["Sigmaf"]


    dir = f"figures/phase-info/{case}-DMD"
    os.system(f"mkdir -p {dir}")
    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    vr = np.load(f"{os.getcwd()}/data/{case}/data/body_V_r.npy")

    r=14
    vr.resize(3, nx, ny, r)

    for n in range(r):
        fig, ax = plt.subplots(figsize=(5, 3))
        qi = np.angle(vr[2, :, :, n]).T
        print(np.max(qi), np.min(qi))
        lim = np.pi
        levels = np.linspace(-lim, lim, 44)
        _cmap = sns.color_palette("seismic", as_cmap=True)

        cs = ax.contourf(
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
        ax.set_aspect(1)
        plt.savefig(f"{dir}/{n}.png", dpi=300)
        plt.close()


# fig, ax = plt.subplots(figsize=(5, 3))
# qi = vr[2, :, :, 0].real.T

# f_transform = np.fft.fft2(qi)
# f_transform_shifted = np.fft.fftshift(f_transform)

# # Create a mask for a low-pass filter with a radius of 30
# rows, cols = qi.shape
# cx, cy = rows // 2, cols // 2
# mask = np.zeros((rows, cols), np.uint8)
# cv2.circle(mask, (cx, cy), 100, 1, thickness=-1)

# # Apply the mask to the shifted Fourier Transform
# f_transform_shifted *= mask


# magnitude_spectrum = np.abs(f_transform_shifted)
# # magnitude_spectrum = np.log(magnitude_spectrum + 1)

# # Identify peaks
# threshold = np.max(magnitude_spectrum) * 0.1  # adjust as needed
# peaks = np.argwhere(magnitude_spectrum >= threshold)
# print(peaks)


# lim = np.std(qi)*2
# levels = np.linspace(-lim, lim, 44)
# _cmap = sns.color_palette("seismic", as_cmap=True)

# cs = ax.contourf(
# pxs,
# pys,
# qi,
# levels=levels,
# vmin=-lim,
# vmax=lim,
# # norm=norm,
# cmap=_cmap,
# extend="both",
# # alpha=0.7,
# )
# ax.set_aspect(1)
# plt.savefig(f"{dir}/0.pdf", dpi=300)
# plt.close()