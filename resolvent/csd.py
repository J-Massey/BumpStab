import os
import numpy as np
from pydmd import FbDMD
from scipy.fftpack import fft2, fftshift, fftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')


def phase_average(ts, p_ny):
    # split into 4 equal parts
    p_ny1 = p_ny[:int(len(ts)/4)]
    p_ny2 = p_ny[int(len(ts)/4):int(len(ts)/2)]
    p_ny3 = p_ny[int(len(ts)/2):int(3*len(ts)/4)]
    p_ny4 = p_ny[int(3*len(ts)/4):]
    return (p_ny1 + p_ny2 + p_ny3 + p_ny4)/4


def init(cases):
    p_ns = []
    for idx, case in enumerate(cases):
        p_n = np.load(f"data/{case}/data/p_n.npy")
        ts = np.arange(0, 0.005*p_n.shape[0], 0.005)
        p_ns.append(phase_average(ts, p_n))
    p_ns = np.array(p_ns)
    pxs = np.linspace(0, 1, p_ns.shape[-1])
    return ts, pxs, p_ns


def plot_difference(cases):
    ts, pxs, p_ns = init(cases)
    p_ns_filtered = gaussian_filter1d(p_ns, 3, radius=12, axis=1)
    p_ns_filtered = gaussian_filter1d(p_ns_filtered, 3, radius=12, axis=2)
    p_ns_filtered = p_ns

    phi = ts[:ts.size//4]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    fig.text(0.5, 0.05, r"$x$", ha='center', va='center')
    fig.text(0.05, 0.5, r"$\varphi$", ha='center', va='center', rotation='vertical')

    ax[1, 0].set_title(r"$\lambda = 1/64$", fontsize=10)
    ax[1, 1].set_title(r"$\lambda = 1/128$", fontsize=10)
    ax[0, 0].set_title(r"$\lambda = 1/16$", fontsize=10)
    ax[0, 1].set_title(r"$\lambda = 1/32$", fontsize=10)

    [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]
    [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]
    
    lims = [-0.01, 0.01]

    dp_64 = p_ns_filtered[0]-p_ns_filtered[1]
    dp_128 = p_ns_filtered[0]-p_ns_filtered[2]
    dp_16 = p_ns_filtered[0]-p_ns_filtered[3]
    dp_32 = p_ns_filtered[0]-p_ns_filtered[4]

    print([(p_ns_filter).sum() for p_ns_filter in p_ns])
    # print(dp_16.sum(), dp_32.sum(), dp_64.sum(), dp_128.sum())

    ax[0,0].imshow(
        dp_16,
        extent=[pxs[0], pxs[-1], phi[0], phi[-1]],
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower'
    )

    ax[0,1].imshow(
        dp_32,
        extent=[pxs[0], pxs[-1], phi[0], phi[-1]],
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower'
    )

    ax[1,0].imshow(
        dp_64,
        extent=[pxs[0], pxs[-1], phi[0], phi[-1]],
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower'
    )
    
    im128 = ax[1,1].imshow(
        dp_128,
        extent=[pxs[0], pxs[-1], phi[0], phi[-1]],
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower'
    )

    # plot colorbar
    cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    cb = plt.colorbar(im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal")
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\langle p_{s,smooth}-p_{s,rough} \rangle$", labelpad=-25, rotation=0)    

    plt.savefig(f"figures/phase-info/surface/difference.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/difference.png", dpi=450, transparent=True)
    plt.close()


# Function to compute 2D (x-t) cross-spectral density
def compute_crosscorr(array1, array2):
    # Normalise
    array1 /= np.linalg.norm(array1)
    array2 /= np.linalg.norm(array2)

    FFT1 = np.fft.fft2(array1)
    FFT2 = np.fft.fft2(array2)
    cross_corr_freq = FFT1 * np.conj(FFT2)

    # Inverse Fourier Transform to get cross-correlation in spatial domain
    cross_corr = np.fft.ifft2(cross_corr_freq).real
    return cross_corr


def plot_difference_spectra(cases):
    ts, pxs, p_ns = init(cases)
    phi = ts[:ts.size//4]
    p_ns_filtered = gaussian_filter1d(p_ns, 3, radius=12, axis=1)
    p_ns_filtered = gaussian_filter1d(p_ns_filtered, 3, radius=12, axis=2)
    # p_ns_filtered = p_ns

    dx = 4/4096  # Spatial step
    dt = 0.005  # Temporal step
    
    cc64 = np.fft.fft2(p_ns_filtered[0]-p_ns_filtered[1])
    cc64 = np.fft.fftshift(cc64)
    cc128 = np.fft.fft2(p_ns_filtered[0]-p_ns_filtered[2])
    cc128 = np.fft.fftshift(cc128)
    cc16 = np.fft.fft2(p_ns_filtered[0]-p_ns_filtered[3])
    cc16 = np.fft.fftshift(cc16)
    cc32 = np.fft.fft2(p_ns_filtered[0]-p_ns_filtered[4])
    cc32 = np.fft.fftshift(cc32)

    num_rows, num_cols = p_ns[0].shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(num_cols, 4/4096))
    freq_y = np.fft.fftshift(np.fft.fftfreq(num_rows, 0.005))

    extent=[freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    fig.text(0.5, 0.01, r"$1/k_x$", ha='center', va='center')
    fig.text(0.03, 0.5, r"$f^*$", ha='center', va='center', rotation='vertical')
    # title for each plot
    ax[1, 0].set_title(r"$\lambda = 1/64$")
    ax[1, 1].set_title(r"$\lambda = 1/128$")
    ax[0, 0].set_title(r"$\lambda = 1/16$")
    ax[0, 1].set_title(r"$\lambda = 1/32$")

    # [ax.set_xlim(0, 1) for ax in ax]
    # [ax.set_ylim(0, 1) for ax in ax]
    
    lims = [0, 4]
    cmap = sns.color_palette("icefire", as_cmap=True)

    im64 = ax[1, 0].imshow(
        np.log(np.abs(cc64)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=cmap,
        aspect='auto',
        origin='lower',
    )
    
    im128 = ax[1, 1].imshow(
        np.log(np.abs(cc128)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=cmap,
        aspect='auto',
        origin='lower',
    )

    im16 = ax[0, 0].imshow(
        np.log(np.abs(cc16)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=cmap,
        aspect='auto',
        origin='lower'
    )

    im32 = ax[0, 1].imshow(
        np.log(np.abs(cc32)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=cmap,
        aspect='auto',
        origin='lower',
    )

    # set all axes with symlog
    [[ax.set_xscale('symlog') for ax in ax[n, :]] for n in range(2)]
    [[ax.set_yscale('symlog') for ax in ax[n, :]] for n in range(2)]

    # annotate with a box
    [[ax.axhspan(2.5, 10, facecolor='grey', alpha=0.2, edgecolor='none') for ax in ax[n, :]] for n in range(2)]
    [[ax.axvspan(-24, -8, facecolor='grey', alpha=0.2, edgecolor='none') for ax in ax[n, :]] for n in range(2)]
    [[ax[n, m].plot([-24, -8], [2.5, 2.5], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    [[ax[n, m].plot([-24, -8], [10, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    [[ax[n, m].plot([-24, -24], [2.5, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    [[ax[n, m].plot([-8, -8], [2.5, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]


    # plot colorbar
    cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    cb = plt.colorbar(im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal")
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$PSD(\langle p_{s,smooth}-p_{s,rough} \rangle)$", labelpad=-25, rotation=0)
    
    plt.savefig(f"figures/phase-info/surface/csd.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/csd.png", dpi=450, transparent=True)

if __name__ == "__main__":
    colours = sns.color_palette("colorblind", 7)
    order = [2, 4, 1, 0, 3]
    lams = [1e9, 1/64, 1/128, 1/16, 1/32]
    labs = [f"$\lambda = 1/{int(1/lam)}$" for lam in lams]
    cases = ["test/span64", "0.001/64", "0.001/128", "0.001/16", "0.001/32"]

    # plot_csd(cases)
    plot_difference(cases)
    plot_difference_spectra(cases)
    

