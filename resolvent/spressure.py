import os
from tqdm import tqdm
import sys
import numpy as np

from matplotlib import colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import interp1d
from scipy.signal import welch, savgol_filter
from scipy.interpolate import CubicSpline


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")


def read_forces(force_file, interest="p", direction="x"):
    names = [
        "t",
        "dt",
        "px",
        "py",
        "pz",
        "cp",
        "vx",
        "vy",
        "vz",
        "E",
        "tke",
    ]
    fos = np.transpose(np.genfromtxt(force_file))

    forces_dic = dict(zip(names, fos))
    t = forces_dic["t"]
    u = forces_dic[interest + direction]

    u = np.squeeze(np.array(u))
    return t, u


def fnorm(case):
    if case == "0.001/16" or case == "0.001/32":
        span = 128
    else:
        span = 64
    normalise = 0.1 * 4096 * span * 4 / 2
    return normalise


def load_cp(cases):
    if os.path.isfile("data/spressure.npy"):
        body = np.load("data/spressure.npy")
    else:
        t, pxs, body = [], [], []
        for case in tqdm(cases):
            cp = np.genfromtxt(f"data/{case}/spressure/fort.1")  # Load the 1D array
            start = np.where(cp[0] != 0)[0][0]
            bod = cp[:, start : start + 1024] / fnorm(case)
            # body.append()   # Clip to the body
            tr, _ = read_forces(f"data/{case}/spressure/fort.9", "cp", "")
            # Resample onto uniform time
            ts = np.linspace(0.001, 0.999, 2000)
            # Sort and remove duplicates
            sorted_indices = np.argsort(tr % 1)
            sorted_x = (tr % 1)[sorted_indices]
            sorted_y = bod[sorted_indices]
            unique_x, unique_idx = np.unique(sorted_x, return_index=True)
            unique_y = sorted_y[unique_idx]
            # Create spline
            cs = CubicSpline(unique_x, unique_y)
            y_spline = cs(ts)
            body.append(y_spline)
        np.save("data/spressure.npy", body)
    return np.linspace(0.001, 0.999, 2000), np.linspace(0, 1, 1024), body


def straight_line(a, b):
    m = (b[1] - a[1]) / (b[0] - a[0])
    c = a[1] - m * a[0]
    print(m)
    return [c, m + c]

    
def plot_cp(ts, pxs, body):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    fig.text(0.5, 0.05, r"$x$", ha="center", va="center")
    fig.text(0.05, 0.5, r"$\varphi$", ha="center", va="center", rotation="vertical")

    ax[1, 0].set_title(r"$\lambda = 1/64$", fontsize=10)
    ax[1, 1].set_title(r"$\lambda = 1/128$", fontsize=10)
    ax[0, 0].set_title(r"$\lambda = 1/0$", fontsize=10)
    ax[0, 1].set_title(r"$\lambda = 1/32$", fontsize=10)

    [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]
    [[ax.set_ylim(0, 1) for ax in ax[n, :]] for n in range(2)]

    lims = [-0.002, 0.002]
    norm = TwoSlopeNorm(vcenter=0, vmin=lims[0], vmax=lims[1])

    # dp_64 = body[0]-body[1]
    # dp_128 = body[0]-body[2]
    # dp_16 = body[0]-body[3]
    # dp_32 = body[0]-body[4]

    ax[0,0].imshow(
        body[0],
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower',
        norm=norm,
    )


    # ax[0,1].imshow(
    #     body[3],
    #     extent=[0, 1, 0, 1],
    #     # vmin=lims[0],
    #     # vmax=lims[1],
    #     cmap=sns.color_palette("seismic", as_cmap=True),
    #     aspect='auto',
    #     origin='lower',
    #     norm=norm,
    # )

    ax[1, 0].imshow(
        body[1],
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect="auto",
        origin="lower",
        norm=norm,
    )

    im128 = ax[1, 1].imshow(
        body[2],
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect="auto",
        origin="lower",
        norm=norm,
    )

    ap, bp = [0.007142857142857173, 0.7608069164265131], [0.16571428571428565, 0.985849855907781]
    [[a.plot([0, 1], straight_line(ap, bp), color="k", linewidth=0.4, ls='-.') for a in ax[n, :]] for n in range(2)]
    ap, bp = [0, 0.25], [0.528, 1]
    [[a.plot([0, 1], straight_line(ap, bp), color="k", linewidth=0.4, ls='-.') for a in ax[n, :]] for n in range(2)]
    ap, bp = [0.18, 0], [0.6857142857142857, 0.7202247838616716]
    [[a.plot([0, 1], straight_line(ap, bp), color="k", linewidth=0.4, ls='-.') for a in ax[n, :]] for n in range(2)]
    ap, bp = [0.6638571428571428, 0.19452449567723362], [0.8399999999999999, 0.4438040345821327]
    [[a.plot([0, 1], straight_line(ap, bp), color="k", linewidth=0.4, ls='-.') for a in ax[n, :]] for n in range(2)]
    # plot colorbar
    cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    cb = plt.colorbar(
        im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal"
    )
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(r"$\langle p_{s} \rangle$", labelpad=-25, rotation=0)

    plt.savefig(f"figures/phase-info/surface/spressure.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/spressure.png", dpi=450, transparent=True)
    plt.close()


def plot_cp_diff(ts, pxs, body):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    fig.text(0.5, 0.01, r"$x$", ha="center", va="center")
    fig.text(0.07, 0.5, r"$\varphi$", ha="center", va="center", rotation="vertical")

    # ax[0].set_title(r"$\lambda = 1/64$", fontsize=10)
    # ax[1].set_title(r"$\lambda = 1/128$", fontsize=10)

    # [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]
    # [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]

    lims = [-0.001, 0.001]
    norm = TwoSlopeNorm(vcenter=0, vmin=lims[0], vmax=lims[1])

    dp_64 = body[0] - body[1]
    dp_128 = body[0] - body[2]
    # dp_32 = body[0]-body[3]
    # dp_16 = body[0]-body[3]
    # dp_32 = body[0]-body[4]

    ax[0].imshow(
        dp_64,
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect="auto",
        origin="lower",
        norm=norm,
    )
    ax[0].set_aspect(1)

    # new axis next to the plot
    divider = make_axes_locatable(ax[0])
    vax1 = divider.append_axes("right", size="20%", pad=0.1)
    fig.add_axes(vax1)
    t_int64 = dp_64.sum(axis=1)
    vax1.plot(t_int64[t_int64>0], ts[t_int64>0], color='red', marker='o', linestyle='none', markersize=.1)
    vax1.plot(t_int64[t_int64<0], ts[t_int64<0], color='blue', marker='o', linestyle='none', markersize=.1)
    vax1.set_ylim(0, 1)
    vax1.set_xlim(-0.03, 0.03)

    vax1.xaxis.set_visible(False)
    vax1.yaxis.set_visible(False)
        
    hax1 = divider.append_axes("top", size="20%", pad=0.1)
    fig.add_axes(hax1)
    x_int64 = dp_64.sum(axis=0)
    hax1.plot(pxs[x_int64>0], x_int64[x_int64>0], color='red', marker='o', linestyle='none', markersize=.1)
    hax1.plot(pxs[x_int64<0], x_int64[x_int64<0], color='blue', marker='o', linestyle='none', markersize=.1)
    hax1.set_xlim(0, 1)
    hax1.set_ylim(-0.1, 0.1)
    print(pxs[x_int64.argmax()], x_int64.min())
    hax1.xaxis.set_visible(False)
    hax1.yaxis.set_visible(False)
    
    im128 = ax[1].imshow(
        dp_128,
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect="auto",
        origin="lower",
        norm=norm,
    )
    ax[1].set_aspect(1)

    divider = make_axes_locatable(ax[1])
    vax2 = divider.append_axes("right", size="20%", pad=0.1)
    fig.add_axes(vax2)
    t_int128 = dp_128.sum(axis=1)
    vax2.plot(t_int128[t_int128>0], ts[t_int128>0], color='red', marker='o', linestyle='none', markersize=.1)
    vax2.plot(t_int128[t_int128<0], ts[t_int128<0], color='blue', marker='o', linestyle='none', markersize=.1)
    vax2.set_ylim(0, 1)
    vax2.set_xlim(-0.03, 0.03)
    vax2.xaxis.set_visible(False)
    vax2.yaxis.set_visible(False)

    hax2 = divider.append_axes("top", size="20%", pad=0.1)
    fig.add_axes(hax2)
    x_int128 = dp_128.sum(axis=0)
    hax2.plot(pxs[x_int128>0], x_int128[x_int128>0], color='red', marker='o', linestyle='none', markersize=.1)
    hax2.plot(pxs[x_int128<0], x_int128[x_int128<0], color='blue', marker='o', linestyle='none', markersize=.1)
    hax2.set_xlim(0, 1)
    hax2.set_ylim(-0.1, 0.1)
    print(x_int128.max(), x_int128.min())
    hax2.xaxis.set_visible(False)
    hax2.yaxis.set_visible(False)

    # plot colorbar
    cax = fig.add_axes([0.175, 0.95, 0.7, 0.08])
    cb = plt.colorbar(
        im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal"
    )
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(
        r"$\langle p_{s,smooth}-p_{s,rough} \rangle$", labelpad=-24, rotation=0
    )

    plt.savefig(f"figures/phase-info/surface/diff_ps.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/diff_ps.png", dpi=450, transparent=True)
    plt.close()


def plot_difference_spectra(ts, pxs, body):
    dx = 4 / 4096  # Spatial step
    dt = 1 / body[0].shape[0]  # Temporal step

    cc64 = np.fft.fft2(body[0] - body[1])
    cc64 = np.fft.fftshift(cc64)
    cc128 = np.fft.fft2(body[0] - body[2])
    cc128 = np.fft.fftshift(cc128)
    np.log(np.abs(cc64))

    num_rows, num_cols = body[0].shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(num_cols, 4 / 4096))
    freq_y = np.fft.fftshift(np.fft.fftfreq(num_rows, dt))

    extent = [freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    fig.text(0.5, 0.01, r"$k_x$", ha="center", va="center")
    fig.text(0.07, 0.5, r"$f^*$", ha="center", va="center", rotation="vertical")

    ax[0].set_title(r"$\lambda = 1/64$", fontsize=10)
    ax[1].set_title(r"$\lambda = 1/128$", fontsize=10)

    # [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]
    # [[ax.set_xlim(0, 1) for ax in ax[n, :]] for n in range(2)]

    lims = [0, 2.5]

    ax[0].imshow(
        np.log(np.abs(cc64)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("icefire", as_cmap=True),
        aspect="auto",
        origin="lower",
        # norm=norm,
    )
    print(np.log(np.abs(cc128)).max())
    im128 = ax[1].imshow(
        np.log(np.abs(cc128)),
        extent=extent,
        vmin=lims[0],
        vmax=lims[1],
        cmap=sns.color_palette("icefire", as_cmap=True),
        aspect="auto",
        origin="lower",
        # norm=norm,
    )

    # set all axes with symlog
    [ax.set_xscale("symlog") for ax in ax]
    [ax.set_yscale("symlog") for ax in ax]

    # # annotate with a box
    # [[ax.axhspan(2.5, 10, facecolor='grey', alpha=0.2, edgecolor='none') for ax in ax[n, :]] for n in range(2)]
    # [[ax.axvspan(-24, -8, facecolor='grey', alpha=0.2, edgecolor='none') for ax in ax[n, :]] for n in range(2)]
    # [[ax[n, m].plot([-24, -8], [2.5, 2.5], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    # [[ax[n, m].plot([-24, -8], [10, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    # [[ax[n, m].plot([-24, -24], [2.5, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]
    # [[ax[n, m].plot([-8, -8], [2.5, 10], color='green', linewidth=1) for m in range(2)] for n in range(2)]

    # # plot colorbar
    # cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    # cb = plt.colorbar(im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal")
    # cb.ax.xaxis.tick_top()  # Move ticks to top
    # cb.ax.xaxis.set_label_position('top')  # Move label to top
    # cb.set_label(r"$PSD(\langle p_{s,smooth}-p_{s,rough} \rangle)$", labelpad=-25, rotation=0)

    plt.savefig(f"figures/phase-info/surface/diff_spec.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/diff_spec.png", dpi=450, transparent=True)


if __name__ == "__main__":
    # extract arrays from fort.7
    lams = [1e9, 1 / 64, 1 / 128]
    labs = [f"$\lambda = 1/{int(1/lam)}$" for lam in lams]
    cases = ["test/span64", "0.001/64", "0.001/128"]  # , "0.001/32"]#, "0.001/32"]
    offsets = [0, 2, 4, 6, 8]
    colours = sns.color_palette("colorblind", 7)
    ts, pxs, body = load_cp(cases)
    # plot_cp(ts, pxs, body)
    plot_cp_diff(ts, pxs, body)
    # plot_difference_spectra(ts, pxs, body)
