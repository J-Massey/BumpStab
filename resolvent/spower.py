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
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import interp1d
from scipy.signal import welch, savgol_filter
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")


def custom_cmap():
    green_to_white = sns.color_palette("Greens_r", n_colors=128)
    white_to_red = sns.color_palette("Reds_r", n_colors=128)

    # Concatenate the palettes, placing white in the middle
    full_palette = np.vstack((green_to_white, white_to_red[::-1]))

    # Convert the concatenated palette to a colormap
    cmap = LinearSegmentedColormap.from_list('GreenWhiteRed', full_palette)
    return cmap


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
    if case==16:
        span = 256
    elif case==32:
        span = 192
    elif case==0:
        span = 16
    else:
        span = 64
    normalise = 0.1 * 4096 * span * 4 / 2
    return normalise


def splineit(tr, cp, T=2, bodshape=1024, sigma=2):
    start = np.where(cp[0] != 0)[0][0]
    bod = cp[:, start : start + bodshape]
    bod = gaussian_filter1d(bod, sigma=sigma, axis=1)
    bod = gaussian_filter1d(bod, sigma=sigma, axis=0)
    sorted_indices = np.argsort(tr % T)
    unique_x, unique_indices = np.unique((tr % T)[sorted_indices], return_index=True)
    sorted_y = bod[sorted_indices]
    unique_y = sorted_y[unique_indices]
    cs = CubicSpline(unique_x, unique_y)
    y_spline = cs(np.linspace(0.001, 1.999, 10000))
    return y_spline



def load_phase_avg_cp(cases):
    bodshape = 1024
    T = 2
    ts = np.linspace(0.001, 0.999, 5000)
    if os.path.isfile("data/cp_phase_map.npy") and os.path.isfile("data/cp_instantaneous.npy"):
        phase_avg = np.load("data/cp_phase_map.npy")
        instantaneous = np.load("data/cp_instantaneous.npy")
    else:
        phase_avg = []; instantaneous = []
        for case in tqdm(cases):
            tr, _ = read_forces(f"data/0.001/{case}/spressure/fort.9", "cp", "")

            cp = np.genfromtxt(f"data/0.001/{case}/spressure/fort.1")  # Load the 1D array
            y_spline_top = splineit(tr, cp)/fnorm(case)
            onetop, twotop = y_spline_top[:int(len(y_spline_top)//2)], y_spline_top[int(len(y_spline_top)//2):]

            cp = np.genfromtxt(f"data/0.001/{case}/spressure/fort.2")  # Load the 1D array
            y_spline_bot = splineit(tr, cp)/fnorm(case)
            y_spline_bot = np.roll(y_spline_bot, y_spline_bot.shape[0]//(2*T), axis=0)
            onebot, twobot = y_spline_bot[:int(len(y_spline_bot)//2)], y_spline_bot[int(len(y_spline_bot)//2):]

            instantaneous.append(np.array([onetop, twotop, onebot, twobot]))
            phase_avg.append((onetop + twotop + onebot + twobot)/4)

        np.save("data/cp_phase_map.npy", phase_avg)
        np.save("data/cp_instantaneous.npy", instantaneous)
    return ts, np.linspace(0, 1, bodshape), phase_avg, instantaneous


def straight_line(a, b):
    m = (b[1] - a[1]) / (b[0] - a[0])
    c = a[1] - m * a[0]
    points = [c, m + c]
    print(m)
    return points


def plot_cp_smooth():
    fig, ax = plt.subplots(figsize=(4, 4), sharex=True, sharey=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varphi$")

    lims = [-0.002, 0.002]
    norm = TwoSlopeNorm(vcenter=0, vmin=lims[0], vmax=lims[1])

    phase_avg = np.load("data/cp_phase_map.npy")

    cs = ax.imshow(
        phase_avg[0],
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower',
        norm=norm,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.05)
    cb = plt.colorbar(
        cs, cax=cax, orientation="vertical", ticks=np.linspace(lims[0], lims[1], 5)
    )
    # cb.ax.xaxis.tick_top()  # Move ticks to top
    # cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(r"$ c_P $", labelpad=-49, rotation=90)

    plt.savefig(f"figures/variable-roughness/spressure.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/variable-roughness/spressure.png", dpi=450, transparent=True)
    plt.close()


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


    ax[0,1].imshow(
        body[4],
        extent=[0, 1, 0, 1],
        # vmin=lims[0],
        # vmax=lims[1],
        cmap=sns.color_palette("seismic", as_cmap=True),
        aspect='auto',
        origin='lower',
        norm=norm,
    )

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
    cb.set_label(r"$ c_P $", labelpad=-25, rotation=0)

    plt.savefig(f"figures/phase-info/surface/spressure.pdf", dpi=450, transparent=True)
    plt.savefig(f"figures/phase-info/surface/spressure.png", dpi=450, transparent=True)
    plt.close()

def plot_lines():
    coords = [
        [0.8356890459363957, 0.034615384615384936],
        [0.8851590106007066, 0.18653846153846176],
        [0.8657243816254414, 0.02499999999999991],
        [0.9204946996466432, 0.18269230769230838],
        [0.9540636042402826, 0.20000000000000018],
        [0.9770318021201414, 0.2692307692307696],
    ]
    for i in range(0, len(coords), 2):
        ap, bp = coords[i], coords[i+1]
        points = straight_line(ap, bp)



def vertical_integral(fig, divider, t_ints):
    vax = divider.append_axes("right", size="20%", pad=0.05)
    fig.add_axes(vax)
    for t_int in t_ints:
        vax.plot(t_int[t_int>0], ts[t_int>0], color='red', marker='o', linestyle='none', markersize=.1)
        vax.plot(t_int[t_int<0], ts[t_int<0], color='green', marker='o', linestyle='none', markersize=.1)
    vax.plot(t_ints.mean(axis=0), ts, color='k', linewidth=0.5, linestyle='--', alpha=0.8)
    vax.text(1.3, 0.5, r"$ \Delta C_P $", transform=vax.transAxes, ha='center', va='center', fontsize=8, rotation=270)
    vax.yaxis.set_visible(False)
    vax.set_ylim(0, 1)
    vax.set_xlim(-0.03, 0.03)
    vax.set_xticks([-0.03, 0.03])
    vax.set_xticklabels([-0.03, 0.03], fontsize=8)
    vax.xaxis.tick_top()
    return vax


def horizontal_integral(fig, divider, x_ints):
    hax = divider.append_axes("bottom", size="20%", pad=0.05)
    fig.add_axes(hax)
    for x_int in x_ints:
        hax.plot(pxs[x_int>0], x_int[x_int>0], color='red', marker='o', linestyle='none', markersize=.05)
        hax.plot(pxs[x_int<0], x_int[x_int<0], color='green', marker='o', linestyle='none', markersize=.05)
    hax.plot(pxs, x_ints.mean(axis=0), color='k', linewidth=0.5, linestyle='--', alpha=0.8)
    hax.set_xlim(0, 1)
    hax.set_ylim(-0.1, 0.1)
    hax.xaxis.set_visible(False)
    hax.set_yticks([-0.1, 0, 0.1])
    hax.set_yticklabels([-0.1, 0, 0.1], fontsize=8)
    hax.text(0.5, -.3, r"$\overline{\Delta c_P}$", transform=hax.transAxes, ha='center', va='center', fontsize=8)
    return hax
        

def plot_cp_diff(ts, pxs, ph_avg, instant):
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6), sharey=True)
    ax = axs.ravel()
    ax[0].text(-0.15, 0.98, r"(a)", transform=ax[0].transAxes)
    ax[1].text(-0.15, 0.98, r"(b)", transform=ax[1].transAxes)
    ax[2].text(-0.15, 0.98, r"(c)", transform=ax[2].transAxes)
    ax[3].text(-0.15, 0.98, r"(d)", transform=ax[3].transAxes)
    
    fig.text(0.5, 0.07, r"$x$", ha="center", va="center")
    ax[0].set_ylabel(r"$\varphi$")
    ax[2].set_ylabel(r"$\varphi$")

    ax[0].text(0.1, 0.85, r"$\lambda = 1/16$", fontsize=10)
    ax[1].text(0.1, 0.85, r"$\lambda = 1/32$", fontsize=10)
    ax[2].text(0.1, 0.85, r"$\lambda = 1/64$", fontsize=10)
    ax[3].text(0.1, 0.85, r"$\lambda = 1/128$", fontsize=10)

    for ax_id in ax[:-2]:
        ax_id.xaxis.tick_top()
        ax_id.xaxis.set_label_position('top')

        # Set major ticks and their labels at the top
        ax_id.set_xticks([0.25, 0.5, 0.75])
        ax_id.set_xticklabels([0.25, 0.5, 0.75])

        # Set ticks at both top and bottom but no labels at the bottom
        ax_id.xaxis.set_tick_params(which='both', top=True, bottom=True)
        ax_id.xaxis.set_tick_params(which='both', labelbottom=False)

    for ax_id in ax[2:]:
        ax_id.set_xticklabels([])

    
    ax[0].set_yticks([0.25, 0.5, 0.75])
    ax[0].set_yticklabels([0.25, 0.5, 0.75])

    lims = [-0.0002, 0.0002]
    norm = TwoSlopeNorm(vcenter=0, vmin=lims[0], vmax=lims[1])
    _cmap = custom_cmap()

    dp_64 = - ph_avg[0] + ph_avg[1]
    inst_64 = - instant[0] + instant[1]
    dp_128 = - ph_avg[0] + ph_avg[2]
    inst_128 = - instant[0] + instant[2]
    dp_16 = - ph_avg[0] + ph_avg[3]
    inst_16 = - instant[0] + instant[3]
    dp_32 = - ph_avg[0] + ph_avg[4]
    inst_32 = - instant[0] + instant[4]

    ax[0].imshow(
        dp_16,
        extent=[0, 1, 0, 1],
        cmap=_cmap,
        aspect="auto",
        origin="lower",
        norm=norm,
        
    )
    divider = make_axes_locatable(ax[0])
    hax1 = horizontal_integral(fig, divider, inst_16.sum(axis=1))
    vax1 = vertical_integral(fig, divider, inst_16.sum(axis=2))
    
    ax[1].imshow(
        dp_32,
        extent=[0, 1, 0, 1],
        cmap=_cmap,
        aspect="auto",
        origin="lower",
        norm=norm,
    )
    divider = make_axes_locatable(ax[1])    
    hax2 = horizontal_integral(fig, divider, inst_32.sum(axis=1))
    vax2 = vertical_integral(fig, divider, inst_32.sum(axis=2))

    ax[2].imshow(
        dp_64,
        extent=[0, 1, 0, 1],
        cmap=_cmap,
        aspect="auto",
        origin="lower",
        norm=norm,
    )
    divider = make_axes_locatable(ax[2])    
    hax3 = horizontal_integral(fig, divider, inst_64.sum(axis=1))
    vax3 = vertical_integral(fig, divider, inst_64.sum(axis=2))

    im128 = ax[3].imshow(
        dp_128,
        extent=[0, 1, 0, 1],
        cmap=_cmap,
        aspect="auto",
        origin="lower",
        norm=norm,
    )
    divider = make_axes_locatable(ax[3])    
    hax4 = horizontal_integral(fig, divider, inst_128.sum(axis=1))
    vax4 = vertical_integral(fig, divider, inst_128.sum(axis=2))

    # Grey box where we visualise vorticity
    # ax[3].plot([0.6, 0.6], [0.2, 0.45], color='grey', linewidth=0.5, linestyle='--', alpha=0.7)
    # ax[3].plot([0.6, 1], [0.2, 0.2], color='grey', linewidth=0.5, linestyle='--', alpha=0.7)
    # ax[3].plot([0.6, 1], [0.45, 0.45], color='grey', linewidth=0.5, linestyle='--', alpha=0.7)


    # plot colorbar
    cax = fig.add_axes([0.15, 0.95, 0.7, 0.04])
    cb = plt.colorbar(im128, ticks=np.linspace(lims[0], lims[1], 5), cax=cax, orientation="horizontal")
    tick_labels = [f"{tick:.1f}" for tick in np.linspace(lims[0], lims[1], 5) * 1e4]  # Adjust the format as needed
    cb.set_ticklabels(tick_labels)
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(r"$ \Delta c_P  \quad \times 10^{4}$", labelpad=-25, rotation=0, fontsize=9)

    plt.savefig(f"figures/phase-info/surface/diff_ps.pdf")
    plt.savefig(f"figures/phase-info/surface/diff_ps.png", dpi=450)
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

    plt.savefig(f"figures/phase-info/surface/diff_spec.pdf", dpi=450)
    plt.savefig(f"figures/phase-info/surface/diff_spec.png", dpi=450)


if __name__ == "__main__":
    # extract arrays from fort.7
    lams = [1e9, 1 / 64, 1 / 128, 1/16, 1/32]
    labs = [f"$\lambda = 1/{int(1/lam)}$" for lam in lams]
    cases = [0, 64, 128, 16, 32]
    offsets = [0, 2, 4, 6, 8]
    colours = sns.color_palette("colorblind", 7)
    ts, pxs, ph_avg, instant = load_phase_avg_cp(cases)
    # plot_lines()
    # plot_cp(ts, pxs, ph_avg)
    plot_cp_diff(ts, pxs, ph_avg, instant)
    # plot_difference_spectra(ts, pxs, ph_avg)
