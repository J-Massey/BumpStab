import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import os
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")


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

    return (
        a * xp
        + b * xp**2
        + c * xp**3
        + d * xp**4
        + e * xp**5
        + f * xp**6
        + g * xp**7
        + h * xp**8
        + i * xp**9
        + j * xp**10
    )


def fwarp(t: float, pxs):
    if isinstance(pxs, float):
        x = pxs
        xp = min(max(x, 0.0), 1.0)
        return -0.5*(0.28 * xp**2 - 0.13 * xp + 0.05) * np.sin(2*np.pi*(t - (1.42* xp)))
    else:
        return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def kappa(pxs, ts):
    y = np.empty((len(pxs), len(ts)))
    for idx, t in enumerate(ts):
        y[:, idx] = np.array([naca_warp(xp) for xp in pxs])
    dy_dt = np.gradient(y, ts, axis=1)
    d2y_dt2 = np.gradient(dy_dt, ts, axis=1)
    d2y_d2t = np.gradient(np.gradient(y, ts, axis=1), ts, axis=1)
    dy_dx = np.gradient(y, pxs, axis=0)
    d2y_dx2 = np.gradient(dy_dx, pxs, axis=0)
    return -d2y_dx2


def normal_to_surface(x: np.ndarray, t):
    y = np.array([naca_warp(xp) for xp in x]) - np.array([fwarp(t, xp) for xp in x])
    y = y * 1  # Scale the y-coordinate to get away from the body

    df_dx = np.gradient(y, x, edge_order=2)
    df_dy = -1

    # Calculate the normal vector to the surface
    mag = np.sqrt(df_dx**2 + df_dy**2)
    nx = -df_dx / mag
    ny = -df_dy / mag
    return nx, ny


def tangent_to_surface(x: np.ndarray, t):
    # Evaluate the surface function y(x, t)
    y = np.array([naca_warp(xp) for xp in x]) - np.array([fwarp(t, xp) for xp in x])

    # Calculate the gradient dy/dx
    df_dx = np.gradient(y, x, edge_order=2)

    # The tangent vector components are proportional to (1, dy/dx)
    tangent_x = 1 / np.sqrt(1 + df_dx**2)
    tangent_y = df_dx / np.sqrt(1 + df_dx**2)

    return tangent_x, tangent_y


def pressure_value(case):
    if os.path.isfile(
        f"data/0.001/{case}/unmasked/pressure_profile.npy"
    ):
        pressure_profile = np.load(f"data/0.001/{case}/unmasked/pressure_profile.npy")
        ts = np.arange(0, pressure_profile.shape[0], 1)
        pxs = np.linspace(0, 1, pressure_profile.shape[1])
        return ts, pxs, pressure_profile
    else:
        bod = np.load(f"data/0.001/{case}/unmasked/uvp.npy")
        pxs = np.linspace(-0.35, 2, bod.shape[1])
        bod_mask = np.where((pxs > 0) & (pxs < 1))
        pys = np.linspace(-0.35, 0.35, bod.shape[2])

        ts = np.arange(0, bod.shape[-1], 1)
        pressure_profile = np.empty((ts.size, pxs.size))
        for idt, t_idx in tqdm(enumerate(ts), total=ts.size):
            t = (t_idx / 200) % 1
            # Interpolation function for the body
            f_p = RegularGridInterpolator(
                (pxs, pys),
                bod[2, :, :, t_idx].astype(np.float64),
                bounds_error=False,
                fill_value=1,
            )
            y_surface = np.array([naca_warp(xp) for xp in pxs]) - np.array(
                [fwarp(t, xp) for xp in pxs]
            )

            nx, ny = normal_to_surface(pxs, t)
            sx, sy = tangent_to_surface(pxs, t)

            for pidx in range(pxs.size):
                x1, y1 = pxs[pidx], y_surface[pidx]
                pressure_profile[idt, pidx] = f_p(np.array([x1, y1]))

        pressure_profile = pressure_profile[:, bod_mask[0]]
        np.save(f"data/0.001/{case}/unmasked/pressure_profile.npy", pressure_profile)
        return ts, pxs[bod_mask[0]], pressure_profile


def plot_surface_pressures(cases):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    fig.text(0.5, 0.01, r"$x$", ha="center")
    # fig.text(0.01, 0.5, r"$\varphi$", va='center', rotation='vertical')
    ax[0].set_ylabel(r"$\varphi$")
    ax[2].set_ylabel(r"$\varphi$")

    colours = sns.color_palette("colorblind", 5)
    c_order = [2, 0, 3, 4, 1]

    pressure_profiles = []
    for idcase, case in enumerate(cases):
        ts, pxs, pressure_profile = pressure_value(case)
        pressure_profiles.append(pressure_profile)
        print(pressure_profile.std(), pressure_profile.min())

    lims = [-0.25, 0.25]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
    _cmap = custom_cmap()
    for idx in range(4):
        ax[idx].set_title(rf"$\lambda=1/{cases[idx]}$", fontsize=9)
        cs = ax[idx].imshow(
            -pressure_profiles[idx][:200],
            extent=[0, 1, 0, 1],
            cmap=_cmap,
            aspect="auto",
            origin="lower",
            norm=norm,
            # vmin=lims[0],
            # vmax=lims[1],
        )
        ax[idx].set_aspect(1)
        ax[idx].set_ylim([0, 1])

    cax = fig.add_axes([0.175, 0.92, 0.7, 0.03])
    cb = plt.colorbar(
        cs, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5)
    )
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(r"$ -p\cdot\hat{n} \times 10^3 $", labelpad=-25, rotation=0)

    # fig.tight_layout()
    plt.savefig(f"figures/variable-roughness/pressure-n.pdf")
    plt.savefig(f"figures/variable-roughness/pressure-n.png", dpi=800)

def plot_smooth_surface_pressure():
    fig, ax = plt.subplots(figsize=(6, 6), sharex=True, sharey=True)
    ax.set_xlabel(r"$x$")
    # fig.text(0.01, 0.5, r"$\varphi$", va='center', rotation='vertical')
    ax.set_ylabel(r"$\varphi$")

    ts, pxs, pressure_profile = pressure_value(0)

    lims = [-2.5, 2.5]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
    _cmap = custom_cmap()
    ax.set_title(rf"$\lambda=1/0$", fontsize=9)

    # cs = ax.imshow(
    #     pressure_profile[:200]/1024*1e3,
    #     extent=[0, 1, 0, 1],
    #     cmap=_cmap,
    #     aspect="auto",
    #     origin="lower",
    #     norm=norm,
    # )

    cs = ax.contourf(
        pxs,
        np.linspace(0, 1, 200),
        pressure_profile[:200]/1024*1e4,
        cmap=_cmap,
        norm=norm,
        levels=np.linspace(lims[0], lims[1], 22),
        extend='both',
    )

    np.save(f"data/variable-roughness/pressure_profile.npy", pressure_profile[:200]/1024*1e4)

    ax.set_aspect(1)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.05)
    cb = plt.colorbar(
        cs, cax=cax, orientation="vertical", ticks=np.linspace(lims[0], lims[1], 5)
    )
    cb.set_label(r"$ |\vec{f}| \quad \times 10^4 $", labelpad=-49, rotation=90)

    ts = np.linspace(0, 1, 200)
    ax.contour(pxs, ts, kappa(pxs, ts).T, levels=[2.5], colors='k', linewidths=0.25, linestyles='--')
    ax.contourf(pxs, ts, kappa(pxs, ts).T, levels=[0, np.inf], colors='gray', alpha=0.4)

    # fig.tight_layout()
    plt.savefig(f"figures/variable-roughness/pressure-n.pdf")
    plt.savefig(f"figures/variable-roughness/pressure-n.png", dpi=800)


def plot_smooth_dp_dn():
    fig, ax = plt.subplots(figsize=(6, 6), sharex=True, sharey=True)
    ax.set_xlabel(r"$x$")
    # fig.text(0.01, 0.5, r"$\varphi$", va='center', rotation='vertical')
    ax.set_ylabel(r"$\varphi$")

    ts, pxs, pressure_profile = pressure_value(0)

    lims = [-2.5, 2.5]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
    _cmap = custom_cmap()
    ax.set_title(rf"$\lambda=1/0$", fontsize=9)

    cs = ax.contourf(
        pxs,
        np.linspace(0, 1, 200),
        pressure_profile[:200]/1024*1e4,
        cmap=_cmap,
        norm=norm,
        levels=np.linspace(lims[0], lims[1], 22),
        extend='both',
    )

    ax.set_aspect(1)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.05)
    cb = plt.colorbar(
        cs, cax=cax, orientation="vertical", ticks=np.linspace(lims[0], lims[1], 5)
    )
    # cb.ax.xaxis.tick_top()  # Move ticks to top
    # cb.ax.xaxis.set_label_position("top")  # Move label to top
    cb.set_label(r"$ -p\cdot\hat{n} \times 10^4 $", labelpad=-49, rotation=90)

    # fig.tight_layout()
    plt.savefig(f"figures/variable-roughness/pressure-n.pdf")
    plt.savefig(f"figures/variable-roughness/pressure-n.png", dpi=800)


def custom_cmap():
    green_to_white = sns.color_palette("Purples_r", n_colors=128)
    white_to_red = sns.color_palette("Blues_r", n_colors=128)

    # Concatenate the palettes, placing white in the middle
    full_palette = np.vstack((green_to_white, white_to_red[::-1]))

    # Convert the concatenated palette to a colormap
    cmap = LinearSegmentedColormap.from_list("GreenWhiteRed", full_palette)
    return cmap


if __name__ == "__main__":
    cases = [0, 32, 64, 128]
    # case = cases[]
    plot_smooth_surface_pressure()
