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


def fwarp(t, x):
    xp = min(max(x, 0.0), 1.0)
    return (
        -0.5
        * (0.28 * xp**2 - 0.13 * xp + 0.05)
        * np.sin(2 * np.pi * (t - (1.42 * xp)))
    )


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


def point_at_distance(x1, y1, nx, ny, s=0.1):
    """
    Calculates a point that is 's' units away from the point (x1, x2)
    in the direction given by the vector (nx, ny).

    :param x1: x-coordinate of the initial point
    :param y1: y-coordinate of the initial point
    :param nx: x-component of the direction vector
    :param ny: y-component of the direction vector
    :param s: distance to move from the initial point
    :return: Tuple representing the new point coordinates
    """
    # Initial point and direction vector
    p = np.array([x1, y1])
    d = np.array([nx, ny])

    # Normalize the direction vector
    d_norm = d / np.linalg.norm(d)

    # Calculate the new point
    p_new = p + s * d_norm
    return p_new


def tangental_profiles(case, prof_dist=0.05):
    if os.path.isfile(f"data/0.001/{case}/unmasked/s_profile.npy") and os.path.isfile(
        f"data/0.001/{case}/unmasked/omega_profile.npy"
    ):
        omega_profile = np.load(f"data/0.001/{case}/unmasked/omega_profile.npy")
        s_profile = np.load(f"data/0.001/{case}/unmasked/s_profile.npy")
        ts = np.arange(0, s_profile.shape[0], 1)
        pxs = np.linspace(0, 1, s_profile.shape[1])
        return ts, pxs, s_profile, omega_profile
    else:
        bod = np.load(f"data/0.001/{case}/unmasked/uvp.npy")
        pxs = np.linspace(-0.35, 2, bod.shape[1])
        bod_mask = np.where((pxs > 0) & (pxs < 1))
        pys = np.linspace(-0.35, 0.35, bod.shape[2])
        num_points = int(prof_dist * 4096)

        ts = np.arange(0, bod.shape[-1], 1)
        s_profile = np.empty((ts.size, pxs.size, num_points))
        omega_profile = np.empty((ts.size, pxs.size, num_points))
        for idt, t_idx in tqdm(enumerate(ts), total=ts.size):
            t = (t_idx / 200) % 1
            # Interpolation function for the body
            f_u = RegularGridInterpolator(
                (pxs, pys),
                bod[0, :, :, t_idx].astype(np.float64),
                bounds_error=False,
                fill_value=1,
            )
            f_v = RegularGridInterpolator(
                (pxs, pys),
                bod[1, :, :, t_idx].astype(np.float64),
                bounds_error=False,
                fill_value=1,
            )
            omega = np.gradient(
                bod[1, :, :, t_idx], pxs, edge_order=2, axis=0
            ) - np.gradient(bod[0, :, :, t_idx], pys, edge_order=2, axis=1)
            f_omega = RegularGridInterpolator(
                (pxs, pys), omega.astype(np.float64), bounds_error=False, fill_value=1
            )
            y_surface = np.array([naca_warp(xp) for xp in pxs]) - np.array(
                [fwarp(t, xp) for xp in pxs]
            )

            nx, ny = normal_to_surface(pxs, t)
            sx, sy = tangent_to_surface(pxs, t)

            for pidx in range(pxs.size):
                x1, y1 = pxs[pidx], y_surface[pidx]
                x2, y2 = point_at_distance(x1, y1, nx[pidx], ny[pidx], s=prof_dist)
                line_points = np.linspace(
                    np.array([x1, y1]), np.array([x2, y2]), num_points
                )
                u_profile = f_u(line_points)
                v_profile = f_v(line_points)
                s_profile[idt, pidx] = u_profile * sx[pidx] + v_profile * sy[pidx]
                omega_profile[idt, pidx] = f_omega(line_points)

        s_profile = s_profile[:, bod_mask[0], :]
        omega_profile = omega_profile[:, bod_mask[0], :]
        np.save(f"data/0.001/{case}/unmasked/s_profile.npy", s_profile)
        np.save(f"data/0.001/{case}/unmasked/omega_profile.npy", omega_profile)
        return ts, pxs[bod_mask[0]], s_profile, omega_profile


def delta_tilde(profile, normal_dis):
    delta = np.empty(profile.shape[0])
    for idxp in range(profile.shape[0]):
        mask = profile[idxp] > 0.99 * profile[idxp][-1]
        delta[idxp] = normal_dis[mask][0]
    return delta


def delta_shear(profile, normal_dis):
    du_dn = np.gradient(profile, normal_dis, edge_order=2, axis=1)
    d2u_dn2 = du_dn  # np.gradient(du_dn, normal_dis, edge_order=2, axis=1)
    d2u_dn2 = d2u_dn2 / np.ptp(d2u_dn2, axis=1)[:, None]
    d2u_dn2 = gaussian_filter1d(d2u_dn2, sigma=1, axis=1)
    delta = np.empty(d2u_dn2.shape[0])
    for idxp in range(d2u_dn2.shape[0]):
        mask = abs(d2u_dn2[idxp]) > 0.01
        delta[idxp] = normal_dis[mask][-1]
    return delta


def delta_yomega(profile, normal_dis):
    delta = np.empty(profile.shape[0])
    y_profile = -profile * normal_dis
    for idxp in range(y_profile.shape[0]):
        target = 0.02 * np.max(y_profile[idxp])
        mask = y_profile[idxp] > target
        inside = np.where(normal_dis == normal_dis[mask][-1])[0] + 1
        delta[idxp] = normal_dis[inside]
    return delta


def plot_profile_ident(cases):
    fig, axs = plt.subplots(len(cases), 1, figsize=(6, 9), sharex=True, sharey=True)
    fig.text(0.5, 0.0, r"$x_{n=0}+u_s/10$", ha="center")
    fig.text(0.0, 0.5, r"$n$", va="center", rotation="vertical")
    # fig.set_xlabel(r"$x_{n=0}+u_s/10$")
    # ax.set_ylabel(r"$n$")

    prof_dist = 0.05
    num_points = int(prof_dist * 4096)
    axs[0].set_xlim([0, 1.2])
    axs[0].set_ylim([0, prof_dist])
    for idax, case in enumerate(cases):
        ax = axs[idax]
        ax.set_title(f"$\lambda=1/{case}$", fontsize=9)
        ts, pxs, s_profile, omega_profile = tangental_profiles(
            case, prof_dist=prof_dist
        )

        normal_dis = np.linspace(0, prof_dist, num_points)
        x_samples = np.array([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        n_profs = x_samples.size
        closest_index = [np.argmin(abs(pxs - x)) for x in x_samples]

        for idt, t_idx in tqdm(enumerate(ts), total=ts.size, desc="Plot Loop"):
            for idxp in range(n_profs):
                profile = (
                    s_profile[idt, closest_index[idxp]]
                    - s_profile[idt, closest_index[idxp], 0]
                )  # Subtract the body velocity
                ax.plot(
                    profile / 10 + x_samples[idxp],
                    normal_dis,
                    color="grey",
                    linewidth=0.15,
                    alpha=0.025,
                )

                # range_omega = np.ptp(omega_profile[idt, closest_index[idxp]])
                # ax.plot(0.1*omega_profile[idt, closest_index[idxp]]/range_omega+x_samples[idxp], normal_dis, color='blue', linewidth=0.1, alpha=0.02)

                # delt = delta_tilde(s_profile[idt], normal_dis)[closest_index[idxp]]
                # nrst_x = np.argmin(abs(normal_dis-delt))
                # ax.plot(profile[nrst_x]/10+x_samples[idxp], delt, color='green', marker='o', markersize=2, alpha=0.5, markeredgewidth=0)

                # delt = delta_shear(s_profile[idt], normal_dis)[closest_index[idxp]]
                # nrst_x = np.argmin(abs(normal_dis-delt))
                # ax.plot(profile[nrst_x]/10+x_samples[idxp], delt, color='yellow', marker='o', markersize=2, alpha=0.5, markeredgewidth=0)

                delt = delta_yomega(omega_profile[idt], normal_dis)[closest_index[idxp]]
                nrst_x = np.argmin(abs(normal_dis - delt))
                ax.plot(
                    profile[nrst_x] / 10 + x_samples[idxp],
                    delt,
                    color="purple",
                    marker="o",
                    markersize=2,
                    alpha=0.5,
                    markeredgewidth=0,
                )

        # Find the time avg
        stationary = s_profile - s_profile[:, :, 0][:, :, None]
        avg_profile = np.mean(stationary, axis=0)

        for idxp in range(n_profs):
            ax.plot(
                avg_profile[closest_index[idxp]] / 10 + x_samples[idxp],
                normal_dis,
                color="red",
                linewidth=0.5,
                alpha=0.6,
                ls="-.",
            )

    leg_elements = [
        plt.Line2D([0], [0], color="grey", linewidth=0.5, alpha=0.6, label=r"$ u_s $"),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linewidth=0.5,
            alpha=0.6,
            ls="-.",
            label=r"$\langle u_s \rangle$",
        ),
        # plt.Line2D([0], [0], color='green', marker='o', markersize=3, alpha=0.5, markeredgewidth=0, linestyle='None', label=r"$\delta_{\tilde{u_s}}$"),
        # plt.Line2D([0], [0], color='yellow', marker='o', markersize=3, alpha=0.5, markeredgewidth=0, linestyle='None', label=r"$\delta_{\partial u_s/\partial dn}$"),
        plt.Line2D(
            [0],
            [0],
            color="purple",
            marker="o",
            markersize=3,
            alpha=0.5,
            markeredgewidth=0,
            linestyle="None",
            label=r"$\delta$",
        ),
    ]
    axs[0].legend(
        handles=leg_elements, loc="upper left", ncol=1, frameon=False, fontsize=9
    )

    fig.tight_layout()
    plt.savefig(f"figures/profiles/time_profile_ident.pdf")
    plt.savefig(f"figures/profiles/time_profile_ident.png", dpi=800)


def plot_smooth_profiles():
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
    fig.text(0.5, 0.01, r"$x_{n=0}+u_s/10$", ha="center")
    fig.text(0.01, 0.5, r"$n$", va="center", rotation="vertical")
    # fig.set_xlabel(r"$x_{n=0}+u_s/10$")
    # ax.set_ylabel(r"$n$")

    prof_dist = 0.05
    num_points = int(prof_dist * 4096)
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, prof_dist])

    ts, pxs, s_profile, omega_profile = tangental_profiles(0, prof_dist=prof_dist)

    normal_dis = np.linspace(0, prof_dist, num_points)
    x_samples = np.array([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_profs = x_samples.size
    closest_index = [np.argmin(abs(pxs - x)) for x in x_samples]

    delta_omega = np.empty((ts.size, pxs.size))
    for idt, t_idx in tqdm(enumerate(ts), total=ts.size, desc="Plot Loop"):
        delta_omega[idt] = delta_yomega(omega_profile[idt], normal_dis)
        for idxp in range(n_profs):
            profile = (
                s_profile[idt, closest_index[idxp]]
                - s_profile[idt, closest_index[idxp], 0]
            )  # Subtract the body velocity
            ax.plot(
                profile / 10 + x_samples[idxp],
                normal_dis,
                color="grey",
                linewidth=0.1,
                alpha=0.02,
            )

    # Find the time avg
    stationary = s_profile - s_profile[:, :, 0][:, :, None]
    avg_profile = np.mean(stationary, axis=0)

    for idxp in range(n_profs):
        ax.plot(
            avg_profile[closest_index[idxp]] / 10 + x_samples[idxp],
            normal_dis,
            color="red",
            linewidth=0.5,
            alpha=0.6,
            ls="-.",
        )

    avg_delta = np.mean(delta_omega, axis=0)
    for idxx in range(pxs.size):
        nrstx = np.argmin(abs(avg_delta[idxx] - normal_dis))
        ax.plot(
            avg_profile[idxx][nrstx] / 10 + pxs[idxx],
            avg_delta[idxx],
            color="purple",
            marker="o",
            markersize=2,
            alpha=0.5,
            markeredgewidth=0,
        )

    leg_elements = [
        plt.Line2D([0], [0], color="grey", linewidth=0.5, alpha=0.6, label=r"$ u_s $"),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linewidth=0.5,
            alpha=0.6,
            ls="-.",
            label=r"$\langle u_s \rangle$",
        ),
        plt.Line2D(
            [0], [0], color="purple", ls="--", linewidth=1, label=r"$\overline{\delta}$"
        ),
    ]
    ax.legend(handles=leg_elements, loc="upper left", ncol=1, frameon=False, fontsize=8)
    plt.savefig(f"figures/profiles/smooth_profiles.pdf")
    plt.savefig(f"figures/profiles/smooth_profiles.png", dpi=800)


def plot_deltas(cases):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    ax = axs[0]
    # fig.text(0.5, 0.01, r"$x_{n=0}+u_s/10$", ha='center')
    # fig.text(0.01, 0.5, r"$n$", va='center', rotation='vertical')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\overline{\delta}, \sigma_{\delta}$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$\mathcal{A}_{\delta}$")

    prof_dist = 0.05
    num_points = int(prof_dist * 4096)
    ax.set_xlim([0, 1])

    colours = sns.color_palette("colorblind", 5)
    c_order = [2, 0, 3, 4, 1]
    # c_order = [2, 4, 1]

    for idcase, case in enumerate(cases):
        ts, pxs, s_profile, omega_profile = tangental_profiles(
            case, prof_dist=prof_dist
        )
        normal_dis = np.linspace(0, prof_dist, num_points)
        # delta_omega = np.empty((ts.size, pxs.size))
        # for idt, t_idx in tqdm(enumerate(ts), total=ts.size, desc="delta_omega Loop"):
        #     delta_omega[idt] = delta_yomega(omega_profile[idt], normal_dis)
        # np.save(f"data/0.001/{case}/unmasked/delta_omega.npy", delta_omega)
        delta_omega = np.load(f"data/0.001/{case}/unmasked/delta_omega.npy")

        avg_delta = np.mean(delta_omega, axis=0)
        ax.plot(
            pxs,
            avg_delta,
            color=colours[c_order[idcase]],
            linewidth=0.5,
            label=rf"$\lambda=1/{case}$",
        )
        # Plot fluctuations
        ax.plot(
            pxs,
            np.std(delta_omega, axis=0),
            color=colours[c_order[idcase]],
            linewidth=0.5,
            ls="--",
        )
        # Plot autocorrelation
        autoco = [auto_corr(delta_omega[:, idx]) for idx in range(pxs.size)]
        savgoled = savgol_filter(autoco, 7, 3)
        axs[1].plot(
            pxs, savgoled, color=colours[c_order[idcase]], linewidth=0.5, ls="-."
        )

    axs[1].set_xlim([0.5, 1])
    mean_lines = [
        Line2D([0], [0], color=colours[c_order[idcase]], linewidth=1)
        for idcase, case in enumerate(cases)
    ]
    color_legend = axs[1].legend(
        mean_lines,
        [rf"$\lambda=1/{case}$" for case in cases],
        loc="lower left",
        frameon=False,
        fontsize=9,
    )
    style_legend_lines = [
        Line2D([0], [0], color="black", linewidth=0.5),
        Line2D([0], [0], color="black", linewidth=0.5, ls="--"),
        Line2D([0], [0], color="black", linewidth=0.5, ls="-."),
    ]
    ax.legend(
        style_legend_lines,
        [r"$\overline{\delta}$", r"$\sigma_{\delta}$"],
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    plt.savefig(f"figures/profiles/delta_x.pdf")
    plt.savefig(f"figures/profiles/delta_x.png", dpi=800)


def plot_Delta_deltas(cases):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    ax = axs.ravel()
    fig.text(0.5, 0.01, r"$x$", ha="center")
    # fig.text(0.01, 0.5, r"$\varphi$", va='center', rotation='vertical')
    ax[0].set_ylabel(r"$\varphi$")
    ax[2].set_ylabel(r"$\varphi$")

    prof_dist = 0.05
    num_points = int(prof_dist * 4096)

    colours = sns.color_palette("colorblind", 5)
    c_order = [2, 0, 3, 4, 1]

    delta_omegas = []
    for idcase, case in enumerate(cases):
        ts, pxs, s_profile, omega_profile = tangental_profiles(
            case, prof_dist=prof_dist
        )
        # normal_dis = np.linspace(0, prof_dist, num_points)
        # delta_omega = np.empty((ts.size, pxs.size))
        # for idt, t_idx in tqdm(enumerate(ts), total=ts.size, desc="delta_omega Loop"):
        #     delta_omega[idt] = delta_yomega(omega_profile[idt], normal_dis)
        # np.save(f"data/0.001/{case}/unmasked/delta_omega.npy", delta_omega)
        delta_omegas.append(
            np.load(f"data/0.001/{case}/unmasked/delta_omega.npy")[:200]
        )

    lims = [-1, 1]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
    _cmap = custom_cmap()
    for idx in range(4):
        smooth = delta_omegas[0]
        diff = delta_omegas[1 + idx] - smooth
        ax[idx].set_title(rf"$\lambda=1/{cases[1+idx]}$", fontsize=9)
        cs = ax[idx].imshow(
            diff*1e3,
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
    cb.set_label(r"$ \Delta \delta \times 10^3 $", labelpad=-25, rotation=0)

    # fig.tight_layout()
    plt.savefig(f"figures/profiles/Delta_deltas.pdf")
    plt.savefig(f"figures/profiles/Delta_deltas.png", dpi=800)


def custom_cmap():
    green_to_white = sns.color_palette("Purples_r", n_colors=128)
    white_to_red = sns.color_palette("Blues_r", n_colors=128)

    # Concatenate the palettes, placing white in the middle
    full_palette = np.vstack((green_to_white, white_to_red[::-1]))

    # Convert the concatenated palette to a colormap
    cmap = LinearSegmentedColormap.from_list("GreenWhiteRed", full_palette)
    return cmap


def plot_Delta_delta_128():
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varphi$")

    dos = np.load(f"data/0.001/0/unmasked/delta_omega.npy")[:200]
    do128 = np.load(f"data/0.001/128/unmasked/delta_omega.npy")[:200]
    lims = [-0.001, 0.001]
    norm = TwoSlopeNorm(vmin=lims[0], vcenter=0, vmax=lims[1])
    _cmap = custom_cmap()
    diff = do128 - dos
    diff = gaussian_filter1d(diff, sigma=1, axis=0)
    diff = gaussian_filter1d(diff, sigma=1, axis=1)
    cs = ax.imshow(
        diff,
        extent=[0, 1, 0, 1],
        cmap=_cmap,
        aspect="auto",
        origin="lower",
        norm=norm,
        # vmin=lims[0],
        # vmax=lims[1],
    ) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.21)
    cb = plt.colorbar(
        cs, cax=cax, orientation="vertical", ticks=np.linspace(lims[0], lims[1], 5)
    )
    cb.set_label(r"$ \Delta \delta $", labelpad=-56, rotation=0)

    # fig.tight_layout()
    plt.savefig(f"figures/profiles/Delta_delta_128.pdf")


def auto_corr(x):
    """
    Compute the autocorrelation of the signal x
    x: (T, N) where T is the number of time steps and N is the number of data points.
    return:
    """
    x1 = x[:-1] - x[:-1].mean()
    x2 = x[1:] - x[1:].mean()
    cor = np.sum(x1 * x2)
    var1 = np.sum((x1) ** 2)
    var2 = np.sum((x2) ** 2)
    r = cor / (np.sqrt(var1 * var2))
    return r


if __name__ == "__main__":
    cases = [0, 16, 32, 64, 128]
    case = cases[0]
    # plot_smooth_profiles()
    # cases = [0, 64, 128]
    # plot_deltas(cases)
    plot_Delta_deltas(cases)
    plot_Delta_delta_128()
    # plot_profile_ident(cases)
