import os
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch, savgol_filter


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{txfonts}')


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

    # u = ux * 0.2 / 2.12, uy * 0.2 / 2.12

    return t, u


def save_fig(save_path):
    plt.savefig(save_path, dpi=700)
    plt.close()


def load_plot(path, ax, omega_span, colour, label):
    gain = np.load(path)
    ax.loglog(
        omega_span / (2 * np.pi),
        np.sqrt(gain[:, 0]),
        color=colour,
        label=label,
        alpha=0.8,
        linewidth=0.7,
    )


def plot_thrust():
    lams = [16, 32, 64, 128]
    cases = [f"0.001/{lam}" for lam in lams]

    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle C_T \rangle $")
    ax.set_xlabel(r"$t/T$")

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="p", direction="x")
        t_new = t % 1
        f = interp1d(t_new, force, fill_value="extrapolate")
        force_av = f(t_sample)


        wrap_indices = np.where(np.diff(t_new) < 0)[0] + 1
        wrap_indices = np.insert(wrap_indices, 0, 0)  # Include the start index
        wrap_indices = np.append(wrap_indices, len(t_new))  # Include the end index


        force_bins = [force[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]
        t_bins = [t_new[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]


        # Calculate the standard deviation of each bin
        force_diff = np.empty((4, t_sample.size))
        for id in range(len(force_bins)):
            f_bins = interp1d(t_bins[id], force_bins[id], fill_value="extrapolate")
            force_bins[id] = f_bins(t_sample)
            force_diff[id] = force_bins[id] - force_av

        ax.plot(
            t_sample,
            force_av,
            color=colours[idx],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
        )

        ax.fill_between(
            t_sample,
            force_av + np.min(force_diff, axis=0),
            force_av + np.max(force_diff, axis=0),
            color=colours[idx],
            alpha=0.3,
            edgecolor="none",
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="p", direction="x")
    t, force = t[((t > 8) & (t < 12))], force[((t > 8) & (t < 12))]
    t = t % 1
    f = interp1d(t, force, fill_value="extrapolate")
    force = f(t_sample)

    ax.plot(
        t_sample,
        force,
        color=colours[idx + 1],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
    )

    save_path = f"figures/thrust.png"
    # ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def plot_power():
    lams = [16, 32, 64, 128]
    cases = [f"0.001/{lam}" for lam in lams]

    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle C_P \rangle $")
    ax.set_xlabel(r"$t/T$")

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        t_new = t % 1
        f = interp1d(t_new, force, fill_value="extrapolate")
        force = f(t_sample)


        wrap_indices = np.where(np.diff(t_new) < 0)[0] + 1
        wrap_indices = np.insert(wrap_indices, 0, 0)  # Include the start index
        wrap_indices = np.append(wrap_indices, len(t_new))  # Include the end index


        force_bins = [force[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]
        t_bins = [t_new[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]


        # Calculate the standard deviation of each bin
        force_diff = np.empty((4, t_sample.size))
        for id in range(len(force_bins)):
            f_bins = interp1d(t_bins[id], force_bins[id], fill_value="extrapolate")
            force_bins[id] = f_bins(t_sample)
            force_diff[id] = force_bins[id] - force

        ax.plot(
            t_sample,
            force,
            color=colours[idx],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
        )

        ax.axhline(np.mean(force), color=colours[idx], alpha=0.8, linewidth=0.7)

        ax.fill_between(
            t_sample,
            force + np.min(force_diff, axis=0),
            force + np.max(force_diff, axis=0),
            color=colours[idx],
            alpha=0.3,
            edgecolor="none",
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="cp", direction="")
    t, force = t[((t > 8) & (t < 12))], force[((t > 8) & (t < 12))]
    t = t % 1
    f = interp1d(t, force, fill_value="extrapolate")
    force_av_s = f(t_sample)

    ax.plot(
        t_sample,
        force_av_s,
        color=colours[idx + 1],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
    )


    ax.axhline(np.mean(force_av_s), color=colours[idx+1], alpha=0.8, linewidth=0.7)
    
    print((np.mean(force)-np.mean(force_av_s))/np.mean(force_av_s) * 100)

    save_path = f"figures/power.pdf"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def plot_fft():
    lams = [16, 128]
    cases = [f"0.001/{lam}" for lam in lams]
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"Power")
    ax.set_xlabel(r"$f^*$")
    ax.set_xlim(0.1, 150)

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        dt = 4/len(t)

        freq, Pxx = welch(force, 1/dt, nperseg=len(t//8))
        # Pxx = savgol_filter(Pxx, 4, 1)


        ax.loglog(freq, Pxx, color=colours[idx], label=labels[idx], alpha=0.8, linewidth=0.7)

    # Adding the 'Smooth' curve
    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="cp", direction="")
    t, force = t[((t > 8) & (t < 12))], force[((t > 8) & (t < 12))]
    dt = np.mean(np.diff(t))

    freq, Pxx = welch(force, 1/dt, nperseg=len(t//4))
    # Applay savgiol filter
    # Pxx = savgol_filter(Pxx, 4, 1)
    ax.loglog(freq, Pxx, color=colours[idx + 1], label="Smooth", alpha=0.8, linewidth=0.7)

    save_path = f"figures/fft_power.pdf"
    ax.legend(loc="lower left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def test_E_scaling():
    path = "/research/sharkdata/research_filesystem/thicc-swimmer/128_z_res_test/128/128/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_short = t[t > 5], enst[t > 5]/(64*2)
    ts = t % 1
    path = "data/0.001/128/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_long = t[t > 5], enst[t > 5]/(64*4)
    tl = t % 1
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle E \rangle $")
    ax.set_xlabel(r"$t/T$")
    ax.plot(ts, enst_short, label="Short", ls="none", marker="o", markersize=0.1)
    ax.plot(tl, enst_long, label="Long", ls="none", marker="o", markersize=.1)
    ax.legend()
    plt.savefig(f"figures/E_test.pdf", dpi=300)
    plt.close()
    print((enst_short.max()-enst_long.max())/enst_long.max())


def plot_E_body_ts_3d():
    cmap = sns.color_palette("Reds", as_cmap=True)
    Ls = np.array([4096, 8192])

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.set_xlabel(f"$ t $")
    ax.set_ylabel(r"$ E $")

    t_sample = np.linspace(2.01, 6.99, 300)

    norm = colors.LogNorm(vmin=min(4 / (Ls*2)), vmax=max(4 / (Ls/2)))
    for L in (Ls):
        try:
            f=field_eval_helper(L, 'grid-3-medium', 'E','b')
            ax.plot(
                t_sample,
                f(t_sample),
                color=cmap(norm(4 / L)),
                ls=':',
            )
            f=field_eval_helper(L, '3d-check', 'E','b')
            ax.plot(
                t_sample,
                f(t_sample)/(L/8),
                color=cmap(norm(4 / L)),
                ls='-',
            )
        except FileNotFoundError or ValueError:
            pass

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    cb.set_label(r"$\Delta x$", rotation=0)

    legend_elements = [Line2D([0], [0], ls='-', label='3-D', c='k'),
                       Line2D([0], [0], ls=':', label='2-D', c='k')]
    ax.legend(handles=legend_elements)
    plt.savefig(f"{os.getcwd()}/figures/E.pdf", bbox_inches="tight", dpi=200)


def field_eval_helper(L, crit_str='3d-check', interest='v', direction='x'):
    t, ux, *_ = read_forces(
            f"/ssdfs/users/jmom1n15/thicc-swimmer/two-dim-convergenence-test/{crit_str}/{L}/fort.9",
            interest=interest,
            direction=direction,
        )
    t, ux = t[t > 2], ux[t > 2]
    f = interp1d(t, ux, fill_value="extrapolate")

    return f


def plot_E():
    lams = [16, 32, 64, 128]#
    order = [0, 3, 4, 1]

    cases = [f"0.001/{lam}" for lam in lams]

    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle E \rangle $")
    ax.set_xlabel(r"$t/T$")

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, enst = read_forces(path, interest="E", direction="")
        t, enst = t[((t > 8.1) & (t < 12))], enst[((t > 8) & (t < 12))]

        # t_new = t % 1
        # f = interp1d(t_new, enst, fill_value="extrapolate")
        # enst = f(t_sample)


        # wrap_indices = np.where(np.diff(t_new) < 0)[0] + 1
        # wrap_indices = np.insert(wrap_indices, 0, 0)  # Include the start index
        # wrap_indices = np.append(wrap_indices, len(t_new))  # Include the end index


        # enst_bins = [enst[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]
        # t_bins = [t_new[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]


        # # Calculate the standard deviation of each bin
        # enst_diff = np.empty((4, t_sample.size))
        # for id in range(len(enst_bins)):
        #     f_bins = interp1d(t_bins[id], enst_bins[id], fill_value="extrapolate")
        #     enst_bins[id] = f_bins(t_sample)
        #     enst_diff[id] = enst_bins[id] - enst
        
        enst = enst/span(1/lams[idx])

        ax.plot(
            t,
            enst,
            color=colours[order[idx]],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst = t[((t > 8) & (t < 12))], enst[((t > 8) & (t < 12))]
    # t = t % 1
    # f = interp1d(t, enst, fill_value="extrapolate")
    # enst = f(t_sample)

    ax.plot(
        t,
        enst,
        color=colours[2],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
    )

    # ax.set_yscale("log")

    save_path = f"figures/E.pdf"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_E_fft():
    lams = [16, 128]
    cases = [f"0.001/{lam}" for lam in lams]
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"PSD(E)")
    ax.set_xlabel(r"$f^*$")
    ax.set_xlim(0.1, 150)

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, enst = read_forces(path, interest="tke", direction="")
        dt = 4/len(t)

        enst = enst/span(1/lams[idx])
        # enst = enst - np.mean(enst)

        freq, Pxx = welch(enst, 1/dt, nperseg=len(t//1))
        # Pxx = savgol_filter(Pxx, 4, 1)


        ax.loglog(freq, Pxx, color=colours[idx], label=labels[idx], alpha=0.8, linewidth=0.7)

    # Adding the 'Smooth' curve
    path = f"data/test/up/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst = t[((t > 8) & (t < 12))], enst[((t > 8) & (t < 12))]
    dt = np.mean(np.diff(t))

    # enst = enst - np.mean(enst)

    freq, Pxx = welch(enst/4, 1/dt, nperseg=len(t//1))
    # Applay savgiol filter
    # Pxx = savgol_filter(Pxx, 4, 1)
    ax.loglog(freq, Pxx, color=colours[idx + 1], label="Smooth", alpha=0.8, linewidth=0.7)

    save_path = f"figures/fft_E.pdf"
    ax.legend(loc="lower left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def SA_enstrophy_scaling(span):
        return (
            1 / 0.1             # A
            / (1)     # L
            / (span * 4096)  # span
        )

def span(lam):
    if lam == 1/16:
        span = 256
    elif lam == 1/32:
        span = 192
    else:
        span = 64
    
    return span*4


if __name__ == "__main__":
    # test_E_scaling()
    
    plot_E()
    # plot_E_fft()

    # plot_E_body_ts_3d()