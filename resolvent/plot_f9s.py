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
    ax.legend(loc="upper left")
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
    t, force = read_forces(path, interest="cp", direction="")
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

    save_path = f"figures/power.png"
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
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def plot_E():
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
    t, force = read_forces(path, interest="cp", direction="")
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

    save_path = f"figures/E.png"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()





if __name__ == "__main__":
    # plot_thrust()
    # plot_power()
    plot_fft()
