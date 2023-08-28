import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

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
    lams = [128, 16, 64]
    cases = [f"0.001/{lam}" for lam in lams]
    # cases.append("test/up")
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)
    linestyles = ["-", "-."]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_i$")
    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="p", direction="x")
        print(force.mean())

        ax.plot(
            t,
            force,
            color=colours[idx],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
            # linestyle=linestyles[idd],
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="p", direction="x")
    t, force = t[((t>8)&(t<12))], force[((t>8)&(t<12))]
    print(force.mean())

    ax.plot(
        t,
        force,
        color=colours[idx+1],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
        # linestyle=linestyles[idd],
    )
    
    save_path = f"figures/thrust.pdf"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def plot_power():
    lams = [128, 16, 64]
    cases = [f"0.001/{lam}" for lam in lams]
    # cases.append("test/up")
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_i$")
    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        print(force.mean())

        ax.plot(
            t,
            force,
            color=colours[idx],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
            # linestyle=linestyles[idd],
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="cp", direction="")
    t, force = t[((t>8)&(t<12))], force[((t>8)&(t<12))]
    print(force.mean())

    ax.plot(
        t,
        force,
        color=colours[idx+1],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
        # linestyle=linestyles[idd],
    )
    
    save_path = f"figures/power.pdf"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()

if __name__ == "__main__":
    # plot_thrust()
    plot_power()