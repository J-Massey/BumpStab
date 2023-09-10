import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{txfonts}')



def plot_ax():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_i$")
    # ax.set_ylim(0, 10)
    return ax


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


# Sample usage
if __name__ == "__main__":
    omega_span = np.logspace(np.log10(0.1), np.log10(150*2*np.pi), 1000)
    lams = [16, 128]
    cases = [f"0.001/{lam}" for lam in lams]
    cases.append("test/up")
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)
    linestyles = ["-", "-."]
    doms = ["body", "wake"]
    for idd, dom in enumerate(doms):
        ax = plot_ax()
        for idx, case in enumerate(cases):
            path = f"data/{case}/data/{dom}_gain.npy"
            gain = np.load(path)
            ax.set_xlim(0.1, 75)
            ax.loglog(
                omega_span / (2 * np.pi),
                np.sqrt(gain[:, 0]),
                color=colours[idx],
                label=labels[idx],
                alpha=0.8,
                linewidth=0.7,
                linestyle=linestyles[idd],
            )

        save_path = f"figures/{dom}_gain.png"
        ax.legend(loc="upper left")
        save_fig(save_path)
        plt.close()
