import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns
from scipy.signal import find_peaks

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

def plot_gain(omega_span):
    ax = plot_ax()
    path = f"data/0.001/0/unmasked/sp_gain.npy"
    gain = np.load(path)
    # ax.set_xlim(0.1, 150)
    ax.loglog(
        omega_span / (2*np.pi),
        np.sqrt(gain[:, 0]),
        color=sns.color_palette("colorblind", 7)[2],
        # label=labels[idx],
        alpha=0.8,
        linewidth=0.7,
        linestyle="-",
    )
    # plot max value cross at each peak
    maxs = find_peaks(np.sqrt(gain[:, 0]), prominence=0.1)[0]

    # for max in maxs:
    #     print(omega_span[max] / (2*np.pi))
    #     ax.plot(
    #         omega_span[max] / (np.pi),
    #         np.sqrt(gain[max, 0]),
    #         color=colours[idx+2],
    #         alpha=0.8,
    #         marker="x",
    #         markersize=3,
    #         linewidth=0.7,
    #     )

    save_path = f"figures/body_gain.pdf"
    # ax.legend(loc="upper right")
    save_fig(save_path)
    plt.close()


# Sample usage
if __name__ == "__main__":
    lss = ["-", "-.", "--"]
    omega_span = np.logspace(np.log10(0.5*np.pi), np.log10(300*np.pi), 500)
    plot_gain(omega_span)
    
