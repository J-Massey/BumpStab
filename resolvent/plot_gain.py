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
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_1$")
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
    path = f"data/0.001/0/unmasked/fb_gain.npy"
    gain = np.load(path)
    # ax.set_xlim(0.1, 150)
    lss = ["-", "-.", "--", ":"]
    for i in range(3):
        ax.loglog(
            omega_span / (2*np.pi),
            np.sqrt(gain[:, i]),
            color=sns.color_palette("colorblind", 7)[2],
            # label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
            linestyle=lss[i],
        )
    # plot max value cross at each peak
    maxs = find_peaks(np.sqrt(gain[:, 0]), prominence=0.1)[0]
    plt.savefig(f"figures/body_gain.pdf")
    plt.savefig(f"figures/body_gain.png", dpi=400)
    plt.close()


# Sample usage
if __name__ == "__main__":
    lss = ["-", "-.", "--"]
    omega_span = np.logspace(np.log10(0.25*2*np.pi), np.log10(200*2*np.pi), 500)
    plot_gain(omega_span)
    
