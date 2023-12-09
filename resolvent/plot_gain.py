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

def plot_gain():
    fig, ax = plt.subplots(2, figsize=(5.7, 5))
    fig.text(0.5, -0.12, r"$f$", transform=ax[0].transAxes, ha="center")
    ax[1].set_xlabel(r"$f^*$")
    ax[0].set_ylabel(r"$\sigma_1$")
    ax[1].set_ylabel(r"$\sigma_1$")
    path = f"data/0.001/0/unmasked/fb_gain.npy"
    gain = np.load(path)
    omega_span = np.logspace(np.log10(0.25*2*np.pi), np.log10(200*2*np.pi), 500)
    ax[1].loglog(
        omega_span / (2*np.pi),
        np.sqrt(gain[:, 0]),
        color=sns.color_palette("colorblind", 7)[2],
        # label=labels[idx],
        alpha=0.8,
        linewidth=0.7,
        linestyle='-',
    )
    path = f"data/stationary/fb_gain.npy"
    gain = np.load(path)
    omega_span = np.logspace(np.log10(0.025*2*np.pi), np.log10(200*2*np.pi), 500)
    ax[0].loglog(
        omega_span / (2*np.pi),
        np.sqrt(gain[:, 0]),
        color='red',
        # label=labels[idx],
        alpha=0.8,
        linewidth=0.7,
        linestyle='-',
    )

    fig.text(-0.12, 1, r"(a)", transform=ax[0].transAxes, ha="center")
    fig.text(-0.12, 1, r"(b)", transform=ax[1].transAxes, ha="center")
    
    plt.savefig(f"figures/gain.pdf")
    plt.savefig(f"figures/gain.png", dpi=400)
    plt.close()


# Sample usage
if __name__ == "__main__":
    lss = ["-", "-.", "--"]
    plot_gain()
    
