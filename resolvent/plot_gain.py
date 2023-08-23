import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


def plot_ax():
    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_ylim(0, 10)
    return ax

def save_fig(save_path):
    plt.savefig(save_path, dpi=700)
    plt.close()

def load_plot(path, ax, omega_span, colour):
    gain = np.load(path)
    ax.loglog(omega_span/(2*np.pi), np.sqrt(gain[:, 0]), color=colour)


# Sample usage
if __name__ == "__main__":
    colours = sns.color_palette("colorblind")
    cases = ["test", "0.001/128"]
    dom = "body"
    for idx, case in enumerate(cases):
        path = f"data/{case}/data/{dom}_gain.npy"
        save_path = f"figures/{case}_gain_{dom}.png"
        ax = plot_ax()
        load_plot(path, ax, np.linspace(0.1, 100*2*np.pi, 2000), colour=colours[idx])
        save_fig(save_path)
