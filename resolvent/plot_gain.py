import numpy as np

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


def plot_gain(path, save_path, omega_span):
    gain = np.load(path)
    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_xlabel(r"$f^*$")
    ax.set_ylabel(r"$\sigma_i$")
    # ax.set_xlim(0, 10)
    for i in range(3):
        ax.loglog(omega_span/(2*np.pi), np.sqrt(gain[:, i]))
    plt.savefig(save_path, dpi=700)
    plt.close()


# Sample usage
if __name__ == "__main__":
    case = SystemError.argv[1]
    doms = ["body", "wake"]
    for dom in doms:
        path = f"data/{case}/data/{dom}_gain.npy"
        save_path = f"figures/{case}_gain_{dom}.png"
        plot_gain(path, save_path, omega_span=np.linspace(0.1, 100*2*np.pi, 2000))
