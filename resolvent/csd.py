import os
from tqdm import tqdm
import sys
import numpy as np

from matplotlib import colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter

from scipy.interpolate import interp1d
from scipy.signal import welch, savgol_filter
from scipy.interpolate import CubicSpline


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")


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
    return t, u


def fnorm(case):
    if case == "0.001/16" or case == "0.001/32":
        span = 128
    else:
        span = 64
    normalise = 0.1 * 4096 * span * 4 / 2
    return normalise


def load_phase_avg_cp(cases):
    if os.path.isfile("data/spressure.npy"):
        body = np.load("data/spressure.npy")
    else:
        t, pxs, body = [], [], []
        for case in tqdm(cases):
            cp = np.genfromtxt(f"data/{case}/spressure/fort.1")  # Load the 1D array
            start = np.where(cp[0] != 0)[0][0]
            bodtop = cp[:, start : start + 1024] / fnorm(case)

            cp = np.genfromtxt(f"data/{case}/spressure/fort.2")  # Load the 1D array
            start = np.where(cp[0] != 0)[0][0]
            bodbot = cp[:, start : start + 1024] / fnorm(case)

            tr, _ = read_forces(f"data/{case}/spressure/fort.9", "cp", "")
            ts = np.linspace(0.001, 0.999, 10000)

            # Sort and average duplicates
            sorted_indices_top = np.argsort(tr % 1)
            sorted_indices_bot = np.argsort((tr+0.5) % 1)
            unique_x_top, unique_indices_top = np.unique((tr % 1)[sorted_indices_top], return_index=True)
            unique_x_bot, unique_indices_bot = np.unique(((tr+0.5) % 1)[sorted_indices_bot], return_index=True)

            avg_y_top = np.zeros((unique_x_top.shape[0], bodtop.shape[1]))
            avg_y_bot = np.zeros((unique_x_bot.shape[0], bodbot.shape[1]))

            for col in tqdm(range(1024)):
                sorted_col_top = bodtop[:, col][sorted_indices_top]
                sorted_col_bot = bodbot[:, col][sorted_indices_bot]

                for i, idx in enumerate(unique_indices_top):
                    avg_y_top[i, col] = np.mean(sorted_col_top[sorted_indices_top == idx])
                for i, idx in enumerate(unique_indices_bot):
                    avg_y_bot[i, col] = np.mean(sorted_col_bot[sorted_indices_bot == idx])
            
            unique_y = (avg_y_top + avg_y_bot)/2

            # Create spline
            cs = CubicSpline(unique_x_top, unique_y)
            y_spline = cs(ts)
            body.append(y_spline)
        np.save("data/spressure.npy", body)
    return np.linspace(0.001, 0.999, 10000), np.linspace(0, 1, 1024), body