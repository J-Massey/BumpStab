import os
import numpy as np
from pydmd import FbDMD
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')


def init(case):
    snapshot = np.load(f"data/{case}/data/uvp.npy")
    _, nx, ny, nt = snapshot.shape
    p = snapshot[2, :, :, :]
    pxs  = np.linspace(-0.35, 2, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    mask = (pxs > 0) & (pxs < 1)
    p = p[mask, :, :]
    nx, ny, nt = p.shape

    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    return nt,p,pxs,pys


def normal_pressure(nt, p, pxs, pys):
    ts = np.arange(0, 0.005*nt, 0.005)
    nx, ny, nt = p.shape

    # dp_dx = np.gradient(p, pxs, axis=0)
    # dp_dy = np.gradient(p, pys, axis=1)

    # Find the values of dp/dx at the position of the body (y)
    p_ny = np.zeros((len(ts), nx))
    for tidx, t in tqdm(enumerate(ts), total=len(ts)):
        y = fwarp(t, pxs) + np.array([naca_warp(xp) for xp in pxs])
        y = y*1.00005  # Scale the y-coordinate to get away from the body

        ceil_index = np.array([np.where(pys > y[xidix])[0][0] for xidix in range(nx)])
        floor_index = ceil_index - 1
        alpha = (y - pys[floor_index]) / (pys[ceil_index] - pys[floor_index])


        p = np.array([(1-alpha[idx])*p[idx, floor_index[idx], tidx] + alpha[idx]*p[idx, ceil_index[idx], tidx] for idx in range(nx)])
        # dpdy = np.array([(1-alpha[idx])*dp_dy[idx, floor_index[idx], tidx] + alpha[idx]*dp_dy[idx, ceil_index[idx], tidx] for idx in range(nx)])

        normx, normy = normal_to_surface(pxs, t)
        p_ny[tidx] = (p*normy)*velocity(t, pxs)
        
    return ts,p_ny


def phase_average(ts, p_ny):
    # split into 4 equal parts
    p_ny1 = p_ny[:int(len(ts)/4)]
    p_ny2 = p_ny[int(len(ts)/4):int(len(ts)/2)]
    p_ny3 = p_ny[int(len(ts)/2):int(3*len(ts)/4)]
    p_ny4 = p_ny[int(3*len(ts)/4):]
    return (p_ny1 + p_ny2 + p_ny3 + p_ny4)/4


def naca_warp(x):
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274

    xp = min(max(x, 0.0), 1.0)
    
    return (a * xp + b * xp**2 + c * xp**3 + d * xp**4 + e * xp**5 + 
            f * xp**6 + g * xp**7 + h * xp**8 + i * xp**9 + j * xp**10)


def fwarp(t: float, pxs: np.ndarray):
    return 0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def velocity(t, pxs):
    return np.pi * (0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.cos(2 * np.pi * (t - 1.42 * pxs))


def normal_to_surface(x: np.ndarray, t):
    y = np.array([naca_warp(xp) for xp in x]) + fwarp(t, x)

    df_dx = np.gradient(y, x, edge_order=2)
    df_dy = 1

    # Calculate the normal vector to the surface
    mag = np.sqrt(df_dx**2 + df_dy**2)
    nx = -df_dx/mag
    ny = df_dy/mag
    return nx, ny


def plot_pa_recovery(phase_average, colours, order, cases):
    fig, ax = plt.subplots(figsize=(3, 3))
    divider = make_axes_locatable(ax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varphi$")

    ax.set_xlim(0, 1)
    # ax.set_ylim(0, 0.1)

    lim = 50  # min(np.max(dpdn), np.abs(np.min(dpdn)))
    levels = np.linspace(-lim, lim, 6)
    for idx, case in enumerate(cases):
        nt, p, pxs, pys = init(case)
        ts, dpdn = normal_pressure(nt, p, pxs, pys)
        np.save(f"data/{case}/data/dpdn.npy", dpdn)

        dpdn = np.load(f"data/{case}/data/dpdn.npy")
        nt = dpdn.shape[0]
        pxs = np.linspace(0, 1, dpdn.shape[1])
        ts = np.arange(0, 0.005*nt, 0.005)

        dpdn_filtered = gaussian_filter1d(dpdn, sigma=2, axis=1)
        dpdn_filtered = gaussian_filter1d(dpdn_filtered, sigma=2, axis=0)
        pa = phase_average(ts, dpdn_filtered)

        phi = ts[:nt//4]
        phi_recovery_mask = (phi > 0.2) & (phi < 0.4)
        cs = ax.contour(
        pxs,
        phi[phi_recovery_mask],
        pa[phi_recovery_mask, :],
        levels=levels,
        vmin=-lim,
        vmax=lim,
        colors=[colours[order[idx]]],
        linewidths=0.5,
        )
        
        # label only +ve contours
        # positive_levels = [level for level in cs.levels if level > 0]
        ax.clabel(cs, levels=levels, inline=True, fontsize=6, fmt="%.1f", colors='grey')

    # ax.set_title(r"$ \vec{v}\cdot\frac{\partial p}{\partial n}\Big |_{n=0}$", rotation=0)
    plt.savefig(f"figures/phase-info/surface/phase_average_recovery.pdf", dpi=450, transparent=True)
    plt.close()

def mid_body_recovery(phase_average, lams, cases):
    for idx, case in enumerate(cases):
        fig, ax = plt.subplots(figsize=(3, 3))
        divider = make_axes_locatable(ax)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\varphi$")

        lim = 50  # min(np.max(dpdn), np.abs(np.min(dpdn)))
        levels = np.linspace(-lim, lim, 22)

        dpdn = np.load(f"data/{case}/data/dpdn.npy")
        nt = dpdn.shape[0]
        pxs = np.linspace(0, 1, dpdn.shape[1])
        ts = np.arange(0, 0.005*nt, 0.005)

        dpdn_filtered = gaussian_filter1d(dpdn, sigma=2, axis=1)
        dpdn_filtered = gaussian_filter1d(dpdn_filtered, sigma=2, axis=0)
        dpdn_filtered = dpdn
        pa = phase_average(ts, dpdn_filtered)

        phi = ts[:nt//4]
        phi_recovery_mask = (phi > 0.2) & (phi < 0.4)
        body_recovery_mask = (pxs > 0.4) & (pxs < 0.8)
        cs = ax.contourf(
        pxs[body_recovery_mask],
        phi[phi_recovery_mask],
        pa[phi_recovery_mask, :][:, body_recovery_mask],
        levels=levels,
        vmin=-lim,
        vmax=lim,
        cmap=sns.color_palette("icefire", as_cmap=True),
        extend="both",
        )

        cax = divider.append_axes("right", size="7%", pad=0.2)
        fig.add_axes(cax)
        cb = plt.colorbar(cs, cax=cax, orientation="vertical", ticks=np.linspace(-lim, lim, 5))
                

        # ax.set_title(r"$ \vec{v}\cdot\frac{\partial p}{\partial n}\Big |_{n=0}$", rotation=0)
        plt.savefig(f"figures/phase-info/surface/mid_body_recovery_{int(1/lams[idx])}.pdf", dpi=450, transparent=True)
        plt.close()

if __name__ == "__main__":
    colours = sns.color_palette("colorblind", 7)
    order = [2, 4, 1]
    lams = [1e9, 1/64, 1/128]
    labs = [f"$\lambda = 1/{int(1/lam)}$" for lam in lams]
    cases = ["test/span64", "0.001/64", "0.001/128"]

    plot_pa_recovery(phase_average, colours, order, cases)
    mid_body_recovery(phase_average, lams, cases)

    for idx, case in enumerate(cases):
        fig, ax = plt.subplots(figsize=(3, 3))
        divider = make_axes_locatable(ax)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\varphi$")

        lim = 50  # min(np.max(dpdn), np.abs(np.min(dpdn)))
        levels = np.linspace(-lim, lim, 22)

        dpdn = np.load(f"data/{case}/data/dpdn.npy")
        nt = dpdn.shape[0]
        pxs = np.linspace(0, 1, dpdn.shape[1])
        ts = np.arange(0, 0.005*nt, 0.005)

        dpdn_filtered = gaussian_filter1d(dpdn, sigma=2, axis=1)
        dpdn_filtered = gaussian_filter1d(dpdn_filtered, sigma=2, axis=0)
        dpdn_filtered = dpdn
        pa = phase_average(ts, dpdn_filtered)

        phi = ts[:nt//4]
        phi_recovery_mask = (phi > 0.2) & (phi < 0.4)
        body_recovery_mask = (pxs > 0.7) & (pxs < 1.0)
        cs = ax.contourf(
        pxs[body_recovery_mask],
        phi[phi_recovery_mask],
        pa[phi_recovery_mask, :][:, body_recovery_mask],
        levels=levels,
        vmin=-lim,
        vmax=lim,
        cmap=sns.color_palette("icefire", as_cmap=True),
        extend="both",
        )

        cax = divider.append_axes("right", size="7%", pad=0.2)
        fig.add_axes(cax)
        cb = plt.colorbar(cs, cax=cax, orientation="vertical", ticks=np.linspace(-lim, lim, 5))
                

        # ax.set_title(r"$ \vec{v}\cdot\frac{\partial p}{\partial n}\Big |_{n=0}$", rotation=0)
        plt.savefig(f"figures/phase-info/surface/tail_recovery_{int(1/lams[idx])}.pdf", dpi=450, transparent=True)
        plt.close()
    


