import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import os
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')


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


def fwarp(t, x):
    xp = min(max(x, 0.0), 1.0)
    return -0.5*(0.28 * xp**2 - 0.13 * xp + 0.05) * np.sin(2*np.pi*(t - (1.42* xp)))


def normal_to_surface(x: np.ndarray, t):
    y = np.array([naca_warp(xp) for xp in x]) - np.array([fwarp(t, xp) for xp in x])
    y = y*1  # Scale the y-coordinate to get away from the body 

    df_dx = np.gradient(y, x, edge_order=2)
    df_dy = -1

    # Calculate the normal vector to the surface
    mag = np.sqrt(df_dx**2 + df_dy**2)
    nx = -df_dx/mag
    ny = -df_dy/mag
    return nx, ny


def tangent_to_surface(x: np.ndarray, t):
    # Evaluate the surface function y(x, t)
    y = np.array([naca_warp(xp) for xp in x]) - np.array([fwarp(t, xp) for xp in x])

    # Calculate the gradient dy/dx
    df_dx = np.gradient(y, x, edge_order=2)

    # The tangent vector components are proportional to (1, dy/dx)
    tangent_x = 1 / np.sqrt(1 + df_dx**2)
    tangent_y = df_dx / np.sqrt(1 + df_dx**2)

    return tangent_x, tangent_y


def point_at_distance(x1, y1, nx, ny, s=0.1):
    """
    Calculates a point that is 's' units away from the point (x1, x2)
    in the direction given by the vector (nx, ny).
    
    :param x1: x-coordinate of the initial point
    :param y1: y-coordinate of the initial point
    :param nx: x-component of the direction vector
    :param ny: y-component of the direction vector
    :param s: distance to move from the initial point
    :return: Tuple representing the new point coordinates
    """
    # Initial point and direction vector
    p = np.array([x1, y1])
    d = np.array([nx, ny])

    # Normalize the direction vector
    d_norm = d / np.linalg.norm(d)

    # Calculate the new point
    p_new = p + s * d_norm
    return p_new


def delta(profile, normal_dis):
    du_dn = np.gradient(profile, normal_dis, edge_order=2, axis=1)
    d2u_dn2 = np.gradient(du_dn, normal_dis, edge_order=2, axis=1)
    d2u_dn2 = (d2u_dn2/np.ptp(d2u_dn2, axis=1)[:, None])
    d2u_dn2 = gaussian_filter1d(d2u_dn2, sigma=1, axis=1)
    delta = np.empty(d2u_dn2.shape[0])
    for idxp in range(d2u_dn2.shape[0]):
        mask = abs(d2u_dn2[idxp]) > 0.01
        delta[idxp] = normal_dis[mask][-1]
    return delta


def tangental_profiles(case, prof_dist=0.05, num_points=4096):
    if os.path.isfile(f"data/0.001/{case}/data/s_profile.np"):
        s_profile = np.load(f"data/0.001/{case}/data/s_profile.npy")
        ts = np.arange(0, s_profile.shape[0], 1)
        pxs = np.linspace(0, 1, s_profile.shape[1])
        return ts, pxs, s_profile
    else:
        bod = np.load(f"data/0.001/{case}/data/uvp.npy")
        pxs = np.linspace(-0.35, 2, bod.shape[1])
        bod_mask = np.where((pxs > 0) & (pxs < 1))
        pys = np.linspace(-0.35, 0.35, bod.shape[2])

        ts = np.arange(0, bod.shape[-1], 10)
        s_profile = np.empty((ts.size, pxs.size, num_points))
        for idt, t_idx in tqdm(enumerate(ts), total=ts.size):
            t = (t_idx/200)%1
            # Interpolation function for the body
            f_u = RegularGridInterpolator((pxs, pys), bod[0, :, :, t_idx].astype(np.float64), bounds_error=False, fill_value=1)
            f_v = RegularGridInterpolator((pxs, pys), bod[1, :, :, t_idx].astype(np.float64), bounds_error=False, fill_value=1)
            y_surface = np.array([naca_warp(xp) for xp in pxs]) - np.array([fwarp(t, xp) for xp in pxs])

            nx, ny = normal_to_surface(pxs, t)
            sx, sy = tangent_to_surface(pxs, t)

            for pidx in range(pxs.size):
                x1, y1 = pxs[pidx], y_surface[pidx]
                x2, y2 = point_at_distance(x1, y1, nx[pidx], ny[pidx], s=prof_dist)
                line_points = np.linspace(np.array([x1, y1]), np.array([x2, y2]), num_points)
                u_profile = f_u(line_points)
                v_profile = f_v(line_points)    
                s_profile[idt, pidx] = u_profile*sx[pidx] + v_profile*sy[pidx]
        
        s_profile = s_profile[:, bod_mask[0], :]
        np.save(f"data/0.001/{case}/data/s_profile.npy", s_profile)
        return ts, pxs, s_profile


def plot_profiles(case):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
    ax.set_xlabel(r"$x_{n=0}+u_s/10$")
    ax.set_ylabel(r"$n$")

    prof_dist = 0.05
    num_points = int(prof_dist*4096)
    ts, pxs, s_profile = tangental_profiles(case, prof_dist=prof_dist, num_points=num_points)

    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, prof_dist])


    normal_dis = np.linspace(0, prof_dist, num_points)
    x_samples = np.array([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_profs = x_samples.size
    closest_index = [np.argmin(abs(pxs-x)) for x in x_samples]

    for idt, t_idx in tqdm(enumerate(ts), total=ts.size):
        for idxp in range(n_profs):
            profile  = s_profile[idt, closest_index[idxp]]-s_profile[idt, closest_index[idxp], 0]  # Subtract the body velocity
            ax.plot(profile/10+x_samples[idxp], normal_dis, color='grey', linewidth=0.1, alpha=0.02)
            delt = delta(s_profile[idt], normal_dis)
            ax.plot(profile/10+x_samples[idxp], delt[closest_index[idxp]], color='green', marker='o', markersize=0.5, alpha=0.3, markerfacecolor='none')
    
    # Find the time avg
    stationary = s_profile - s_profile[:, :, 0][:, :, None]
    avg_profile = np.mean(stationary, axis=0)

    for idxp in range(n_profs):
        ax.plot(avg_profile[closest_index[idxp]]/10 + x_samples[idxp], normal_dis, color='red', linewidth=0.5, alpha=0.6, ls='-.')
    

    plt.savefig(f"figures/profiles/body_profiles.pdf")
    plt.savefig(f"figures/profiles/body_profiles.png", dpi=800)


if __name__ == "__main__":
    cases = [0, 16, 32, 128]
    case = cases[0]
    ts, pxs, s_profile = tangental_profiles(case)
    plot_profiles(case)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
    ax.set_xlabel(r" $x_{n=0}+$ Locally scaled $ \partial^2 u_s/\partial n^2$")
    ax.set_ylabel(r"$n$")

    prof_dist = 0.05
    num_points = int(prof_dist*4096)

    ts = np.arange(0, bod.shape[-1], 1)
    normal_dis = np.linspace(0, prof_dist, num_points)
    x_samples = np.array([0.001, 0.002, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    x_samples = np.linspace(0, 1, 10)
    
    ax.set_xlim([x_samples[0]-0.1, x_samples[-1]+.2])
    ax.set_ylim([0, prof_dist])

    n_profs = x_samples.size
    closest_index = [np.argmin(abs(pxs-x)) for x in x_samples]
    avg_profile = np.load(f"data/0.001/{case}/data/avg_profile.npy")
    
    # Find du/dn
    delt = delta(avg_profile, normal_dis)
    delt = delt[bod_mask]
    ax.plot(pxs[bod_mask], delt, color='black', linewidth=0.5, alpha=0.6, ls='-.')
    
    plt.savefig(f"figures/profiles/body_test.pdf")
    plt.savefig(f"figures/profiles/body_test.png", dpi=800)
