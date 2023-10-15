import os
import numpy as np
from pydmd import FbDMD
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import sys
import seaborn as sns

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')

colours = sns.color_palette("colorblind", 7)
order = [2, 4, 1]
labs = [r"$\lambda = 1/0$", r"$\lambda = 1/64$", r"$\lambda = 1/128$"]

case = "test/up"
# cases=["test/up", "0.001/64", "0.001/128"]
# cases=["test/up"]

snapshot = np.load(f"data/{case}/data/uvp.npy")
_, nx, ny, nt = snapshot.shape
p = snapshot[0, :, :, :]
pxs  = np.linspace(-0.35, 2, nx)
pys = np.linspace(-0.35, 0.35, ny)
mask = (pxs > 0) & (pxs < 1)
p = p[mask, :, :]
nx, ny, nt = p.shape

pxs = np.linspace(0, 1, nx)
pys = np.linspace(-0.35, 0.35, ny)


def normal_pressure(p, lam):
    dp_dx = np.gradient(p, axis=0)
    dp_dy = np.gradient(p, axis=1)

    # Find the values of dp/dx at the position of the body (y)
    # for idt, t in tqdm(enumerate(ts), total=len(ts)):
    tidx = 0
    t=tidx*0.005
    y = fwarp(t, pxs) + np.array([naca_warp(xp) for xp in pxs])
    y = y*1.001  # Scale the y-coordinate to get away from the body

    ceil_index = np.array([np.where(pys > y[xidix])[0][0] for xidix in range(nx)])
    floor_index = ceil_index - 1
    alpha = (y - pys[floor_index]) / (pys[ceil_index] - pys[floor_index])


    dpdx = np.array([(1-alpha[idx])*dp_dx[idx, floor_index[idx], tidx] + alpha[idx]*dp_dx[idx, ceil_index[idx], tidx] for idx in range(nx)])
    dpdy = np.array([(1-alpha[idx])*dp_dy[idx, floor_index[idx], tidx] + alpha[idx]*dp_dy[idx, ceil_index[idx], tidx] for idx in range(nx)])

    normx, normy = normal_to_surface(pxs, t)
    dpdn = dpdx*normx + dpdy*normy


# Plot dpdx
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(pxs, dpdx, lw=.2, color="red", label=r"$\frac{\partial p}{\partial x}$")
ax.plot(pxs, dpdy, lw=.2, color="blue", label=r"$\frac{\partial p}{\partial y}$")
ax.plot(pxs, dpdn, lw=.2, color="green", label=r"$\frac{\partial p}{\partial n}$")
ax.legend()
plt.savefig("figures/test.pdf", dpi=200)

def roughness(lam):
    return 0.001*np.sin(2*np.pi*lam*pxs)



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


def normal_to_surface(x: np.ndarray, t):
    y = np.array([naca_warp(xp) for xp in x]) + fwarp(t, x)
    y = y*1  # Scale the y-coordinate to get away from the body 

    df_dx = np.gradient(y, x, edge_order=2)
    df_dy = -1

    # Calculate the normal vector to the surface
    mag = np.sqrt(df_dx**2 + df_dy**2)
    nx = -df_dx/mag
    ny = -df_dy/mag
    return nx, ny


def dp_dn(bsnap, fsnap, phi=0):
    x, y = fsnap.X.mean(axis=2), fsnap.Y.mean(axis=2)
    # Move eps away from bbox
    contour = 1.0
    b_cont_locs = bsnap.body_contour_idx(contour)
    # get the normal vectors
    nx, ny = bsnap.norm_vecs(contour)

    # get the top surface
    top = ny > 0
    nx, ny = nx[top], ny[top]

    # Find the velocity gradient normal to the surface
    ddx, ddy = (
        np.gradient(fsnap.p.mean(axis=2), axis=0)[*b_cont_locs],
        np.gradient(fsnap.p.mean(axis=2), axis=1)[*b_cont_locs],
    )
    ddx, ddy = ddx[top], ddy[top]
    dpdn = nx * ddx + ny * ddy

    # sort the order of the points out
    x_re = x[*b_cont_locs][top]
    p = np.argsort(x_re)

    # Apply a savitsky-golay smoothing
    x, dpdn = x_re[p], dpdn[p]
    dpdn = savgol_filter(
        dpdn,
        int(len(dpdn) / 8),
        2,
    )


