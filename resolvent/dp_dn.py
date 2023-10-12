import os
import numpy as np
from pydmd import FbDMD

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

case = "test/down"
# cases=["test/up", "0.001/64", "0.001/128"]
# cases=["test/up"]

snapshot = np.load(f"data/{case}/data/uvp.npy")
nx, ny, nt = np.load(f"data/{case}/data/body_nxyt.npy")
pxs = np.linspace(0, 1, nx)
pys = np.linspace(-0.25, 0.25, ny)
dy = 0.5/ny

snapshot.resize(3, nx, ny, nt)
p = snapshot[2, :, :, :]

ts = np.linspace(0, 4, nt)

dp_dx = np.gradient(p, pxs, axis=0)
dp_dy = np.gradient(p, pys, axis=1)

# Find the values of dp/dx at the position of the body (y)
for idt, t in tqdm(enumerate(ts), total=len(ts)):
    y = np.array([naca_warp(xp) for xp in pxs]) + fwarp(t, pxs)
    ceil_index = np.argmin(np.abs(pys - y))
    floor_index = ceil_index + 1
    
    # Calculate the shifts for each x-coordinate
    real_shift = y / dy
    shifts = np.round(real_shift).astype(int)
    shift_up = np.ceil(y / dy).astype(int)
    shift_down = np.floor(y / dy).astype(int)
    
    for i in range(nx):
        shift = shifts[i]
        ushift = shift_up[i]
        dshift = shift_down[i]
        alpha = shift-real_shift[i]
        unwarped[i, :, idt] = apply_shift_interpolation(ushift, dshift, alpha, self.body[d, i, :, idt])


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
    return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


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
    
    return np.array(x), np.array(dpdn)
