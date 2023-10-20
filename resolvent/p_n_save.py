import numpy as np
from tqdm import tqdm
from scipy.stats import norm


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


def gaussian_kernel(y, ys, sigma):
    return norm.pdf(y, loc=ys, scale=sigma)


def normal_pressure(nt, p, pxs, pys, offset=4, sigma=1):
    ts = np.arange(0, 0.005*nt, 0.005)
    nx, ny, nt = p.shape

    p_ny = np.zeros((len(ts), nx))

    for tidx, t in tqdm(enumerate(ts), total=len(ts)):
        y = fwarp(t, pxs) + np.array([naca_warp(xp) for xp in pxs])

        p_ny[tidx] = np.zeros(nx)
        for idx in range(nx):
            y_center = y[idx]
            indices = np.arange(max(0, idx), min(nx, idx + 3))  # Taking a radius of 2 grid points
            weights = gaussian_kernel(y_center, pys[indices], sigma)
            weights /= np.sum(weights)  # Normalize weights

            p_ny[tidx, idx] = np.dot(weights, p[idx, indices, tidx])

        p_ny[tidx] = -p_ny[tidx] * velocity(t, pxs)
        
    return ts, p_ny


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

if __name__ == "__main__":
    lams = [1e9, 1/64, 1/128]
    labs = [f"$\lambda = 1/{int(1/lam)}$" for lam in lams]
    cases = ["test/span64", "0.001/64", "0.001/128", "0.001/16", "0.001/32"]
    offsets = [0, 2, 4, 6, 8, 10, 14]

    for idx, case in enumerate(cases):
            nt, p, pxs, pys = init(case)
            ts, p_n = normal_pressure(nt, p, pxs, pys, offset=2)
            np.save(f"data/{case}/data/p_n.npy", p_n)







