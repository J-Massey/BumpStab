import numpy as np
import os
import time
import sys


class SaveDMD:
    def __init__(self, path, dom) -> None:
        self.path = path
        self.dom = dom
        self._load_modes(dom)
        self.fluc1 = self.flucs[:, :-1]
        self.fluc2 = self.flucs[:, 1:]
        self.dt = 0.005

    def _load_modes(self, dom):
        self.flucs = np.load(f"{self.path}/{self.dom}_flucs.npy")
        with np.load(f"{self.path}/{dom}_svd.npz") as data:
            self.Ub = data["Ub"]
            self.Sigmab = data["Sigmab"]
            self.VTb = data["VTb"]
            self.Uf = data["Uf"]
            self.Sigmaf = data["Sigmaf"]
            self.VTf = data["VTf"]
        self.nx, self.ny, self.nt = np.load(f"{self.path}/{self.dom}_nxyt.npy")

    def fbDMD(self, r=400):
        print(f"\n----- Calculating fbDMD with r={r} -----")
        t0 = time.time()
        U_r = self.Ub[:, :r]
        Sigmar = np.diag(self.Sigmab[:r])
        VT_r = self.VTb[:r, :]
        Atildeb = np.linalg.solve(Sigmar.T, (U_r.T @ self.fluc1 @ VT_r.T).T).T

        U_r = self.Uf[:, :r]
        S_r = np.diag(self.Sigmaf[:r])
        VT_r = self.VTf[:r, :]
        Atildef = np.linalg.solve(
            S_r.T, (U_r.T @ self.fluc2 @ VT_r.T).T
        ).T  # Step 2 - Find the linear operator using psuedo inverse

        # Find the linear operator
        A_tilde = 1 / 2 * (Atildef + np.linalg.inv(Atildeb))

        rho, W = np.linalg.eig(A_tilde)

        # Find the eigenfunction from spectral expansion
        Lambda = np.log(rho) / self.dt

        # Find the DMD modes
        V_r = np.dot(np.dot(self.fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

        print(f"fbDMD calculated in {time.time() - t0:.2f} seconds.")

        return Lambda, V_r


    def save_fbDMD(self, r=400):
        Lambda, V_r = self.fbDMD(r)
        np.save(f"{self.path}/{self.dom}_Lambda.npy", Lambda)
        np.save(f"{self.path}/{self.dom}_V_r.npy", V_r)


# Sample usage
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from scipy.linalg import svd

    plt.style.use(["science"])
    plt.rcParams["font.size"] = "10.5"
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{txfonts}')


    dom="body"
    case = "test/up"
    resolvent = SaveDMD(f"{os.getcwd()}/data/{case}/data", dom)

    X = resolvent.fluc1
    Y = resolvent.fluc2

    def gaussian_kernel(X, Y, sigma=1.0):
        pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-pairwise_dists / (2 * sigma ** 2))

    
    K_X = gaussian_kernel(X.T, X.T)
    K_Y = gaussian_kernel(Y.T, X.T)


    # SVD of Kernel matrix
    U, S, Vh = svd(K_X)
    U_r = U
    S_r_inv = np.diag(1 / S)

    # Reduced A matrix in feature space
    A_tilde = U_r.T.conj() @ K_Y @ U_r @ S_r_inv

    # Eigen decomposition of A_tilde
    lambda_vec, W = np.linalg.eig(A_tilde)

    # Compute alpha (coefficients for modes in feature space)
    alpha = U_r @ W

    vr = Y @ Vh.T @ S_r_inv @ W


    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/{dom}_nxyt.npy")
    nt = vr.shape[-1]
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)

    vr.resize(3, nx, ny, nt)

    for n in tqdm(range(nt)):
        fig, ax = plt.subplots(figsize=(5, 3))
        qi = vr[2, :, :, n].real.T
        lim = np.std(qi)*4
        levels = np.linspace(-lim, lim, 44)
        _cmap = sns.color_palette("seismic", as_cmap=True)

        cs = ax.contourf(
            pxs,
            pys,
            qi,
            levels=levels,
            vmin=-lim,
            vmax=lim,
            # norm=norm,
            cmap=_cmap,
            extend="both",
            # alpha=0.7,
        )
        ax.set_aspect(1)
        plt.savefig(f"low-mode/dmd-comparison/lando/{n}.png", dpi=300)
        plt.close()