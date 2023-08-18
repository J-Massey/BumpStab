import numpy as np
from plot_field import plot_field
import os
import time


class SaveDMD:
    def __init__(self, path, dom, T) -> None:
        self.path = path
        self.dom = dom
        self.T = T
        self._load_modes(dom)
        self.fluc1 = self.flucs[:, :-1]
        self.fluc2 = self.flucs[:, 1:]

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

    def fbDMD(self, r=None):
        dt = self.T / self.nt
        if r is None:
            r = self.nt // self.T

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
        Lambda = np.log(rho) / dt

        # Find the DMD modes
        V_r = np.dot(np.dot(self.fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

        print(f"fbDMD calculated in {time.time() - t0:.2f} seconds.")

        return Lambda, V_r

    def save_fbDMD(self, r=None):
        Lambda, V_r = self.fbDMD(r)
        np.save(f"{self.path}/{self.dom}_Lambda.npy", Lambda)
        np.save(f"{self.path}/{self.dom}_V_r.npy", V_r)


# Sample usage
if __name__ == "__main__":
    case = "0"
    doms = ["body", "wake"]
    for dom in doms:
        resolvent = SaveDMD(f"{os.getcwd()}/data/{case}/data", dom, 4)
        resolvent.save_fbDMD()
