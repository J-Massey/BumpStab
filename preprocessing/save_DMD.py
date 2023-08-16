from scipy.linalg import svd, cholesky
import numpy as np
from plot_field import plot_field
import os


class SaveDMD:
    def __init__(self, path, dom) -> None:
        self.path = path
        self.dom = dom
        self._load_modes(dom)
        self.fluc1 = self.flucs[:, :-1]
        self.fluc2 = self.flucs[:, 1:]

    def _load_modes(self, dom):
        self.flucs = np.load(f"{self.path}/{self.dom}_flucs.npy")
        self.Ub, self.Sigmab, self.VTb, self.Uf, self.Sigmaf, self.VTf = np.load(
            f"{self.path}/{dom}_svd.npz", allow_pickle=True
        )
        self.nx, self.ny, self.nt = np.load(f"{self.path}/{self.dom}_nxyt.npy")

    def fbDMD(self, r=None):
        dt = 14 / self.nt
        if r is None:
            r = self.nt - 1
        # Sigma_plot(Sigma)
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

        return Lambda, V_r
    
    def save_fbDMD(self, r=None):
        Lambda, V_r = self.fbDMD(r)
        np.save(f"{self.path}/{self.dom}_Lambda.npy", Lambda)
        np.save(f"{self.path}/{self.dom}_V_r.npy", V_r)
    
def F_tilde(V_r):
    # Find the hermatian adjoint of the eigenvecs
    V_r_star_Q = V_r.conj().T
    V_r_star_Q_V_r = np.dot(V_r_star_Q, V_r)
    # Cholesky factorization
    F_tilde = cholesky(V_r_star_Q_V_r)
    return F_tilde


# Sample usage
if __name__ == "__main__":
    case = "0.001/128"
    doms = ["body", "wake"]
    for dom in doms:
        resolvent = SaveDMD(f"{os.getcwd()}/data/{case}/data", dom)
        resolvent.save_fbDMD()
        resolvent.flucs.shape
        print(resolvent.nx, resolvent.ny, resolvent.nt)
        q = resolvent.flucs.reshape(3, resolvent.nx, resolvent.ny, resolvent.nt)
        pxs = np.linspace(0, 1, resolvent.nx)
        pys = np.linspace(-0.25, 0.25, resolvent.ny)
        plot_field(q[1, :, :, 0].T, pxs, pys, f"figures/test_{dom}.pdf", lim=[-0.5, 0.5])
