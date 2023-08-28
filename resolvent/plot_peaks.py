import numpy as np
from scipy.linalg import cholesky, svd, inv

from plot_field import plot_field
import matplotlib.pyplot as plt
import scienceplots
import sys


class PlotPeaks:
    def __init__(self, path, dom):
        self.path = path
        self.dom = dom
        self._load()
    
    def _load(self):
        self.Lambda = np.load(f"{self.path}/{self.dom}_Lambda.npy")
        self.V_r = np.load(f"{self.path}/{self.dom}_V_r.npy")
        self.nx, self.ny, self.nt = np.load(f"{self.path}/{self.dom}_nxyt.npy")
        print("----- Plotting modes -----")
        self.peak_omegas = np.load(f"{self.path}/{self.dom}_peak_omegas.npy")
        print(f"Peak omegas: {self.peak_omegas/(2*np.pi)}")
    
    @property
    def F_tilde(self):
            # Find the hermatian adjoint of the eigenvectors
            V_r_star_Q = self.V_r.conj().T
            V_r_star_Q_V_r = np.dot(V_r_star_Q, self.V_r)
            # Cholesky factorization
            F_tilde = cholesky(V_r_star_Q_V_r)
            return F_tilde

    def plot_forcing(self, case):

        for omega in self.peak_omegas:
            Psi, Sigma, Phi = svd(self.F_tilde@inv((-1j*omega)*np.eye(self.Lambda.shape[0])-np.diag(self.Lambda))@inv(self.F_tilde))
            for i in range(len(Sigma)):
                Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
                Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
                Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

            forcing = (self.V_r @ inv(self.F_tilde)@Psi).reshape(3, self.nx, self.ny, len(Sigma))

            field = forcing[1, :, :, 0].real
            pxs = np.linspace(0, 1, self.nx)
            pys = np.linspace(-0.25, 0.25, self.ny)
            try:
                plot_field(field.T, pxs, pys, f"figures/{case}-modes/{self.dom}_forcing_{omega/(2*np.pi):.2f}.png", _cmap="seismic")
            except ValueError:
                print(f"ValueError, {omega/(2*np.pi):.2f} dodgy")
            
    def plot_response(self, case):
        for omega in self.peak_omegas:
            Psi, Sigma, Phi = svd(self.F_tilde@inv((-1j*omega)*np.eye(self.Lambda.shape[0])-np.diag(self.Lambda))@inv(self.F_tilde))
            for i in range(len(Sigma)):
                Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
                Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
                Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

            response = (self.V_r @ inv(self.F_tilde)@Phi).reshape(3, self.nx, self.ny, len(Sigma))

            field = response[1, :, :, 0].real
            pxs = np.linspace(0, 1, self.nx)
            pys = np.linspace(-0.25, 0.25, self.ny)
            try:
                plot_field(field.T, pxs, pys, f"figures/{case}-modes/{self.dom}_response_{omega/(2*np.pi):.2f}.png", _cmap="seismic")
            except ValueError:
                print(f"ValueError, {omega/(2*np.pi):.2f} dodgy")


# Sample usage
if __name__ == "__main__":
    import os
    case = sys.argv[1]
    os.system(f"mkdir -p figures/{case}-modes")
    doms = ["body", "wake"]
    for dom in doms:
        pp = PlotPeaks(f"{os.getcwd()}/data/{case}/data", dom)
        pp.plot_forcing(case)
        pp.plot_response(case)
