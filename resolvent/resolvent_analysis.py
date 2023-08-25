import numpy as np
from tqdm import tqdm
from scipy.linalg import cholesky, svd, inv
from scipy.signal import find_peaks
import sys


class ResolventAnalysis:
    def __init__(self, path, dom, omega_span=np.linspace(0.1, 1000, 2000)) -> None:
        self.path = path
        self.dom = dom
        self._load(dom)
        self.omega_span = omega_span
        self.gain_cache = None

    def _load(self, dom):
        self.Lambda = np.load(f"{self.path}/{dom}_Lambda.npy")
        self.V_r = np.load(f"{self.path}/{dom}_V_r.npy")

    @property
    def F_tilde(self):
        # Find the hermatian adjoint of the eigenvectors
        V_r_star_Q = self.V_r.conj().T
        V_r_star_Q_V_r = np.dot(V_r_star_Q, self.V_r)
        # Cholesky factorization
        F_tilde = cholesky(V_r_star_Q_V_r)
        return F_tilde

    @property
    def gain(self):
        if self.gain_cache is None:
            self.gain_cache = np.empty((self.omega_span.size, self.Lambda.size))
            for idx, omega in tqdm(enumerate(self.omega_span), total=len(self.omega_span)):
                R = svd(
                    self.F_tilde
                    @ inv(
                        (-1j * omega) * np.eye(self.Lambda.shape[0]) - np.diag(self.Lambda)
                    )
                    @ inv(self.F_tilde),
                    compute_uv=False,
                )
                self.gain_cache[idx] = R**2
        return self.gain_cache
    
    def save_gain(self):
        np.save(f"{self.path}/{self.dom}_gain.npy", self.gain)
    
    @property
    def omega_peaks(self):
        # Find peaks in the gain data
        peak_indices, _ = find_peaks(self.gain[:,0])
        # Extract the omega values corresponding to these peaks
        peak_omegas = self.omega_span[peak_indices]
        return peak_omegas
    
    def save_omega_peaks(self):
        peak_omegas = self.omega_peaks
        np.save(f"{self.path}/{self.dom}_peak_omegas.npy", peak_omegas)


# Sample usage
if __name__ == "__main__":
    import os
    case = sys.argv[1]
    doms = ["body", "wake"]
    for dom in doms:
        ra = ResolventAnalysis(f"{os.getcwd()}/data/{case}/data", dom, omega_span=np.logspace(0.1, 150*2*np.pi, 1000))
        ra.save_gain()
        ra.save_omega_peaks()