import numpy as np
from tqdm import tqdm
from scipy.linalg import cholesky, svd, inv
from scipy.signal import find_peaks
import sys


class ResolventAnalysis:
    def __init__(self, path, dom, omega_span) -> None:
        self.path = path
        self.dom = dom
        self._load(dom)
        self.omega_span = omega_span
        self.gain_cache = None

    def _load(self, dom):
        self.Lambda = np.load(f"{self.path}/{dom}_Lambda.npy", mmap_mode="r")
        self.Lambda = self.Lambda # [:36]
        print(f"Lambda shape: {self.Lambda.shape}")
        self.V_r = np.load(f"{self.path}/{dom}_V_r.npy", mmap_mode="r")
        self.V_r = self.V_r # [:, :36]
        print(f"V_r shape: {self.V_r.shape}")

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
        peak_omegas = []
        for i in range(5):
            peak_indices, _ = find_peaks(self.gain[:,i])
            peak_omegas.append(self.omega_span[peak_indices])
        return peak_omegas
    
    def save_omega_peaks(self):
        peak_omegas = self.omega_peaks
        np.savez(f"{self.path}/{self.dom}_peak_omegas.npz", *peak_omegas)


# Sample usage
if __name__ == "__main__":
    import os
    case = 0
    ra = ResolventAnalysis(f"{os.getcwd()}/data/stationary", "fb", omega_span=np.logspace(np.log10(0.025*2*np.pi), np.log10(200*2*np.pi), 500))
    ra.save_gain()
    ra.save_omega_peaks()