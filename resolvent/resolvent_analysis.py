import numpy as np
import tqdm
from scipy.linalg import cholesky


class ResolventAnalysis:
    def __init__(self, path, dom) -> None:
        self.path = path
        self.dom = dom
        self._load(dom)

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

    def calc_gain(self, omega_span=np.linspace(0, 1000, 2000)):
        gain = np.empty((omega_span.size, self.Lambda.size))
        for idx, omega in tqdm(enumerate(omega_span)):
            R = np.linalg.svd(
                self.F_tilde
                @ np.linalg.inv(
                    (-1j * omega) * np.eye(self.Lambda.shape[0]) - np.diag(self.Lambda)
                )
                @ np.linalg.inv(self.F_tilde),
                compute_uv=False,
            )
            gain[idx] = R**2
        return gain
