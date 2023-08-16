import numpy as np
import tqdm

class ResolventAnalysis:
    def __init__(self, path, dom) -> None:
        self.path = path
        self.dom = dom
        self._load(dom)

    def _load(self, dom):
        self.Lambda = np.load(f"{self.path}/{dom}_Lambda.npy")
        self.V_r = np.load(f"{self.path}/{dom}_V_r.npy")

    def calc_resolvent(self, omega):
        omegaSpan = np.linspace(0, 1000, 2000)
        gain = np.empty((omegaSpan.size, Lambda.size))
        for idx, omega in tqdm(enumerate(omegaSpan)):
            R = np.linalg.svd(F_tilde@np.linalg.inv((-1j*omega)*np.eye(Lambda.shape[0])-np.diag(Lambda))@np.linalg.inv(F_tilde),
                            compute_uv=False)
            gain[idx] = R**2