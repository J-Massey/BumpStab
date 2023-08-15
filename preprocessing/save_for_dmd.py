from load_data import LoadData
import numpy as np
from scipy.linalg import svd
import os
import time


class SVDSave:
    def __init__(self, path, subdomain):
        self.uvp = LoadData(path, T=14).flat_subdomain(subdomain)
        self.path = path
        self.subdomain = subdomain

    def _calc_svd(self):
        """Calculate the SVD of the velocity field."""
        print("\n----- Calculating SVD -----")
        t0 = time.time()
        print((self.uvp.shape))
        Ub, Sigmab, VTb = svd(self.uvp[:, 1:], full_matrices=False)
        Uf, Sigmaf, VTf = svd(self.uvp[:, :-1], full_matrices=False)
        print(f"SVD calculated in {time.time() - t0:.2f} seconds")
        return Ub, Sigmab, VTb, Uf, Sigmaf, VTf
    
    def save_flucs(self):
        """Save the fluctuating part of the velocity field."""
        print(f"\n----- Saving fluctuations -----")
        fnsave = os.path.join(os.path.abspath(self.path + '/..'), f"{self.subdomain}_flucs.npy")
        np.save(fnsave, self.uvp)
        print(f"Fluctuations saved to {fnsave}")

    def save_svd(self):
        """Save the SVD of the velocity field."""
        Ub, Sigmab, VTb, Uf, Sigmaf, VTf = self._calc_svd()
        print(f"\n----- Saving SVD -----")
        fnsave = os.path.join(os.path.abspath(self.path + '/..'), f"{self.subdomain}_svd.npz")
        np.savez(
            fnsave,
            Ub=Ub,
            Sigmab=Sigmab,
            VTb=VTb,
            Uf=Uf,
            Sigmaf=Sigmaf,
            VTf=VTf,
        )
        print(f"SVD saved to {fnsave}")


# Sample usage
if __name__ == "__main__":
    case = "0"
    doms = ["body", "wake"]
    for dom in doms:
        svd_save = SVDSave(f"data/{case}/data", dom)
        svd_save.save_flucs()
        svd_save.save_svd()
    
