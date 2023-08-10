import os
import numpy as np

class LoadData:
    def __init__(self, path, xlims, ylims, tlims, T):
        self.path = path
        self.xlims = xlims
        self.ylims = ylims
        self.tlims = tlims
        self.nx = None
        self.ny = None
        self.nt = None
        self.pxs = None
        self.pys = None
        self.T = T

    def load(self, field):
        data = np.load(os.path.join(self.path, field + ".npy"))
        data = np.einsum("ijk -> kji", data)
        self.nx, self.ny, self.nt = data.shape
        return data #[::2, ::2, :]
    
    def collect_uvp(self):
        u = self.load("u")
        v = self.load("v")
        p = self.load("p")
        assert u.shape == v.shape == p.shape
        self.nx, self.ny, self.nt = u.shape
        uvp = np.concatenate((u, v, p), axis=0)
        del u, v, p
        return uvp

    def wake(self):
        # Assume the body length has been normalised
        self.xlims[0] = 1
        # Assuming pxs is some kind of mask based on xlims, the below line is just a placeholder
        # You'd need to replace it with the appropriate logic.
        mask = np.arange(self.nx) > self.xlims[0]
        uvp = self.collect_uvp()
        return uvp[:, mask, :, :]

# Sample usage
# data_loader = LoadData(path="path_to_files", xlims=[0, 10], ylims=[0, 10], tlims=[0, 10])
# wake_data = data_loader.wake()