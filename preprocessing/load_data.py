import os
import numpy as np
import h5py

class LoadData:
    def __init__(self, path, T=1):
        self.path = path
        self.xlims = [0, 1]  # Default values
        self.ylims = [0, 1]  # Default values
        self.nx = 0
        self.ny = 0
        self.nt = 0
        self.T = T

    def load(self):
        try:
            with h5py.File(os.path.join(self.path, "uvp.hdf5"), "r") as f:
                group = f["data_group"]
                data = np.array(group['uvp'])
                # Read the metadata attributes
                self.xlims = group.attrs['xlims']
                self.ylims = group.attrs['ylims']
                _, self.nx, self.ny, self.nt = data.shape
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        return data

    def wake(self):
        uvp = self.load()
        pxs = np.linspace(*self.xlims, self.nx)
        mask = pxs > 1
        self.xlims[0] = 1
        uvp = uvp[:, mask, :, :]
        _, self.nx, self.ny, self.nt = uvp.shape
        return uvp
    
    def body(self):
        uvp = self.load()
        pxs = np.linspace(*self.xlims, self.nx)
        mask = ((pxs>0)&(pxs < 1))
        self.xlims[1] = 1
        uvp = uvp[:, mask, :, :]
        _, self.nx, self.ny, self.nt = uvp.shape
        return uvp

# Sample usage
data_loader = LoadData(path="data/0/uvp")
uvp = data_loader.wake()
