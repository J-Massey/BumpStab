import os
import numpy as np
import h5py

class LoadData:
    def __init__(self, path, T=1):
        """
        Initialize the LoadData class.
        
        Args:
        - path (str): Path to the data directory.
        - filename (str, optional): Name of the HDF5 file. Defaults to "uvp.hdf5".
        - T (int, optional): Time parameter. Defaults to 1.
        """
        self.path = path
        self.xlims = [0, 1]  # Default values
        self.ylims = [0, 1]  # Default values
        self.T = T
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from the file and set class attributes."""
        try:
            with h5py.File(self.path, "r") as f:
                group = f["data_group"]
                self.xlims = group.attrs['xlims']
                self.ylims = group.attrs['ylims']
                _, self.nx, self.ny, self.nt = group['uvp'].shape
        except Exception as e:
            print(f"Error loading data: {e}")

    def _load_data(self):
        """Load the data from the file and subtract the mean."""
        try:
            with h5py.File(self.path, "r") as f:
                group = f["data_group"]
                uvp = np.array(group['uvp'])
                uvp_mean = uvp.mean(axis=3, keepdims=True)
                uvp = uvp - uvp_mean
                del uvp_mean
                return uvp
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _filter_data(self, mask):
        """Filter data based on a provided mask."""
        uvp = self._load_data()
        if uvp is not None:
            uvp = uvp[:, mask, :, :]
            _, self.nx, self.ny, self.nt = uvp.shape
        return uvp

    def wake(self):
        """Extract data in the wake."""
        pxs = np.linspace(*self.xlims, self.nx)
        mask = pxs > 1
        self.xlims[0] = 1
        return self._filter_data(mask)

    def body(self):
        """Extract data in body region."""
        pxs = np.linspace(*self.xlims, self.nx)
        mask = (pxs > 0) & (pxs < 1)
        self.xlims = [0, 1]
        return self._filter_data(mask)

    def unwarp_velocity_field(self):
        """Unwarp the velocity field in the body region."""
        uvp = self.body()
        pxs = np.linspace(*self.xlims, self.nx)
        ts = np.linspace(0, self.T, self.nt)
        dy = (- self.ylims[0] + self.ylims[1]) / self.ny
        uvp_warped = np.full(uvp.shape, np.nan)
        
        for idt, t in enumerate(ts):
            # Calculate the shifts for each x-coordinate
            fw = fwarp(t, pxs)
            shifts = np.round(fw/dy).astype(int)

            for i in range(uvp.shape[1]):
                shift = shifts[i]
                if shift > 0:
                    uvp_warped[:, i, shift:, idt] = uvp[:, i, :-shift, idt]
                elif shift < 0:
                    uvp_warped[:, i, :shift, idt] = uvp[:, i, -shift:, idt]
                
        return uvp_warped

    def flat_wake(self):
        """Flatten the velocity field."""
        uvp = self.wake()
        print(self.nx, self.ny, self.nt)
        return uvp.reshape(3 * self.nx * self.ny, self.nt)
    
    def flat_body(self):
        """Flatten the velocity field."""
        uvp = self.unwarp_velocity_field()
        print(self.nx, self.ny, self.nt)
        return uvp.reshape(3 * self.nx * self.ny, self.nt)

def fwarp(t: float, pxs: np.ndarray):
    """Warping function based on time and position."""
    # Define constants
    A = -0.5
    B = 0.28
    C = 0.13
    D = 0.05
    k = 1.42

    return A * (B * pxs**2 - C * pxs + D) * np.sin(2*np.pi * (t - (k * pxs)))

# Sample usage
# data_loader = LoadData("data/0/uvp/uvp.hdf5")
# print(data_loader._load_data().shape)
# print(data_loader.xlims)

# data_loader = LoadData("data/0/uvp/uvp.hdf5")
# print(data_loader.wake().shape)
# print(data_loader.xlims)

# data_loader = LoadData("data/0/uvp/uvp.hdf5")
# print(data_loader.body().shape)
# print(data_loader.xlims)

# data_loader = LoadData("data/0/uvp/uvp.hdf5")
# print(data_loader.unwarp_velocity_field().shape)
# print(data_loader.xlims)

data_loader = LoadData("data/0/uvp/uvp.hdf5")
print(data_loader.wake().shape)
print(data_loader.xlims)

data_loader = LoadData("data/0/uvp/uvp.hdf5")
print(data_loader.flat_wake().shape)
print(data_loader.xlims)




