import numpy as np
import time
from tqdm import tqdm


class LoadData:
    def __init__(self, path, T=1, xlims=[-0.35, 2], ylims=[-0.35, 0.35]):
        """
        Initialize the LoadData class.

        Args:
        - path (str): Path to the data directory.
        - filename (str, optional): Name of the HDF5 file. Defaults to "uvp.hdf5".
        - T (int, optional): Time parameter. Defaults to 1.
        """
        self.path = path
        self.xlims = xlims  # Default values
        self.ylims = ylims  # Default values
        self.T = T
        self._load_data()
        self._wake_data_cache = None # Cache for wake data
        self._body_data_cache = None # Cache for body data

    def _load_data(self):
        """Load the data from the file and subtract the mean."""
        try:
            print(f"\n----- Loading data from {self.path}/uvp.npy -----")
            self.uvp = np.load(f"{self.path}/uvp.npy", mmap_mode='r')
            print(f"Data of shape {self.uvp.shape} loaded.")
            _, self.nx, self.ny, self.nt = self.uvp.shape
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def _subtract_mean(self):
        t0 = time.time()
        print("\n----- Subtracting mean -----")
        uvp_mean = self.uvp.mean(axis=3, keepdims=True)
        self.uvp = self.uvp - uvp_mean
        print(f"Mean subtracted in {time.time() - t0:.2f} seconds.")
        del uvp_mean

    def _filter_data(self, mask):
        """Filter data based on a provided mask."""
        print("\n----- Masking data -----")
        t0 = time.time()
        self.uvp = self.uvp[:, mask, :, :]
        _, self.nx, self.ny, self.nt = self.uvp.shape
        print(f"Data masked in {time.time() - t0:.2f} seconds.")

    @property
    def wake(self):
        """Extract data in the wake."""
        if self._wake_data_cache is None:
            pxs = np.linspace(*self.xlims, self.nx)
            mask = pxs > 1
            self.xlims[0] = 1
            self._filter_data(mask)
            np.save(f'{self.path}/wake_nxyt.npy', np.array([self.nx, self.ny, self.nt]))
            self._subtract_mean()
            self._wake_data_cache = self.uvp
        return self._wake_data_cache

    @property
    def body(self):
        """Extract data in body region."""
        if self._body_data_cache is None:
            pxs = np.linspace(*self.xlims, self.nx)
            mask = (pxs > 0) & (pxs < 1)
            self.xlims = [0, 1]
            self._filter_data(mask)
            self._subtract_mean()
            self._body_data_cache = self.uvp
        return self._body_data_cache

    @property
    def unwarped_body(self):
        """Unwarp the velocity field in the body region."""
        pxs = np.linspace(*self.xlims, self.nx)
        ts = np.linspace(0, self.T, self.nt)
        dy = (-self.ylims[0] + self.ylims[1]) / self.ny
        unwarped = np.full(self.body.shape, np.nan)

        t0 = time.time()
        print("\n----- Unwarping body data -----")

        for idt, t in tqdm(enumerate(ts), total=len(ts)):
            # Calculate the shifts for each x-coordinate
            fw = fwarp(t, pxs)
            shifts = np.round(fw / dy).astype(int)

            for i in range(self.body.shape[1]):
                shift = shifts[i]
                if shift > 0:
                    unwarped[:, i, shift:, idt] = self.body[:, i, :-shift, idt]
                elif shift < 0:
                    unwarped[:, i, :shift, idt] = self.body[:, i, -shift:, idt]
        
        del self._body_data_cache
        print(f"Body data unwarped in {time.time() - t0:.2f} seconds.")

        print("\n----- Clipping in y -----")
        t0 = time.time()
        pys = np.linspace(*self.ylims, self.ny)
        self.ylims = [-0.25, 0.25]
        mask = ((pys > self.ylims[0]) & (pys < self.ylims[1]))
        unwarped = unwarped[:, :, mask, :]
        # replace NaN with 0
        unwarped = np.nan_to_num(unwarped)
        print(f"Clipped in y in {time.time() - t0:.2f} seconds, ny went from {self.ny} to {unwarped.shape[2]}.")
        _, self.nx, self.ny, self.nt = unwarped.shape
        np.save(f'{self.path}/body_nxyt.npy', np.array([self.nx, self.ny, self.nt]))
        return unwarped

    def flat_subdomain(self, region):
        """Flatten the velocity field."""
        if region=='wake':
            np.save(f'{self.path}/wake_nxyt.npy', np.array([self.nx, self.ny, self.nt]))
            return self.wake.reshape(3 * self.nx * self.ny, self.nt)
        elif region=='body':
            return self.unwarped_body.reshape(3 * self.nx * self.ny, self.nt)
        else:
            raise ValueError("Invalid region. Must be 'wake' or 'body'.")


def fwarp(t: float, pxs: np.ndarray):
    """Warping function based on time and position."""
    # Define constants
    A = -0.5
    B = 0.28
    C = 0.13
    D = 0.05
    k = 1.42

    return A * (B * pxs**2 - C * pxs + D) * np.sin(2 * np.pi * (t - (k * pxs)))


# Sample usage
if __name__ == "__main__":
    path = "data/0/data"
    data_loader = LoadData(path, T=14)
    data_loader.flat_subdomain('wake').shape
    # data_loader = LoadData(path, T=4)
    # data_loader.flat_subdomain('wake').shape

