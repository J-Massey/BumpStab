import numpy as np
import time
from tqdm import tqdm
from plot_field import plot_field, gif_gen
from scipy.ndimage import binary_dilation


class LoadData:
    def __init__(self, path, dt=0.005, xlims=[-0.35, 2], ylims=[-0.35, 0.35]):
        """
        Initialize the LoadData class.
        Args:
        - path (str): Path to the data directory.
        - filename (str, optional): Name of the HDF5 file. Defaults to "uvp.hdf5".
        - dt (float, optional): Time step parameter. Defaults to 0.005.
        """
        self.path = path
        self.xlims = xlims  # Default values
        self.ylims = ylims  # Default values
        self.dt =dt
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
        self.uvp =  self.uvp - uvp_mean
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
            self._body_data_cache = self.uvp
        return self._body_data_cache

    @property
    def unwarped_body(self):
        """Unwarp the velocity field in the body region."""
        unwarped = np.full(self.body.shape, np.nan)
        pxs = np.linspace(*self.xlims, self.nx)

        ts = np.arange(0, self.dt*self.nt, self.dt)
        dy = (-self.ylims[0] + self.ylims[1]) / self.ny

        t0 = time.time()
        print("\n----- Unwarping body data -----")
        mask_extended = self._mask_data().T

        for idt, t in tqdm(enumerate(ts), total=len(ts)):
            # Calculate the shifts for each x-coordinate
            fw = fwarp(t, pxs)
            shifts = np.round(fw / dy).astype(int)

            for i in range(self.body.shape[1]):
                shift = shifts[i]
                if shift >= 0:
                    unwarped[:, i, shift:, idt] = self.body[:, i, :self.body.shape[2]-shift, idt]
                else:
                    shift = -shift
                    unwarped[:, i, :-shift, idt] = self.body[:, i, shift:, idt]
                
                # Now mask the boundary to avoid artifacts
                for d in range(3):
                    unwarped[d, :, :, idt] = np.ma.masked_array((unwarped[d, :, :, idt]), mask=mask_extended)

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
        t0 = time.time()
        print("\n----- Subtracting mean -----")
        unwarped_mean = unwarped.mean(axis=3, keepdims=True)
        unwarped =  unwarped - unwarped_mean
        print(f"Mean subtracted in {time.time() - t0:.2f} seconds.")
        del unwarped_mean
        _, self.nx, self.ny, self.nt = unwarped.shape
        np.save(f'{self.path}/body_nxyt.npy', np.array([self.nx, self.ny, self.nt]))
        return unwarped
    
    def _mask_data(self):
        def naca_warp(x):
            a = 0.6128808410319363
            b = -0.48095987091980424
            c = -28.092340603952525
            d = 222.4879939829765
            e = -846.4495017866838
            f = 1883.671432625102
            g = -2567.366504265927
            h = 2111.011565214803
            i = -962.2003374868311
            j = 186.80721148226274

            xp = min(max(x, 0.0), 1.0)
            
            return (a * xp + b * xp**2 + c * xp**3 + d * xp**4 + e * xp**5 + 
                        f * xp**6 + g * xp**7 + h * xp**8 + i * xp**9 + j * xp**10)
        
        pxs = np.linspace(*self.xlims, self.nx)
        pys = np.linspace(*self.ylims, self.ny)
        X, Y = np.meshgrid(pxs, pys)
        Z_warp_top = np.array([naca_warp(x) for x in pxs])
        Z_warp_bottom = np.array([-naca_warp(x) for x in pxs])
        mask = (Y <= Z_warp_top) & (Y >= Z_warp_bottom)
        mask_extended = binary_dilation(mask, iterations=4)
        return mask_extended
    
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
    return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


# Sample usage
if __name__ == "__main__":
    import os
    case = "test/down"
    # case = "0.001/16"
    os.system(f"mkdir -p figures/{case}-unwarp")
    dl = LoadData(f"{os.getcwd()}/data/{case}/data", dt=0.005)
    body = dl.body
    uw = dl.unwarped_body
    _, nx, ny, nt = uw.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    plot_field(uw[1, :, :, 0].T, pxs, pys, f"figures/{case}_unwarp.pdf", lim=[-0.5, 0.5], _cmap="seismic")

    for n in range(0, nt, 5):
        plot_field(uw[1, :, :, n].T, pxs, pys, f"figures/{case}-unwarp/{n}.png", lim=[-0.5, 0.5], _cmap="seismic")
    gif_gen(f"figures/{case}-unwarp/", f"figures/{case}_unwarped.gif", 8)