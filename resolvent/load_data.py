import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import affine_transform, rotate
import cv2

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
        self._body_data_cache = None # Cache for body data
        self._wake_data_cache = None # Cache for wake data

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
    
    def body_normal(self, t):
        """Calculate the normal vector to the body surface."""
        # Calculate x and y coordinates of the point on the surface at t
        x=np.linspace(0, 1, 100)
        y = np.array([naca_warp(xp) for xp in x]) + fwarp(t, x)

        df_dx = np.gradient(y, x, edge_order=2)
        df_dy = -1

        # Calculate the normal vector to the surface
        mag = np.sqrt(df_dx**2 + df_dy**2)
        nx = -df_dx/mag
        ny = -df_dy/mag
        return nx, ny

    @property
    def unwarped_body(self):
        """Unwarp the velocity field in the body region."""
        unwarped = np.full(self.body.shape, np.nan)
        pxs = np.linspace(*self.xlims, self.nx)

        ts = np.arange(0, self.dt*self.nt, self.dt)
        dy = 1 / (4*self.nx) # Hard coded grid spacing

        t0 = time.time()
        print("\n----- Unwarping body data -----")

        for idt, t in tqdm(enumerate(ts), total=len(ts)):
            # Calculate the shifts for each x-coordinate
            fw = fwarp(t, pxs)
            real_shift = fw / dy
            shifts = np.round(real_shift).astype(int)
            shift_up = np.ceil(fw / dy).astype(int)
            shift_down = np.floor(fw / dy).astype(int)
            
            for i in range(self.body.shape[1]):
                shift = shifts[i]
                ushift = shift_up[i]
                dshift = shift_down[i]
                alpha = shift-real_shift[i]
                for d in range(3):
                    unwarped[d, i, :, idt] = apply_shift_interpolation(ushift, dshift, alpha, self.body[d, i, :, idt])

        # replace NaN with 0
        unwarped = np.nan_to_num(unwarped)

        # del self._body_data_cache
        unwarped = gaussian_filter1d(unwarped, sigma=1, axis=1)
        unwarped = gaussian_filter1d(unwarped, sigma=1, axis=2)

        print(f"Body data unwarped in {time.time() - t0:.2f} seconds.")
        print("\n----- Clipping in y -----")
        t0 = time.time()
        pys = np.linspace(*self.ylims, self.ny)
        self.ylims = [-0.25, 0.25]
        mask = ((pys > self.ylims[0]) & (pys < self.ylims[1]))
        unwarped = unwarped[:, :, mask, :]
        print(f"Clipped in y in {time.time() - t0:.2f} seconds, ny went from {self.ny} to {unwarped.shape[2]}.")
        _, self.nx, self.ny, self.nt = unwarped.shape
        np.save(f'{self.path}/body_nxyt.npy', np.array([self.nx, self.ny, self.nt]))

        # print("\n----- Binary dilation -----")
        new_mask = mask_data(self.nx, self.ny).T
        # for idt in tqdm(range(unwarped.shape[-1]), total=unwarped.shape[-1]):
        #     # Now mask the boundary to avoid artifacts
        #     for d in range(3):
        #         unwarped[d, :, :, idt][new_mask] = 0.
        return unwarped

    
    def flat_subdomain(self, region):
        """Flatten the velocity field."""
        if region=='wake':
            np.save(f'{self.path}/wake_nxyt.npy', np.array([self.nx, self.ny, self.nt]))
            return self.wake.reshape(3 * self.nx * self.ny, self.nt)
        elif region=='body':
            meanless = self._subtract_mean(self.unwarped_body)
            return self.unwarped_body.reshape(3 * self.nx * self.ny, self.nt)
        else:
            raise ValueError("Invalid region. Must be 'wake' or 'body'.")

    @staticmethod
    def _subtract_mean(self, unwarped):
        t0 = time.time()
        print("\n----- Subtracting mean -----")
        unwarped_mean = unwarped.mean(axis=3, keepdims=True)
        unwarped = unwarped - unwarped_mean
        print(f"Mean subtracted in {time.time() - t0:.2f} seconds.")
        return unwarped


def clip_arrays(arr1, arr2):
    min_shape = min(arr1.shape[1], arr2.shape[1])
    clipped = arr1[:, :min_shape], arr2[:, :min_shape]
    return clipped


def apply_shift_interpolation(ushift, dshift, alpha, slc, s=0):
    """
    Applies a shift to an array using spline interpolation based on the rounding error (alpha).
    
    Parameters:
    - ushift: The rounded shift value rounded up
    - dshift: The rounded shift value rounded down
    - alpha: The rounding error used for interpolation
    - slc: The slice of the array to be shifted
    - s: Smoothing factor for spline interpolation
    
    Returns:
    - interped: The shifted array after spline interpolation
    """
    y_indices = np.arange(len(slc))
    
    # Interpolate using spline for both up and down shifts
    spline_up = UnivariateSpline(y_indices, np.roll(slc, ushift), s=s)
    spline_down = UnivariateSpline(y_indices, np.roll(slc, dshift), s=s)
    
    interped_up = spline_up(y_indices)
    interped_down = spline_down(y_indices)
    
    # Combine both interpolations using alpha
    interped = interped_up * alpha + interped_down * (1 - alpha)
    
    return interped


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


def fwarp(t: float, pxs):
    if isinstance(pxs, float):
        x = pxs
        xp = min(max(x, 0.0), 1.0)
        return -0.5*(0.28 * xp**2 - 0.13 * xp + 0.05) * np.sin(2*np.pi*(t - (1.42* xp)))
    else:
        return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def normal_to_surface(x: np.ndarray, t):
    y = np.array([naca_warp(xp) for xp in x]) + fwarp(t, x)
    y = y*1  # Scale the y-coordinate to get away from the body 

    df_dx = np.gradient(y, x, edge_order=2)
    df_dy = -1

    # Calculate the normal vector to the surface
    mag = np.sqrt(df_dx**2 + df_dy**2)
    nx = -df_dx/mag
    ny = -df_dy/mag
    return nx, ny


def mask_data(nx, ny):    
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    X, Y = np.meshgrid(pxs, pys)
    Z_warp_top = np.array([naca_warp(x) for x in pxs])
    Z_warp_bottom = np.array([-naca_warp(x) for x in pxs])
    mask = (Y <= Z_warp_top) & (Y >= Z_warp_bottom)
    mask_extended = binary_dilation(mask, iterations=4, border_value=0)
    return mask_extended


