import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

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
        dy = (-self.ylims[0] + self.ylims[1]) / self.ny

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

        # del self._body_data_cache
        # Now smooth using a savgol filter
        unwarped = gaussian_filter1d(unwarped, sigma=1, axis=1)

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

        print("\n----- Binary dilation -----")
        new_mask = mask_data(self.nx, self.ny).T
        for idt in tqdm(range(unwarped.shape[-1]), total=unwarped.shape[-1]):
            # Now mask the boundary to avoid artifacts
            for d in range(3):
                unwarped[d, :, :, idt][new_mask] = 0.

        t0 = time.time()
        print("\n----- Subtracting mean -----")
        unwarped_mean = unwarped.mean(axis=3, keepdims=True)
        unwarped =  unwarped - unwarped_mean
        print(f"Mean subtracted in {time.time() - t0:.2f} seconds.")
        del unwarped_mean

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


def clip_arrays(arr1, arr2):
    min_shape = min(arr1.shape[1], arr2.shape[1])
    return arr1[:, :min_shape], arr2[:, :min_shape]


def apply_shift_interpolation(ushift, dshift, alpha, src, s=0):
    """
    Applies a shift to an array using spline interpolation based on the rounding error (alpha).
    
    Parameters:
    - ushift: The rounded shift value rounded up
    - dshift: The rounded shift value rounded down
    - alpha: The rounding error used for interpolation
    - src: The source array to be shifted
    - s: Smoothing factor for spline interpolation
    
    Returns:
    - interped: The shifted array after spline interpolation
    """
    y_indices = np.arange(len(src))
    
    # Interpolate using spline for both up and down shifts
    spline_up = UnivariateSpline(y_indices, np.roll(src, ushift), s=s)
    spline_down = UnivariateSpline(y_indices, np.roll(src, dshift), s=s)
    
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


def fwarp(t: float, pxs: np.ndarray):
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



# Sample usage
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scienceplots
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import FancyArrowPatch


    plt.style.use(["science"])
    plt.rcParams["font.size"] = "10.5"
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
    
    cases = ["test/span64", "0.001/64", "0.001/128"]
    # cases = ["test/down", "test/down", "test/down"]
    bodies = []
    for idxc, case in enumerate(cases):
        dl = LoadData(f"{os.getcwd()}/data/{case}/data", dt=0.005)
        body = dl.body
        bodies.append(body)

    time_values = np.arange(0.7, 0.95, 0.05)
    idxs = (time_values * 200).astype(int)

    fig, axs = plt.subplots(idxs.size, len(cases), sharex=True, sharey=True)
    fig.text(0.5, 0.07, r"$x$", ha='center', va='center')
    fig.text(0.08, 0.5, r"$y$", ha='center', va='center', rotation='vertical')
    fig.text(0.915, 0.844, r'$\varphi$', horizontalalignment='center', verticalalignment='center')

    arrow_ax = fig.add_axes([0.92, 0.054, 0.02, 0.79])
    arrow_ax.axis('off')
    fancy_arrow = FancyArrowPatch((0.5, 1), (0.5, 0.1), mutation_scale=15, arrowstyle='->', color='k')
    arrow_ax.add_patch(fancy_arrow)

    #  Calculate the y-position for each label based on the number of rows and the figure height
    num_rows = len(time_values)
    row_height = 0.795 / num_rows

    # Add time labels
    for idx, time in enumerate(time_values):
        y_pos = 0.89 - (idx + 0.5) * row_height  # Center of each row
        fig.text(0.945, y_pos, f"{time:.2f}", verticalalignment='center')

    lims = [-400, 400]
    levels = np.linspace(lims[0], lims[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)

    # Column Labels
    for idxc, lab in enumerate([r"Smooth", r"$\lambda = 1/64$", r"$\lambda = 1/128$"]):
        axs[0, idxc].set_title(lab, fontsize=10.5)

    for idxc, case in enumerate(cases):
        body = bodies[idxc]
        _, nx, ny, nt = body.shape
        pxs = np.linspace(0, 1, nx)
        pys = np.linspace(-0.35, 0.35, ny)

        vorticity = np.gradient(body[1, :, :, :], pxs, axis=0) - np.gradient(body[0, :, :, :], pys, axis=1)

        for idx, n in enumerate(idxs):
            avg_vort = (vorticity[:, :, n]+vorticity[:, :, n+199]+vorticity[:, :, n+399]+vorticity[:, :, n+599])/4
            cs = axs[idx, idxc].contourf(
                pxs,
                pys,
                vorticity[:, :, n].T,
                levels=levels,
                vmin=lims[0],
                vmax=lims[1],
                cmap=_cmap,
                extend="both",
            )
            axs[idx, idxc].set_ylim([-0.2, 0.2])
            axs[idx, idxc].set_xlim([0.8, 1])
            axs[idx, idxc].set_aspect(1)

    # cbar on top of the plot spanning the whole width
    cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\langle \omega_z \rangle$", labelpad=-25, rotation=0)

    fig_width, fig_height = fig.get_size_inches()
    new_width = 8
    scale_factor = new_width / fig_width
    new_height = fig_height * scale_factor
    fig.set_size_inches(7, 16)

    plt.savefig(f"figures/power-recovery/vorticity_tail.pdf", dpi=800)
    plt.savefig(f"figures/power-recovery/vorticity_tail.png", dpi=800)
    # plt.savefig(f"figures/power-recovery/vorticity.pgf", dpi=800)
    # plt.close()