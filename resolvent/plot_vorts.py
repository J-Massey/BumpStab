import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import affine_transform, rotate
import cv2

from load_data import LoadData
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


def plot_vort_cascade(cases, bodies, time_values, idxs):
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
                avg_vort.T,
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


def plot_du_dy(cases, bodies, time_values, idxs):
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

    lims = [-1000, 1000]
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

        dudy = np.gradient(body[0, :, :, :], pys, axis=1)
        print(dudy.max(), dudy.min())

        for idx, n in enumerate(idxs):
            avg_vort = (dudy[:, :, n]+dudy[:, :, n+199]+dudy[:, :, n+399]+dudy[:, :, n+599])/4
            cs = axs[idx, idxc].contourf(
                pxs,
                pys,
                avg_vort.T,
                levels=levels,
                vmin=lims[0],
                vmax=lims[1],
                cmap=_cmap,
                extend="both",
            )
            axs[idx, idxc].set_ylim([-0.2, 0.2])
            axs[idx, idxc].set_xlim([0., 1])
            axs[idx, idxc].set_aspect(1)

    # cbar on top of the plot spanning the whole width
    cax = fig.add_axes([0.175, 0.92, 0.7, 0.04])
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\langle \partial u / \partial y \rangle$", labelpad=-25, rotation=0)

    fig_width, fig_height = fig.get_size_inches()
    new_width = 8
    scale_factor = new_width / fig_width
    new_height = fig_height * scale_factor
    fig.set_size_inches(new_width, new_height)

    plt.savefig(f"figures/power-recovery/dudy.pdf", dpi=800)
    plt.savefig(f"figures/power-recovery/dudy.png", dpi=800)


def plot_vort(vorticity, tidx, time_value, case=0):
    fig, ax = plt.subplots(figsize=(5,3), sharex=True, sharey=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"$t = {time_value:.2f}$", fontsize=9)

    lims = [-200, 200]
    levels = np.linspace(lims[0], lims[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)

    nx, ny, nt = vorticity.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    tail_mask = pxs > 0.6

    bodt = np.array([naca_warp(xp) - fwarp(time_value, xp) for xp in pxs])
    ax.plot(pxs[tail_mask], bodt[tail_mask], color='k', linewidth=0.7, label=r"Body")
    bodb = np.array([-naca_warp(xp) - fwarp(time_value, xp) for xp in pxs])
    ax.plot(pxs[tail_mask], bodb[tail_mask], color='k', linewidth=0.7)

    avg_vort = vorticity[:, :, tidx]#::200].mean(axis=2)
    cs = ax.contourf(
        pxs,
        pys,
        avg_vort.T,
        levels=levels,
        vmin=lims[0],
        vmax=lims[1],
        cmap=_cmap,
        extend="both",
    )

    ax.set_ylim([-0.1, 0.15])
    ax.set_xlim([0.6, 1])
    ax.set_aspect(1)

    # cbar on top of the plot spanning the whole width
    cax = fig.add_axes([0.175, 0.95, 0.7, 0.06])
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\langle \omega_z \rangle$", labelpad=-22, rotation=0, fontsize=9)

    plt.savefig(f"figures/phase-info/tail-strucs/warped/{case}_vort_{time_value:.2f}.pdf", dpi=800)
    plt.savefig(f"figures/phase-info/tail-strucs/{case}_vort_{time_value:.2f}.png", dpi=800)
    plt.close()


def plot_unwarped_vort(vorticity, tidx, time_value, case=0):
    fig, ax = plt.subplots(figsize=(5,3), sharex=True, sharey=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"$t = {time_value:.2f}$", fontsize=9)

    lims = [-200, 200]
    levels = np.linspace(lims[0], lims[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)

    nx, ny, nt = vorticity.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    tail_mask = pxs > 0.6

    # bodt = np.array([naca_warp(xp) for xp in pxs])
    # ax.plot(pxs[tail_mask], bodt[tail_mask], color='k', linewidth=0.7, label=r"Body")
    # bodb = np.array([-naca_warp(xp) for xp in pxs])
    # ax.plot(pxs[tail_mask], bodb[tail_mask], color='k', linewidth=0.7)

    avg_vort = vorticity[:, :, tidx]#.mean(axis=2)
    cs = ax.contourf(
        pxs,
        pys,
        avg_vort.T,
        levels=levels,
        vmin=lims[0],
        vmax=lims[1],
        cmap=_cmap,
        extend="both",
    )
    ax.set_ylim([0, 0.005])
    ax.set_xlim([0.6, 1])
    # ax.set_aspect(1)

    # cbar on top of the plot spanning the whole width
    cax = fig.add_axes([0.175, 0.95, 0.7, 0.06])
    cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lims[0], lims[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\langle \omega_z \rangle$", labelpad=-22, rotation=0, fontsize=9)

    plt.savefig(f"figures/phase-info/tail-strucs/unwarped/{case}_vort_{time_value:.2f}.pdf", dpi=800)
    plt.savefig(f"figures/phase-info/tail-strucs/unwarped/{case}_vort_{time_value:.2f}.png", dpi=800)
    plt.close()


def load_save_data():
    cases = [0, 128, 64]#, 32, 16]
    bodies = []
    for case in cases:
        dl = LoadData(f"{os.getcwd()}/data/0.001/{case}/unmasked", dt=0.005)
        body = dl.body
        np.save(f"data/0.001/{case}/unmasked/body.npy", body)
        unwarped = dl.unwarped_body
        np.save(f"data/0.001/{case}/unmasked/body_unwarped.npy", unwarped)
        vorts_unwarped(unwarped, case)


def vorts_and_all():
    cases = [0]#, 128, 64, 32]
    for idx, case in enumerate(cases):
        body = np.load(f"data/0.001/{case}/unmasked/body.npy", allow_pickle=True)
        _, nx, ny, nt = body.shape
        pxs = np.linspace(0, 1, nx)
        pys = np.linspace(-0.35, 0.35, ny)
        vorticity = np.gradient(body[1, :, :, :], pxs, axis=0) - np.gradient(body[0, :, :, :], pys, axis=1)

        time_values = np.arange(0., 1, 0.02)
        tidxs = (time_values * 200).astype(int)
        for tidx in tqdm(tidxs, desc="Plot loop"):
            plot_vort(vorticity, tidx, time_values[tidxs==tidx][0], case)


def vorts_unwarped(unwarped, case):
    _, nx, ny, nt = unwarped.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    vort = np.gradient(unwarped[1, :, :, :], pxs, axis=0) - np.gradient(unwarped[0, :, :, :], pys, axis=1)
    time_values = np.arange(0., 1, 0.02)
    tidxs = (time_values * 200).astype(int)
    for tidx in tqdm(tidxs, desc="Plot loop"):
        plot_unwarped_vort(vort, tidx, time_values[tidxs==tidx][0], case)


def rotate_field(field, angle):
    """
    Rotates a given field around its bottom right corner by a specified angle.

    :param field: numpy.ndarray, the input field to be rotated.
    :param angle: float, the angle of rotation in degrees.
    :return: numpy.ndarray, the rotated field.
    """
    # Define the rotation point as the bottom right corner of the field
    rotation_point = (0, field.shape[0] - 1)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)

    # Perform the rotation
    rotated_field = cv2.warpAffine(field, rotation_matrix, (field.shape[1], field.shape[0]))

    return rotated_field


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


def flatten_field(field):
    """Unwarp the velocity field in the body region."""
    unwarped = np.full(field.shape, np.nan)
    nx, ny, nt = field.shape
    pxs = np.linspace(0, 1, nx)
    dt = 0.005

    ts = np.arange(0, dt*nt, dt)
    dy = 1 / (4*nx) # Hard coded grid spacing

    t0 = time.time()
    print("\n----- Unwarping body data -----")

    for idt, t in tqdm(enumerate(ts[:1]), total=len(ts)):
        # Calculate the shifts for each x-coordinate
        fw = fwarp(t, pxs)-[naca_warp(xp) for xp in pxs]
        real_shift = fw / dy
        shifts = np.round(real_shift).astype(int)
        shift_up = np.ceil(fw / dy).astype(int)
        shift_down = np.floor(fw / dy).astype(int)
        
        for i in range(field.shape[0]):
            shift = shifts[i]
            ushift = shift_up[i]
            dshift = shift_down[i]
            alpha = shift-real_shift[i]
            unwarped[i, :, idt] = apply_shift_interpolation(ushift, dshift, alpha, field[i, :, idt])

    # replace NaN with 0
    unwarped = np.nan_to_num(unwarped)

    # del _body_data_cache
    # unwarped = gaussian_filter1d(unwarped, sigma=2, axis=1)
    # unwarped = gaussian_filter1d(unwarped, sigma=2, axis=2)

    print(f"Body data unwarped in {time.time() - t0:.2f} seconds.")
    print("\n----- Clipping in y -----")
    t0 = time.time()
    pys = np.linspace(-0.35, 0.35, ny)
    ylims = [-0.25, 0.25]
    mask = ((pys > ylims[0]) & (pys < ylims[1]))
    unwarped = unwarped[:, mask, :]
    print(f"Clipped in y in {time.time() - t0:.2f} seconds, ny went from {ny} to {unwarped.shape[2]}.")
    return unwarped


def cascade_contours():
    time_values = np.array([0.2, 0.225, 0.25, 0.275, 0.3,0.325, 0.35])
    tidxs = (time_values * 200).astype(int)
    fig, axs = plt.subplots(tidxs.size, 1, sharex=True, sharey=True)
    fig.text(0.5, 0.0, r"$x$", ha='center', va='center')
    fig.text(0.03, 0.5, r"$y$", ha='center', va='center', rotation='vertical')
    fig.text(0.915, 0.844, r'$\varphi$', horizontalalignment='center', verticalalignment='center')

    arrow_ax = fig.add_axes([0.88, 0.01, 0.02, 0.9])
    arrow_ax.axis('off')
    fancy_arrow = FancyArrowPatch((0.5, 1), (0.5, 0.1), mutation_scale=15, arrowstyle='->', color='k')
    arrow_ax.add_patch(fancy_arrow)

    #  Calculate the y-position for each label based on the number of rows and the figure height
    num_rows = len(time_values)
    row_height = 1 / num_rows

    # Add time labels
    for idx, time in enumerate(time_values):
        y_pos = 1 - (idx + 0.5) * row_height  # Center of each row
        fig.text(0.945, y_pos, f"{time:.2f}", verticalalignment='center')

    lims = [-300, 300]
    levels = np.linspace(lims[0], lims[1], 22)
    colours = sns.color_palette("colorblind", 7)
    order = [2, 4, 1]

    # Column Labels
    # for idxc, lab in enumerate([r"Smooth", r"$\lambda = 1/64$", r"$\lambda = 1/128$"]):
    #     axs[0, idxc].set_title(lab, fontsize=10.5)

    for idxc, case in enumerate(cases):
        vorticity = vorts[idxc]
        nx, ny, nt = vorticity.shape
        pxs = np.linspace(0, 1, nx)
        px_mask = pxs > 0.5
        pys = np.linspace(-0.25, 0.25, ny)
        py_mask = np.logical_and(pys > 0, pys < 0.1)
        rot = np.arctan(naca_warp(0.5)/0.5)

        for idx, n in enumerate(tidxs):
            avg_vort = vorticity[:, :, n]
            # clip the vorticity to the tail
            avg_vort = avg_vort[px_mask, :]
            avg_vort = avg_vort[:, py_mask]
            rotated_field = rotate_field(avg_vort, 4*rot*180/np.pi)


            co = axs[idx].contour(
                pxs[px_mask],
                pys[py_mask],
                -rotated_field.T,
                levels=levels,
                colors=[colours[order[idxc]]],
                linewidths=0.25,
                alpha=0.7,
            )
            
            axs[idx].set_ylim([0, 0.025])
            axs[idx].set_xlim([0.5, 1])
            # axs[idx].set_aspect(1)


    fig_width, fig_height = fig.get_size_inches()
    new_width = 8
    scale_factor = new_width / fig_width
    new_height = fig_height * scale_factor
    fig.set_size_inches(6, 9)

    fig.tight_layout()

    plt.savefig(f"figures/power-recovery/vorticity_tail.pdf")
    plt.savefig(f"figures/power-recovery/vorticity_tail.png", dpi=600)

# cascade_contours()

if __name__ == "__main__":
    # load_save_data()
    # cases = [64, 128]
    # bodies = []
    # for case in cases:
    #     dl = LoadData(f"{os.getcwd()}/data/0.001/{case}/unmasked", dt=0.005)
    #     body = dl.body
    #     np.save(f"data/0.001/{case}/unmasked/body.npy", body)
    #     unwarped = dl.unwarped_body
    #     np.save(f"data/0.001/{case}/unmasked/body_unwarped.npy", unwarped)

    # for case in cases:
    #     body = np.load(f"data/0.001/{case}/unmasked/body.npy", allow_pickle=True)
    #     vorts_unwarped(unwarped, case)

    # cases = [0]#, 128, 64, 32]
    case = 0
    # for idx, case in enumerate(cases):
    body = np.load(f"data/0.001/{case}/unmasked/body.npy", allow_pickle=True)
    _, nx, ny, nt = body.shape
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.35, 0.35, ny)
    vorticity = np.gradient(body[1, :, :, :], pxs, axis=0) - np.gradient(body[0, :, :, :], pys, axis=1)
    unwarped = flatten_field(vorticity)
    # np.save(f"data/0.001/{case}/unmasked/vort_unwarped.npy", unwarped)

    plot_unwarped_vort(unwarped, 0, 0, 0)
    #     plot_vort(vorticity, tidx, time_values[tidxs==tidx][0], case)
