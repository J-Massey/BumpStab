import os
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import NullFormatter

from tqdm import tqdm
import sys
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import welch, savgol_filter


plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')


def read_forces(force_file, interest="p", direction="x"):
    names = [
        "t",
        "dt",
        "px",
        "py",
        "pz",
        "cp",
        "vx",
        "vy",
        "vz",
        "E",
        "tke",
    ]
    fos = np.transpose(np.genfromtxt(force_file))

    forces_dic = dict(zip(names, fos))
    t = forces_dic["t"]
    u = forces_dic[interest + direction]

    u = np.squeeze(np.array(u))

    # u = ux * 0.2 / 2.12, uy * 0.2 / 2.12

    return t, u * 0.1


def save_fig(save_path):
    plt.savefig(save_path, dpi=700)
    plt.close()


def load_plot(path, ax, omega_span, colour, label):
    gain = np.load(path)
    ax.loglog(
        omega_span / (2 * np.pi),
        np.sqrt(gain[:, 0]),
        color=colour,
        label=label,
        alpha=0.8,
        linewidth=0.7,
    )


def plot_thrust():
    lams = [16, 32, 64, 128]
    cases = [f"0.001/{lam}" for lam in lams]

    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle C_T \rangle $")
    ax.set_xlabel(r"$t/T$")

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="p", direction="x")
        t_new = t % 1
        f = interp1d(t_new, force, fill_value="extrapolate")
        force_av = f(t_sample)


        wrap_indices = np.where(np.diff(t_new) < 0)[0] + 1
        wrap_indices = np.insert(wrap_indices, 0, 0)  # Include the start index
        wrap_indices = np.append(wrap_indices, len(t_new))  # Include the end index


        force_bins = [force[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]
        t_bins = [t_new[i:j] for i, j in zip(wrap_indices[:-1], wrap_indices[1:])]


        # Calculate the standard deviation of each bin
        force_diff = np.empty((4, t_sample.size))
        for id in range(len(force_bins)):
            f_bins = interp1d(t_bins[id], force_bins[id], fill_value="extrapolate")
            force_bins[id] = f_bins(t_sample)
            force_diff[id] = force_bins[id] - force_av

        ax.plot(
            t_sample,
            force_av,
            color=colours[idx],
            label=labels[idx],
            alpha=0.8,
            linewidth=0.7,
        )

        ax.fill_between(
            t_sample,
            force_av + np.min(force_diff, axis=0),
            force_av + np.max(force_diff, axis=0),
            color=colours[idx],
            alpha=0.3,
            edgecolor="none",
        )

    path = f"data/test/up/lotus-data/fort.9"
    t, force = read_forces(path, interest="p", direction="x")
    t, force = t[((t > 8) & (t < 12))], force[((t > 8) & (t < 12))]
    t = t % 1
    f = interp1d(t, force, fill_value="extrapolate")
    force = f(t_sample)

    ax.plot(
        t_sample,
        force,
        color=colours[idx + 1],
        label="Smooth",
        alpha=0.8,
        linewidth=0.7,
    )

    save_path = f"figures/thrust.png"
    # ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def plot_power(ax=None):
    lams = [0, 16, 32, 64, 128]
    order = [2, 0, 3, 4, 1]
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$ C_P  $")
    ax.set_xlabel(r"$\varphi$")

    for idx, case in enumerate(lams):
        path = f"data/0.001/{case}/{case}/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        t_mask = t > 5
        t_new = t % 1
        ax.scatter(
            t_new[t_mask],
            force[t_mask],
            color=colours[order[idx]],
            alpha=0.8,
            marker=".",
            s=0.2,
            edgecolor="none",
        )
        # print(case, force[t_mask].mean())

        path = f"data/0.001/{case}/spressure/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        t_new = t % 1

        ax.scatter(
            t_new,
            force,
            color=colours[order[idx]],
            alpha=0.7,
            marker=".",
            s=0.2,
            edgecolor="none",
        )

        print(case, force.mean())
    
    path = f"data/variable-roughness/all-rough/var_surface/fort.9"
    t, force = read_forces(path, interest="cp", direction="")
    t_mask = np.logical_and(t > 5, t < 10)
    t_new = t % 1

    ax.scatter(
        t_new[t_mask],
        force[t_mask],
        color="red",
        alpha=0.8,
        marker=".",
        s=.2,
        edgecolor="none",
    )
    print("Var", force[t_mask].mean())

    # if ax is None:
    plt.savefig(f"figures/variable-roughness/power_scat.png", dpi=700)
    plt.savefig(f"figures/variable-roughness/power_scat.pdf")
    # plt.close()
    return ax


def plot_power_fft(ax=None):
    lams = [0, 128]
    order = [2, 1, 2]
    cases = [f"0.001/{lam}" for lam in lams]
    colours = sns.color_palette("colorblind", 7)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"PSD($C_P$)")
    ax.set_xlabel(r"$f^*$")
    ax.set_xlim(0.1, 200)
    # ax.set_ylim(1e-16, 1e-2)
    # axins = ax.inset_axes([0.1, 0.1, 0.5, 0.25], xlim=(10, 40), ylim=(1e-10, 1e-7), xticklabels=[], yticklabels=[])
    # axins.set_xscale("log")
    # axins.set_yscale("log")

    for idx, case in enumerate(cases):
        path = f"data/{case}/spressure/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        dt = 2/len(t)

        freq, Pxx = welch(force, 1/dt, nperseg=len(t//1))
        # Pxx = savgol_filter(Pxx, 4, 1)
        
        ax.loglog(freq, Pxx, color=colours[order[idx]], alpha=0.8, linewidth=0.7)
        # axins.plot(freq, Pxx, color=colours[order[idx]], alpha=0.8, linewidth=0.7)

    save_path = f"figures/fft_power.pdf"

    # Add a box to annotate inset
    # ax.indicate_inset_zoom(axins, edgecolor="k", alpha=0.8, linewidth=0.5)

    
    if ax is None:
        plt.savefig(save_path, dpi=700)
    # plt.close()
    return ax


def plot_power_diff_fft(ax=None):
    lams = [16, 32, 64, 128]
    order = [0, 3, 4, 1]
    cases = [f"0.001/{lam}" for lam in lams]
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"PSD($C_P$)")
    ax.set_xlabel(r"$f^*$")
    ax.set_xlim(0.1, 200)

    # Adding the 'Smooth' curve
    path = f"data/test/span64/lotus-data/fort.9"
    t, force = read_forces(path, interest="cp", direction="")
    t, sforce = t[((t > 12) & (t < 16))], force[((t > 12) & (t < 16))]

    t_smaple = np.linspace(0, 4, 12000)
    f = interp1d(t, sforce, fill_value="extrapolate")
    sforce = f(t_smaple)

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, force = read_forces(path, interest="cp", direction="")
        t, force = t[((t > 8) & (t < 12))], force[((t > 8) & (t < 12))]
        f = interp1d(t, force, fill_value="extrapolate")
        force = f(t_smaple)
        dt = 4/len(t_smaple)

        freq, Pxx = welch(sforce-force, 1/dt, nperseg=len(t_smaple//2))
        # Pxx = savgol_filter(Pxx, 4, 1)

        ax.loglog(freq, Pxx, color=colours[order[idx]], label=labels[idx], alpha=0.8, linewidth=0.7)

    dt = np.mean(np.diff(t))

    freq, Pxx = welch(force, 1/dt, nperseg=len(t//2))
    # Applay savgiol filter
    # Pxx = savgol_filter(Pxx, 4, 1)
    ax.loglog(freq, Pxx, color=colours[2], label="Smooth", alpha=0.8, linewidth=0.7)

    save_path = f"figures/fft_power_diff.pdf"
    
    plt.savefig(f"figures/fft_power_diff.pdf", dpi=700)
    plt.savefig(f"figures/fft_power_diff.png", dpi=700)

    # plt.close()
    return ax


def save_legend(leg):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Transfer the legend to the new figure
    new_legend = plt.legend(handles=leg.legendHandles, labels=[t.get_text() for t in leg.texts], loc='center')

    # Save only the legend
    fig.canvas.draw()
    bbox = new_legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('figures/legend.pdf', dpi=300, bbox_inches=bbox)



def plot_combined():
    """
    Plot the power and fft next to each other
    """
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].text(-0.15, 0.98, r"(a)", transform=axs[0].transAxes)
    axs[1].text(-0.15, 0.98, r"(b)", transform=axs[1].transAxes)

    # First plot
    ax1 = axs[0]
    plot_power(ax=ax1)

    # Second plot
    ax2 = axs[1]
    plot_power_fft(ax=ax2)

    latex_table = generate_latex_table()
    fig.text(0.5, 0.97, f'${latex_table}$', horizontalalignment='center', verticalalignment='bottom', usetex=True)

    fig.tight_layout()
    save_path = "figures/power.pdf"
    plt.savefig(save_path, dpi=700)



def test_E_scaling():
    path = "/research/sharkdata/research_filesystem/thicc-swimmer/128_z_res_test/128/128/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_short = t[t > 5], enst[t > 5]/(64*2)
    ts = t % 1
    path = "data/0.001/128/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_long = t[t > 5], enst[t > 5]/(64*4)
    tl = t % 1
    path = "/scratch/jmom1n15/full-body-bumps/0.001/128/128/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_wide = t[t > 5], enst[t > 5]/(128*4)
    tw = t % 1

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle E \rangle $")
    ax.set_xlabel(r"$t/T$")
    ax.plot(ts[10:], enst_short[10:], label="Short", ls="none", marker="o", markersize=0.1)
    ax.plot(tl[10:], enst_long[10:], label="Long", ls="none", marker="o", markersize=.1)
    ax.plot(tw[10:], enst_wide[10:], label="Wide", ls="none", marker="o", markersize=.1)
    ax.legend()
    plt.savefig(f"figures/test_z_span_128.pdf", dpi=300)
    plt.close()
    print((enst_wide.max()-enst_long.max())/enst_wide.max())

def test_E_span_scaling():
    path = "data/test/span128/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_3d = t[t > 6], enst[t > 6]/(32*4)
    ts = t % 1
    path = "data/test/up/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst_2d = t[t > 6], enst[t > 6]
    tl = t % 1

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle E \rangle $")
    ax.set_xlabel(r"$t/T$")
    ax.plot(ts[10:], enst_3d[10:], label="3-D", ls="none", marker="o", markersize=0.1)
    ax.plot(tl[10:], enst_2d[10:], label="2-D", ls="none", marker="o", markersize=.1)
    ax.legend()
    plt.savefig(f"figures/test_3d_enst.pdf", dpi=300)
    plt.close()
    print((enst_3d.mean()-enst_2d.mean())/enst_3d.mean())


def plot_E_body_ts_3d():
    cmap = sns.color_palette("Reds", as_cmap=True)
    Ls = np.array([4096, 8192])

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.set_xlabel(f"$ t $")
    ax.set_ylabel(r"$ E $")

    t_sample = np.linspace(2.01, 6.99, 300)

    norm = colors.LogNorm(vmin=min(4 / (Ls*2)), vmax=max(4 / (Ls/2)))
    for L in (Ls):
        try:
            f=field_eval_helper(L, 'grid-3-medium', 'E','b')
            ax.plot(
                t_sample,
                f(t_sample),
                color=cmap(norm(4 / L)),
                ls=':',
            )
            f=field_eval_helper(L, '3d-check', 'E','b')
            ax.plot(
                t_sample,
                f(t_sample)/(L/8),
                color=cmap(norm(4 / L)),
                ls='-',
            )
        except FileNotFoundError or ValueError:
            pass

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    cb.set_label(r"$\Delta x$", rotation=0)

    legend_elements = [Line2D([0], [0], ls='-', label='3-D', c='k'),
                       Line2D([0], [0], ls=':', label='2-D', c='k')]
    ax.legend(handles=legend_elements)
    plt.savefig(f"{os.getcwd()}/figures/E.pdf", bbox_inches="tight", dpi=200)


def field_eval_helper(L, crit_str='3d-check', interest='v', direction='x'):
    t, ux, *_ = read_forces(
            f"/ssdfs/users/jmom1n15/thicc-swimmer/two-dim-convergenence-test/{crit_str}/{L}/fort.9",
            interest=interest,
            direction=direction,
        )
    t, ux = t[t > 2], ux[t > 2]
    f = interp1d(t, ux, fill_value="extrapolate")

    return f


def plot_E():
    lams = [16, 32, 64, 128]
    order = [0, 3, 4, 1]

    cases = [f"0.001/{lam}" for lam in lams]

    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    t_sample = np.linspace(0.001, 0.999, 400)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_ylabel(r"$\langle E \rangle $")
    ax.set_xlabel(r"$\phi/2\pi$")

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, enst = read_forces(path, interest="E", direction="")
        t, enst = t[((t > 8.01) & (t < 12))], enst[((t > 8.01) & (t < 12))]

        t_new = t % 1
        
        enst = enst/span(1/lams[idx])

        ax.plot(
            t_new,
            enst,
            color=colours[order[idx]],
            label=labels[idx],
            alpha=0.8,
            linestyle="none",
            marker="o",
            markersize=0.1,
        )

    path = f"data/test/span64/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    t, enst = t[((t > 12.01) & (t < 16))], enst[((t > 12.01) & (t < 16))]/(64*4)
    t_new = t % 1

    ax.plot(
        t_new,
        enst,
        color=colours[2],
        label="Smooth",
        alpha=0.8,
        linestyle="none",
        marker="o",
        markersize=0.1,
    )

    # ax.set_yscale("log")

    save_path = f"figures/E.pdf"
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_E_fft():
    lams = [64, 128]
    order = [4, 1]
    cases = [f"0.001/{lam}" for lam in lams]  
    labels = [f"$\lambda = 1/{lam}$" for lam in lams]
    labels.append("Smooth")
    colours = sns.color_palette("colorblind", 7)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel(r"PSD(E)")
    ax.set_xlabel(r"$f^*$")
    ax.set_xlim(0.1, 150)

    for idx, case in enumerate(cases):
        path = f"data/{case}/lotus-data/fort.9"
        t, enst = read_forces(path, interest="E", direction="")
        dt = 4/len(t)

        enst = enst/span(1/lams[idx])

        freq, Pxx = welch(enst, 1/dt, nperseg=len(t//2))

        # Pxx = savgol_filter(Pxx, 4, 1)
        # ax.axvline(lams[idx]/2, color=colours[order[idx]], alpha=0.8, linewidth=0.7, ls="-.")
        ax.axvline(lams[idx], color=colours[order[idx]], alpha=0.8, linewidth=0.7, ls="-.")

        ax.loglog(freq, Pxx, color=colours[order[idx]], label=labels[idx], alpha=0.8, linewidth=0.7)

    # Adding the 'Smooth' curve
    path = f"data/test/span64/lotus-data/fort.9"
    t, enst = read_forces(path, interest="E", direction="")
    enst = enst/(64*4)
    t, enst = t[((t > 12) & (t < 16))], enst[((t > 12) & (t < 16))]
    dt = np.mean(np.diff(t))


    freq, Pxx = welch(enst, 1/dt, nperseg=len(t//2))
    # Applay savgiol filter
    # Pxx = savgol_filter(Pxx, 4, 1)
    ax.loglog(freq, Pxx, color=colours[2], label="Smooth", alpha=0.8, linewidth=0.7)

    save_path = f"figures/fft_E.pdf"
    ax.legend(loc="lower left")
    plt.savefig(save_path, dpi=700)
    plt.close()


def SA_enstrophy_scaling(span):
        return (
            1 / 0.1             # A
            / (1)     # L
            / (span * 4096)  # span
        )

def span(lam):
    if lam == 1/16:
        span = 128
    elif lam == 1/32:
        span = 128
    else:
        span = 64
    
    return span*4


def generate_latex_table():
    return r'\begin{tabular}{lccccc}' + \
        r'$\lambda$ & $1/16$ & $1/32$ & $1/64$ &  $1/128$ &  $1/0$ \\' + \
        r'$\overline{C_P}$ & 0.139 & 0.138 & 0.125 & 0.122 & 0.124 \\' + \
        r'\end{tabular}'


if __name__ == "__main__":
    # test_E_scaling()
    plot_power()
    # plot_power_diff_fft()
    
    # plot_E()
    # plot_E_fft()
    # plot_combined()
    # test_E_span_scaling()
    # save_legend()

    # plot_E_body_ts_3d()
