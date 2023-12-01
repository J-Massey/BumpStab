import numpy as np
from scipy.linalg import cholesky, svd, inv

import matplotlib.pyplot as plt
import scienceplots
import sys
import os
from tqdm import tqdm

import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import string

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
colours = sns.color_palette("colorblind", 7)


class PlotPeaks:
    def __init__(self, path, dom):
        self.path = path
        self.dom = dom
        self._load()
    
    def _load(self):
        self.Lambda = np.load(f"{self.path}/{self.dom}_Lambda.npy")
        self.V_r = np.load(f"{self.path}/{self.dom}_V_r.npy")
        print("----- Plotting modes -----")
        self.peak_omegas = np.load(f"{self.path}/{self.dom}_peak_omegas.npy")
        print(f"Peak omegas: {self.peak_omegas/(2*np.pi)}")
    
    @property
    def F_tilde(self):
            # Find the hermatian adjoint of the eigenvectors
            V_r_star_Q = self.V_r.conj().T
            V_r_star_Q_V_r = np.dot(V_r_star_Q, self.V_r)
            # Cholesky factorization
            F_tilde = cholesky(V_r_star_Q_V_r)
            return F_tilde

    def plot_forcing(self, case):

        for omega in self.peak_omegas:
            Psi, Sigma, Phi = svd(self.F_tilde@inv((-1j*omega)*np.eye(self.Lambda.shape[0])-np.diag(self.Lambda))@inv(self.F_tilde))
            for i in range(len(Sigma)):
                Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
                Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
                Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

            forcing = (self.V_r @ inv(self.F_tilde)@Psi).reshape(3, self.nx, self.ny, len(Sigma))

            field = forcing[1, :, :, 0].real
            pxs = np.linspace(0, 1, self.nx)
            pys = np.linspace(-0.25, 0.25, self.ny)
            try:
                plot_field(field.T, pxs, pys, f"figures/{case}-modes/{self.dom}_forcing_{omega/(2*np.pi):.2f}.png", _cmap="seismic")
            except ValueError:
                print(f"ValueError, {omega/(2*np.pi):.2f} dodgy")
            
    def plot_response(self, case):
        for omega in self.peak_omegas:
            Psi, Sigma, Phi = svd(self.F_tilde@inv((-1j*omega)*np.eye(self.Lambda.shape[0])-np.diag(self.Lambda))@inv(self.F_tilde))
            for i in range(len(Sigma)):
                Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
                Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
                Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

            response = (self.V_r @ inv(self.F_tilde)@Phi).reshape(3, self.nx, self.ny, len(Sigma))

            field = response[1, :, :, 0].real
            pxs = np.linspace(0, 1, self.nx)
            pys = np.linspace(-0.25, 0.25, self.ny)
            try:
                plot_field(field.T, pxs, pys, f"figures/{case}-modes/{self.dom}_response_{omega/(2*np.pi):.2f}.png", _cmap="seismic")
            except ValueError:
                print(f"ValueError, {omega/(2*np.pi):.2f} dodgy")


def plot_field(qi, pxs, pys, path=None, _cmap="seismic", lim=None, ax=None):
    # Test plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.imshow(
        qi,
        extent=[0, 1, pys.min(), pys.max()],
        cmap=_cmap,
        norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
        origin="lower",
        aspect="auto",
    )
    # ax.set_aspect(1)
    if path is not None:
        plt.savefig(path, dpi=600)
        plt.close()


def save_f_r(n):
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    omegas = smooth.peak_omegas
    s_f_r_modes = np.empty((2, len(omegas), nx, ny))
    n_f_r_modes = np.empty((2, len(omegas), nx, ny))
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Saving modes"):
        Psi, Sigma, Phi = svd(smooth.F_tilde@inv((-1j*omega)*np.eye(smooth.Lambda.shape[0])-np.diag(smooth.Lambda))@inv(smooth.F_tilde))
        # for i in range(len(Sigma)):
        #     Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
        #     Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
        #     Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

        forcing = (smooth.V_r @ inv(smooth.F_tilde)@Psi).reshape(2, nx, ny, len(Sigma))
        response = (smooth.V_r @ inv(smooth.F_tilde)@Phi).reshape(2, nx, ny, len(Sigma))
        # mag = np.sqrt(forcing[1, :, :, n].real**2 +  forcing[0, :, :, n].real**2)
        s_f_r_modes[0, ido] = forcing[0, :, :, n].real
        n_f_r_modes[0, ido] = forcing[1, :, :, n].real
        # mag = np.sqrt(response[1, :, :, n].real**2 +  response[0, :, :, n].real**2)
        s_f_r_modes[1, ido] = response[0, :, :, n].real
        n_f_r_modes[1, ido] = response[1, :, :, n].real
    np.save(f"{os.getcwd()}/data/0.001/0/unmasked/s_f_r_mode{n}.npy", s_f_r_modes)
    np.save(f"{os.getcwd()}/data/0.001/0/unmasked/n_f_r_mode{n}.npy", n_f_r_modes)


def plot_f_r():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(0, 0.25, ny)
    py_mask = np.logical_and(pys > 0, pys < 0.1)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/f_r_modes.npy")
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Plotting modes"):
        mag = f_r_modes[0, ido, :, py_mask]
        lim = 0.005
        plot_field(mag, pxs, pys[py_mask], f"figures/forcing-modes/forcing_{omega/(2*np.pi):.2f}.png", lim=[0, lim], _cmap="icefire")
        mag = f_r_modes[1, ido, :, py_mask]
        plot_field(mag, pxs, pys[py_mask], f"figures/response-modes/response_{omega/(2*np.pi):.2f}.png", lim=[0, lim], _cmap="icefire")
    

def plot_spectra():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/f_r_modes.npy")
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Plotting spectra"):
        forcing = f_r_modes[0, ido, :, :-1]
        response = f_r_modes[1, ido, :, :-1]
        nx, ny = forcing.shape
        pys = np.linspace(-0.25, 0.25, ny)
        zoom_mask = np.logical_and(pys > -0.1, pys < 0.1)
        forcing = forcing[:, zoom_mask]
        response = response[:, zoom_mask]
        nx, ny = forcing.shape

        fs = np.fft.fft2(forcing)
        fs = np.fft.fftshift(fs)
        rs = np.fft.fft2(response)
        rs = np.fft.fftshift(rs)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1/1024))
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1/4096))

        extent = [kx.min(), kx.max(), ky.min(), ky.max()]
        cutoff = 1
        kx_mask = np.logical_and(kx > -cutoff, kx < cutoff)
        ky_mask = np.logical_and(ky > -cutoff, ky < cutoff)

        fmag = np.abs(fs)
        fmag[kx_mask, :] = 0
        fmag[:, ky_mask] = 0
        rmag = np.abs(rs)
        rmag[kx_mask, :] = 0
        rmag[:, ky_mask] = 0

        lim = 4

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

        ax.imshow(
                np.log1p(fmag).T,
                extent=extent,
                vmin=0,
                vmax=lim,
                cmap=sns.color_palette("icefire", as_cmap=True),
                aspect="auto",
                origin="lower",
                # norm=norm,
            )

        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_xticks([-100, -1, 1, 100])
        ax.set_yticks([-100, -1, 1, 100])

        plt.savefig(f"figures/spectra/forcing_{omega/(2*np.pi):.2f}.png", dpi=700)
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

        ax.imshow(
                np.log1p(rmag).T,
                extent=extent,
                vmin=0,
                vmax=lim,
                cmap=sns.color_palette("icefire", as_cmap=True),
                aspect="auto",
                origin="lower",
                # norm=norm,
            )
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_xticks([-100, -1, 1, 100])
        ax.set_yticks([-100, -1, 1, 100])
    
        plt.savefig(f"figures/spectra/response_{omega/(2*np.pi):.2f}.png", dpi=300)
        plt.close()


def plot_large_forcing():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    py_mask = np.logical_and(pys > -0.25, pys < 0.25)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/f_r_modes.npy")
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

    fig, axs = plt.subplots(6,2, figsize=(6, 8.5))
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Plotting bigun"):
        mag = f_r_modes[0, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[0, 5]
        cs = axs[ido, 0].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("icefire", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect=1,
        )
        axs[ido, 0].set_xticks([])
        axs[ido, 0].set_yticks([-.2, -.1, 0, .1, .2])
        axs[ido, 0].set_ylabel(r"$y$")
        axs[ido, 0].text(-0.2, 0.92, f"({letters[ido*2]})", transform=axs[ido, 0].transAxes, fontsize=10)

        fs = np.fft.fft2(mag)
        fs = np.fft.fftshift(fs)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1/1024))
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1/4096))

        extent = [kx.min(), kx.max(), ky.min(), ky.max()]
        cutoff = 1
        kx_mask = np.logical_and(kx > -cutoff, kx < cutoff)
        ky_mask = np.logical_and(ky > -cutoff, ky < cutoff)

        fmag = np.abs(fs)
        fmag[kx_mask, :] = 0
        fmag[:, ky_mask] = 0

        im = axs[ido, 1].imshow(
                np.log1p(fmag).T,
                extent=extent,
                vmin=0,
                vmax=4,
                cmap=sns.color_palette("inferno_r", as_cmap=True),
                aspect=0.6,
                origin="lower",
                # norm=norm,
            )

        axs[ido, 1].set_xscale("symlog")
        axs[ido, 1].set_yscale("symlog")
        axs[ido, 1].set_xticks([])
        axs[ido, 1].set_yticks([-1000, -10, 0, 10, 1000])
        axs[ido, 1].set_ylabel(r"$k_y$")
        axs[ido, 1].text(-0.3, 0.92, f"({letters[ido*2+1]})", transform=axs[ido, 1].transAxes, fontsize=10)


    axs[-1, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[-1, 0].set_xlabel(r"$x$")
    axs[-1, 1].set_xticks([-100, -1, 1, 100])
    axs[-1, 1].set_xlabel(r"$k_x$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    cax1 = fig.add_axes([0.1, 1.01, 0.4, 0.02])
    cb = plt.colorbar(cs, cax=cax1, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$|\vec{u}| \quad \times 10^3$", labelpad=-40, rotation=0)

    cax2 = fig.add_axes([0.55, 1.01, 0.4, 0.02])
    cb = plt.colorbar(im, cax=cax2, orientation="horizontal", ticks=np.linspace(0, 4, 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\ln\big(1+|\mathcal{F}(|\mathbf{U}|)|\big)$", labelpad=-40, rotation=0)

    plt.savefig(f"figures/RA_forcing.pdf")
    plt.savefig(f"figures/RA_forcing.png", dpi=700)
    plt.close()


def plot_large_response():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")

    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)
    py_mask = np.logical_and(pys > -0.25, pys < 0.25)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/f_r_modes.npy")
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

    fig, axs = plt.subplots(6,2, figsize=(6, 8.5))
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Plotting bigun"):
        mag = f_r_modes[1, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[0, 5]
        cs = axs[ido, 0].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("icefire", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect=1,
        )
        axs[ido, 0].set_xticks([])
        axs[ido, 0].set_yticks([-.2, -.1, 0, .1, .2])
        axs[ido, 0].set_ylabel(r"$y$")
        axs[ido, 0].text(-0.2, 0.92, f"({letters[ido*2]})", transform=axs[ido, 0].transAxes, fontsize=10)

        fs = np.fft.fft2(mag)
        fs = np.fft.fftshift(fs)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1/1024))
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1/4096))

        extent = [kx.min(), kx.max(), ky.min(), ky.max()]
        cutoff = 1
        kx_mask = np.logical_and(kx > -cutoff, kx < cutoff)
        ky_mask = np.logical_and(ky > -cutoff, ky < cutoff)

        fmag = np.abs(fs)
        fmag[kx_mask, :] = 0
        fmag[:, ky_mask] = 0

        im = axs[ido, 1].imshow(
                np.log1p(fmag).T,
                extent=extent,
                vmin=0,
                vmax=4,
                cmap=sns.color_palette("inferno_r", as_cmap=True),
                aspect=0.6,
                origin="lower",
                # norm=norm,
            )

        axs[ido, 1].set_xscale("symlog")
        axs[ido, 1].set_yscale("symlog")
        axs[ido, 1].set_xticks([])
        axs[ido, 1].set_yticks([-1000, -10, 0, 10, 1000])
        axs[ido, 1].set_ylabel(r"$k_y$")
        axs[ido, 1].text(-0.3, 0.92, f"({letters[ido*2+1]})", transform=axs[ido, 1].transAxes, fontsize=10)


    axs[-1, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[-1, 0].set_xlabel(r"$x$")
    axs[-1, 1].set_xticks([-100, -1, 1, 100])
    axs[-1, 1].set_xlabel(r"$k_x$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    cax1 = fig.add_axes([0.1, 1.01, 0.4, 0.02])
    cb = plt.colorbar(cs, cax=cax1, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$|\vec{u}| \quad \times 10^3$", labelpad=-40, rotation=0)

    cax2 = fig.add_axes([0.55, 1.01, 0.4, 0.02])
    cb = plt.colorbar(im, cax=cax2, orientation="horizontal", ticks=np.linspace(0, 4, 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\ln\big(1+|\mathcal{F}(|\mathbf{U}|)|\big)$", labelpad=-40, rotation=0)

    plt.savefig(f"figures/RA_response.pdf")
    plt.savefig(f"figures/RA_response.png", dpi=700)

    plt.close()


def plot_large_f_r():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(0, 0.25, ny)
    py_mask = np.logical_and(pys > 0, pys < 0.1)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/f_r_modes.npy")
    letters = [chr(97 + i) for i in range(len(omegas)*2)]
    fig, axs = plt.subplots(len(omegas),2, figsize=(6, 8.5), sharex=True, sharey=True)
    axs[0, 0].set_title("Forcing", fontsize=9)
    axs[0, 1].set_title("Response", fontsize=9)
    for ido, omega in tqdm(enumerate(omegas), total=len(omegas), desc="Plotting bigun"):
        mag = f_r_modes[0, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[0, 5]
        cs = axs[ido, 0].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("icefire", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 0].set_ylabel(r"$n$")
        axs[ido, 0].text(-0.3, 0.94, f"({letters[ido*2]})", transform=axs[ido, 0].transAxes, fontsize=10)
        axs[ido, 0].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 0].transAxes, fontsize=10)

        mag = f_r_modes[1, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[0, 5]
        cs = axs[ido, 1].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("icefire", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 1].text(-0.1, 0.94, f"({letters[ido*2+1]})", transform=axs[ido, 1].transAxes, fontsize=10)
        # axs[ido, 1].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 1].transAxes, fontsize=10)

    axs[-1, 0].set_xlabel(r"$x$")
    axs[-1, 1].set_xlabel(r"$x$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    cax1 = fig.add_axes([0.175, 1.01, 0.7, 0.02])
    cb = plt.colorbar(cs, cax=cax1, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$|\vec{u}| \quad \times 10^3$", labelpad=-37, rotation=0)

    plt.savefig(f"figures/RA_modes.pdf")
    # plt.savefig(f"figures/RA_modes.png", dpi=700)
    plt.close()


def plot_large_s_f_r():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(0, 0.25, ny)
    py_mask = np.logical_and(pys > 0, pys < 0.1)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/s_f_r_modes.npy")
    letters = [chr(97 + i) for i in range(len(omegas)*2)]
    fig, axs = plt.subplots(len(omegas),2, figsize=(6, 8.5), sharex=True, sharey=True)
    axs[0, 0].set_title("Forcing", fontsize=9)
    axs[0, 1].set_title("Response", fontsize=9)
    for ido, omega in tqdm(enumerate(omegas), total=7, desc="Plotting bigun"):
        mag = f_r_modes[0, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[-5, 5]
        cs = axs[ido, 0].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("seismic", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 0].set_ylabel(r"$n$")
        axs[ido, 0].text(-0.3, 0.94, f"({letters[ido*2]})", transform=axs[ido, 0].transAxes, fontsize=10)
        axs[ido, 0].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 0].transAxes, fontsize=10)

        mag = f_r_modes[1, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[-5, 5]
        cs = axs[ido, 1].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("seismic", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 1].text(-0.1, 0.94, f"({letters[ido*2+1]})", transform=axs[ido, 1].transAxes, fontsize=10)
        # axs[ido, 1].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 1].transAxes, fontsize=10)

    axs[-1, 0].set_xlabel(r"$x$")
    axs[-1, 1].set_xlabel(r"$x$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    cax1 = fig.add_axes([0.175, 1.01, 0.7, 0.02])
    cb = plt.colorbar(cs, cax=cax1, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\vec{u_s} \quad \times 10^3$", labelpad=-37, rotation=0)

    plt.savefig(f"figures/RA_s_modes.pdf")
    plt.savefig(f"figures/RA_s_modes.png", dpi=700)
    plt.close()


def plot_large_n_f_r(n=0):
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "sp")
    nx, ny, nt = np.load(f"data/0.001/0/unmasked/nxyt.npy")
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(0, 0.25, ny)
    py_mask = np.logical_and(pys > 0, pys < 0.1)
    omegas = smooth.peak_omegas
    f_r_modes = np.load(f"{os.getcwd()}/data/0.001/0/unmasked/n_f_r_mode{n}.npy")
    letters = [chr(97 + i) for i in range(len(omegas)*2)]
    fig, axs = plt.subplots(len(omegas),2, figsize=(5.8, 8.5), sharex=True, sharey=True)
    axs[0, 0].set_title("Forcing", fontsize=9)
    axs[0, 1].set_title("Response", fontsize=9)
    for ido, omega in tqdm(enumerate(omegas), total=7, desc="Plotting bigun"):
        mag = f_r_modes[0, ido, :, py_mask]
        nx, ny = mag.shape
        lim=[-5, 5]
        cs = axs[ido, 0].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("seismic", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 0].set_ylabel(r"$n$")
        axs[ido, 0].text(-0.3, 0.94, f"({letters[ido*2]})", transform=axs[ido, 0].transAxes, fontsize=10)
        axs[ido, 0].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 0].transAxes, fontsize=10)

        mag = f_r_modes[1, ido, :, py_mask]
        nx, ny = mag.shape
        cs = axs[ido, 1].imshow(
            mag*1000,
            extent=[0, 1, pys[py_mask].min(), pys[py_mask].max()],
            cmap=sns.color_palette("seismic", as_cmap=True),
            norm=TwoSlopeNorm(vmin=lim[0], vcenter=(lim[1]+lim[0])/2, vmax=lim[1]),
            origin="lower",
            aspect='auto',
        )
        axs[ido, 1].text(-0.1, 0.94, f"({letters[ido*2+1]})", transform=axs[ido, 1].transAxes, fontsize=10)
        # axs[ido, 1].text(0.1, 0.77, f"$f^*={omega/(2*np.pi):.2f}$", transform=axs[ido, 1].transAxes, fontsize=10)

    axs[-1, 0].set_xlabel(r"$x$")
    axs[-1, 1].set_xlabel(r"$x$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    cax1 = fig.add_axes([0.175, 1.01, 0.7, 0.02])
    cb = plt.colorbar(cs, cax=cax1, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    cb.ax.xaxis.tick_top()  # Move ticks to top
    cb.ax.xaxis.set_label_position('top')  # Move label to top
    cb.set_label(r"$\vec{u_n} \quad \times 10^3$", labelpad=-37, rotation=0)

    plt.savefig(f"figures/RA/norm_mode{n}.pdf")
    # plt.savefig(f"figures/RA_n_modes.png", dpi=700)
    plt.close()


if __name__ == "__main__":
    for i in range(4):
        save_f_r(i)
        plot_large_n_f_r(i)

    # plot_large_f_r()
    # plot_large_s_f_r()

