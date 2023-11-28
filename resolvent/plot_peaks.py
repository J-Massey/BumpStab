import numpy as np
from scipy.linalg import cholesky, svd, inv

import matplotlib.pyplot as plt
import scienceplots
import sys
import os
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

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
        self.Lambda = np.load(f"{self.path}/{self.dom}_Lambda100.npy")
        self.V_r = np.load(f"{self.path}/{self.dom}_V_r100.npy")
        self.nx, self.ny, self.nt = np.load(f"{self.path}/{self.dom}_nxyt.npy")
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


def plot_field(qi, pxs, pys, path, _cmap="seismic", lim=None):
    # Test plot
    fig, ax = plt.subplots(figsize=(5, 3))
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.imshow(
        qi,
        extent=[0, 1, -0.25, 0.25],
        cmap=_cmap,
        norm=TwoSlopeNorm(vmin=lim[0], vcenter=0, vmax=lim[1]),
    )
    # cbar on top of the plot
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("top", size="7%", pad=0.2)
    # fig.add_axes(cax)
    # cb = plt.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.linspace(lim[0], lim[1], 5))
    # cb.set_label(r"$p$", labelpad=-35, rotation=0)
    ax.set_aspect(1)
    plt.savefig(path, dpi=600)
    plt.close()


def plot_vort():
    smooth = PlotPeaks(f"{os.getcwd()}/data/0.001/0/unmasked", "body")
    omegas = smooth.peak_omegas
    for omega in omegas:
        Psi, Sigma, Phi = svd(smooth.F_tilde@inv((-1j*omega)*np.eye(smooth.Lambda.shape[0])-np.diag(smooth.Lambda))@inv(smooth.F_tilde))
        for i in range(len(Sigma)):
            Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
            Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
            Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

        forcing = (smooth.V_r @ inv(smooth.F_tilde)@Psi).reshape(2, smooth.nx, smooth.ny, len(Sigma))
        response = (smooth.V_r @ inv(smooth.F_tilde)@Phi).reshape(2, smooth.nx, smooth.ny, len(Sigma))

        pxs = np.linspace(0, 1, smooth.nx)
        pys = np.linspace(-0.25, 0.25, smooth.ny)
        field = forcing[0, :, :, 0].real
        # angle = np.angle(field.astype(np.complex128))


        # vort = np.gradient(response[1, :, :, 0].real, pxs, axis=0) - np.gradient(response[0, :, :, 0].real, pys, axis=1)
        mag = np.sqrt(forcing[1, :, :, 0].real**2 +  forcing[0, :, :, 0].real**2)
        lim = np.std(mag)*8
        print(lim)
        plot_field(mag.T, pxs, pys, f"figures/forcing-modes/forcing_{omega/(np.pi):.2f}.png", lim=[-lim, lim], _cmap="seismic")
        mag = np.sqrt(response[1, :, :, 0].real**2 +  response[0, :, :, 0].real**2)
        lim = np.std(mag)*8
        print(lim)
        plot_field(mag.T, pxs, pys, f"figures/response-modes/response_{omega/(np.pi):.2f}.png", lim=[-lim, lim], _cmap="seismic")


if __name__ == "__main__":
    plot_vort()

#         # Perform 2D FFT to transform to frequency (wavenumber) domain

#         field_fft = np.fft.fft2(field)
#         # Shift zero frequency component to center
#         field_fft_shifted = np.fft.fftshift(field_fft)
#         # Compute amplitude spectrum
#         amplitude_spectrum = np.abs(field_fft_shifted)
#         # Find dominant wavenumbers
#         dominant_wavenumber_indices = np.unravel_index(np.argmax(amplitude_spectrum), amplitude_spectrum.shape)
#         dominant_wavenumbers = np.array(dominant_wavenumber_indices) - np.array([case.nx//2, case.ny//2])
#         # Plot amplitude spectrum for visualization
#         plt.imshow(np.log1p(amplitude_spectrum), extent=[-case.ny//2, case.ny//2, -case.nx//2, case.nx//2], cmap='gray')
#         plt.xlabel('kx')
#         plt.ylabel('ky')
#         plt.savefig(f"figures/mode-comparison/kx_{case_label[idx]}_response_{omega/(2*np.pi):.2f}.png", dpi=700)
#         plt.close()

#         dominant_wavenumbers

