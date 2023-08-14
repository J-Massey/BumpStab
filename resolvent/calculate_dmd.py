from scipy.linalg import svd, cholesky
import numpy as np


def fbDMD(flucs, Ub, Sigmab, VTb, Uf, Sigmaf, VTf, nx, ny, nt, r=None):
    fluc1 = flucs[:,:-1]
    fluc2 = flucs[:, 1:]
    dt = 14/nt
    if r is None:
        r = nt-1
    # Sigma_plot(Sigma)
    U_r = Ub[:,:r]
    Sigmar = np.diag(Sigmab[:r])
    VT_r = VTb[:r,:]
    Atildeb = np.linalg.solve(Sigmar.T,(U_r.T @ fluc1 @ VT_r.T).T).T


    U_r = Uf[:, :r]
    S_r = np.diag(Sigmaf[:r])
    VT_r = VTf[:r, :]
    Atildef = np.linalg.solve(S_r.T,(U_r.T @ fluc2 @ VT_r.T).T).T # Step 2 - Find the linear operator using psuedo inverse

    # Find the linear operator
    A_tilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
    rho, W = np.linalg.eig(A_tilde)

    # Find the eigenfunction from spectral expansion
    Lambda = np.log(rho)/dt

    # Find the DMD modes
    V_r = np.dot(np.dot(fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

    # Find the hermatian adjoint of the 
    V_r_star_Q = V_r.conj().T
    V_r_star_Q_V_r = np.dot(V_r_star_Q, V_r)
    # Cholesky factorization
    F_tilde = cholesky(V_r_star_Q_V_r)



if __name__ == "__main__":
    case = "test"
    doms = ["body", "wake"]
    for dom in doms:
        flucs = np.load(f"data/{case}/data/{dom}_flucs.npy")
        Ub, Sigmab, VTb, Uf, Sigmaf, VTf = np.load(f"data/{case}/data/{dom}_svd.npz", allow_pickle=True)
