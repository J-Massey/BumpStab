import os
import numpy as np
import time
from tqdm import tqdm
from kernel import DefKernel


def evalKernel(X, Y):
    if np.isscalar(X) and np.isscalar(Y):
        return np.exp(-(X - Y)**2 / (2 * 0.5 ** 2)) + 1 + 0 * (X * Y) + 3 * (X * Y) ** 2
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    pairwise_dists = np.sum(X**2, axis=0)[:, np.newaxis] + np.sum(Y**2, axis=0) - 2 * np.dot(X.T, Y)
    gaussPart = np.exp(-pairwise_dists / (2 * 0.5 ** 2))
    polyPart = 1 + 0 * np.dot(X.T, Y) + 3 * (np.dot(X.T, Y) ** 2)
    return gaussPart + polyPart


def composite_kernel(X, Y, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, sigma=0.5, c=1, d=2, f=1.0):
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    
    # Gaussian (RBF) Part
    pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    gaussPart = np.exp(-pairwise_dists / (2 * sigma ** 2))
    
    # Polynomial Part
    polyPart = (np.dot(X, Y.T) + c) ** d
    
    # Linear Part
    linearPart = np.dot(X, Y.T)
    
    # Sinusoidal Part
    sinusoidalPart = np.cos(2 * np.pi * f * np.dot(X, Y.T))
    
    # Composite Kernel
    return alpha * gaussPart + beta * polyPart + gamma * linearPart + delta * sinusoidalPart


def trainLANDO(X, Y, nu, kernel, xScl=1, displ=0):
    start_time = time.time()
    evalKernel = kernel[0]
    sX = xScl * X
    ktt = evalKernel(sX[:, 0], sX[:, 0])
    C = np.sqrt(ktt).reshape(1, 1)
    wr = 1
    m = 1
    sXdic = sX[:, 0][:, np.newaxis]
    for t in tqdm(range(1, X.shape[1])):
        sXt = sX[:, t]
        kTildeT = evalKernel(sXdic, sXt)
        pit = np.linalg.solve(C.T, np.linalg.solve(C, kTildeT))
        ktt = evalKernel(sXt, sXt)
        # delta = ktt - kTildeT.T @ pit
        sXdic = np.column_stack((sXdic, sXt))
        C12 = kTildeT.T @ np.linalg.inv(C.T)
        if ktt <= np.linalg.norm(C12) ** 2 and wr:
            print("The Cholesky factor is ill-conditioned.")
            wr = 0
        C = np.vstack((C, C12))
        new_row = np.zeros((1, m))
        new_val = np.abs(np.sqrt(ktt - np.linalg.norm(C12) ** 2))
        new_row = np.append(new_row, new_val)
        C = np.hstack((C, new_row[:, np.newaxis]))
        m += 1
    
    Xdic = sXdic / xScl

    Wtilde = Y @ np.linalg.pinv(evalKernel(sXdic, sX), 1e-10)
    model = lambda x: Wtilde @ evalKernel(sXdic, xScl * x)
    recErr = np.mean(np.linalg.norm(Y - model(X), axis=0) / np.linalg.norm(Y, axis=0))

    if displ:
        print(f"------- LANDO completed -------")
        print(f"Training error:     {recErr * 100:.3f}%")
        print(f"Time taken:         {time.time() - start_time:.2f} secs")
        print(f"Number of samples:  {X.shape[1]}")
        print(f"Size of dictionary: {m}")
    return model, Xdic, Wtilde, recErr

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from scipy.linalg import svd

    plt.style.use(["science"])
    plt.rcParams["font.size"] = "10.5"
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{txfonts}')

    dom="body"
    case = "test/up"
    wtest  = np.load(f"{os.getcwd()}/data/{case}/data/body_flucs.npy")

    X = wtest[:, :-1]
    Y = wtest[:, 1:]

    kern = DefKernel('gaussian', [1, 0.5], 'polynomial', [1, 0, 3])
    k = [kern.evalKernel]

    evalKernel(X[:, 0], X[:, 0])
    
    model, Xdic, Wtilde, recErr = trainLANDO(X[:, ::10], Y[:, ::10], 0.01, k, displ=True)

    Xdic.shape

    nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/{dom}_nxyt.npy")
    nt = Wtilde.shape[-1]
    pxs = np.linspace(0, 1, nx)
    pys = np.linspace(-0.25, 0.25, ny)

    Wtilde.resize(3, nx, ny, nt)

    for n in tqdm(range(10)):
        fig, ax = plt.subplots(figsize=(5, 3))
        qi = Wtilde[2, :, :, n].real
        lim = np.std(qi)*4
        levels = np.linspace(-lim, lim, 44)
        _cmap = sns.color_palette("seismic", as_cmap=True)

        cs = ax.contourf(
            pxs,
            pys,
            qi.T,
            levels=levels,
            vmin=-lim,
            vmax=lim,
            # norm=norm,
            cmap=_cmap,
            extend="both",
            # alpha=0.7,
        )
        ax.set_aspect(1)
        plt.savefig(f"low-mode/dmd-comparison/lando/{n}.png", dpi=300)
        plt.close()