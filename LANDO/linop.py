import numpy as np
import time
from scipy.linalg import svd
from numpy.linalg import pinv, eig
from tqdm import tqdm


def linopLANDO(Xdic, Wtilde, kernel, xBar=None, nModes=None, xScl=1, nargout=2):
    nx = Xdic.shape[0]
    
    if xBar is None:
        xBar = np.zeros((nx, 1))
    if nModes is None:
        nModes = nx

    sXdic = xScl * Xdic
    sxBar = xScl * xBar
    evalKernelDeriv = kernel[1]

    # Project model onto principal components of the kernel gradient
    Ux, Sx, Vx = svd((evalKernelDeriv(sXdic, sxBar) * xScl).T, full_matrices=False)
    nModes = min(nModes, Ux.shape[1])
    Ux = Ux[:, :nModes]
    Sx = Sx[:nModes]
    Vx = Vx[:, :nModes]

    LTilde = (Ux.T @ Wtilde) @ (evalKernelDeriv(sXdic, sxBar) * xScl) @ Ux

    # Perform eigendecomposition of the reduced operator
    eVals, PsiHat = eig(LTilde)
    idx = np.argsort(np.abs(eVals))[::-1]
    eVals = eVals[idx]
    PsiHat = PsiHat[:, idx]

    # Project eigenvectors back onto the full space
    eVecs = Wtilde @ Vx @ np.diag(Sx) @ PsiHat / eVals

    # Output the full linear operator if requested
    linop = None
    if nargout > 2:
        linop = Wtilde @ evalKernelDeriv(sXdic, sxBar) * xScl

    return eVals, eVecs, linop

# Test the function again
eVals, eVecs, _ = linopLANDO(Xdic, Wtilde, [test_evalKernel, test_evalKernelDeriv])

eVals, eVecs
