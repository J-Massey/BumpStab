import numpy as np

def defineKernel(*args):
    # Initialize default parameters
    gaussParams = [0, float('nan')]
    polyParams = float('nan')
    
    # Parse arguments
    j = 0
    while j < len(args):
        v = args[j]
        j += 1
        if v == 'gaussian':
            gaussParams = args[j]
            j += 1
        elif v == 'polynomial':
            polyParams = args[j]
            j += 1
        else:
            raise ValueError("Unrecognized input")
    
    # Define the evaluation functions
    def evalKernel(x, y, gaussParams, polyParams):
        polyPart = 0
        if not np.isnan(polyParams).all():
            XY = x.T @ y
            for jj in range(len(polyParams)):
                if polyParams[jj] != 0:
                    polyPart += polyParams[jj] * (XY ** (jj))
        if np.isnan(gaussParams[1]):
            gaussPart = 0
        else:
            gaussPart = gaussParams[0] * gaussKernel(x, y, gaussParams[1])
        return polyPart + gaussPart
    
    def evalKernelDeriv(x, y, gaussParams, polyParams):
        polyPart = 0
        if not np.isnan(polyParams).all():
            XY = x.T @ y
            for jj in range(1, len(polyParams)):
                if polyParams[jj] != 0:
                    polyPart += (jj) * polyParams[jj] * (XY ** (jj - 1)) * x.T
        if np.isnan(gaussParams[1]):
            gaussPart = 0
        else:
            gaussPart = gaussParams[0] / gaussParams[1] ** 2 * gaussKernel(x, y, gaussParams[1]) * (x - y).T
        return polyPart + gaussPart
    
    def gaussKernel(X1, X2, Sigma):
        if np.isnan(Sigma):
            return 0
        else:
            dim1 = X1.shape[1]
            dim2 = X2.shape[1]

            norms1 = np.sum(X1 ** 2, axis=1)
            norms2 = np.sum(X2 ** 2, axis=1)

            mat1 = np.tile(norms1[:, np.newaxis], (1, dim2))
            mat2 = np.tile(norms2, (dim1, 1))

            distmat = mat1 + mat2 - 2 * X1.T @ X2
            return np.exp(-distmat / (2 * Sigma ** 2))
    
    k = [lambda x, y: evalKernel(x, y, gaussParams, polyParams),
         lambda x, y: evalKernelDeriv(x, y, gaussParams, polyParams)]
    
    return k

# Test the function
k = defineKernel('gaussian', [1, 0.5], 'polynomial', [1, 0, 3])
kernel_value = k[0](np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
kernel_gradient = k[1](np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

kernel_value, kernel_gradient
