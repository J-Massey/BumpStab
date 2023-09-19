import numpy as np


class DefKernel:
    def __init__(self, *args):
        # Initialize default parameters
        self.gaussParams = [0, float('nan')]
        self.polyParams = float('nan')
        
        # Parse arguments
        j = 0
        while j < len(args):
            v = args[j]
            j += 1
            if v == 'gaussian':
                self.gaussParams = args[j]
                j += 1
            elif v == 'polynomial':
                self.polyParams = args[j]
                j += 1
            else:
                raise ValueError("Unrecognized input")

    def evalKernel(self, x, y):
        polyPart = 0
        if not np.isnan(self.polyParams).all():
            XY = x.T @ y
            for jj in range(len(self.polyParams)):
                if self.polyParams[jj] != 0:
                    polyPart += self.polyParams[jj] * (XY ** (jj))
        if np.isnan(self.gaussParams[1]):
            gaussPart = 0
        else:
            gaussPart = self.gaussParams[0] * self.gaussKernel(x, y)
        return polyPart + gaussPart

    def evalKernelDeriv(self, x, y):
        polyPart = 0
        if not np.isnan(self.polyParams).all():
            XY = x.T @ y
            for jj in range(1, len(self.polyParams)):
                if self.polyParams[jj] != 0:
                    polyPart += (jj) * self.polyParams[jj] * (XY ** (jj - 1)) * x.T
        if np.isnan(self.gaussParams[1]):
            gaussPart = 0
        else:
            gaussPart = self.gaussParams[0] / self.gaussParams[1] ** 2 * self.gaussKernel(x, y) * (x - y).T
        return polyPart + gaussPart
    
    def gaussKernel(self, X1, X2):
        if np.isnan(self.gaussParams[1]):
            return 0
        else:
            dim1 = X1.shape[1]
            dim2 = X2.shape[1]

            norms1 = np.sum(X1 ** 2, axis=1)
            norms2 = np.sum(X2 ** 2, axis=1)

            mat1 = np.tile(norms1[:, np.newaxis], (1, dim2))
            mat2 = np.tile(norms2, (dim1, 1))

            distmat = mat1 + mat2 - 2 * X1.T @ X2
            return np.exp(-distmat / (2 * self.gaussParams[1] ** 2))

# Test the class
import numpy as np

k = DefKernel('gaussian', [1, 0.5], 'polynomial', [1, 0, 3])
kernel_value = k.evalKernel(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
kernel_gradient = k.evalKernelDeriv(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

kernel_value, kernel_gradient
