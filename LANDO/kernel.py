import numpy as np


class DefKernel:
    def __init__(self,):
        self.gaussParams = [1, 0.5]  # sigma for Gaussian kernel
        self.polyParams = [1, 0, 3]  # coefficients for polynomial kernel
    
    def eval_kernel(self, X, Y):
        if np.isscalar(X) and np.isscalar(Y):
            return np.exp(-(X - Y)**2 / (2 * self.gaussParams[1] ** 2)) + sum(
                self.polyParams[jj] * (X * Y) ** jj for jj in range(len(self.polyParams))
            )
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        gaussPart = np.exp(-pairwise_dists / (2 * self.gaussParams[1] ** 2))
        polyPart = sum(
            self.polyParams[jj] * (np.dot(X.T, Y) ** jj)
            for jj in range(len(self.polyParams))
        )
        return gaussPart + polyPart
    
    def evalKernelDeriv(self, X, Y):
        # Gaussian part
        pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        gaussPart = -(pairwise_dists / (self.gaussParams[1] ** 4)) * np.exp(-pairwise_dists / (2 * self.gaussParams[1] ** 2))
        
        # Polynomial part
        polyPart = 0
        for jj in range(1, len(self.polyParams)):
            polyPart += jj * self.polyParams[jj] * np.dot(X.T, Y) ** (jj - 1)
        
        return gaussPart + polyPart

k = DefKernel()
randtest = np.random.rand(100,20)
kernel_value = k.eval_kernel(randtest, randtest)

kernel_value
