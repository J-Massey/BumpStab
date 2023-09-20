import numpy as np


class DefKernel:
    def __init__(self,):
        self.gaussParams = [1, 0.5]  # sigma for Gaussian kernel
        self.polyParams = [1, 0, 3]  # coefficients for polynomial kernel
    
    def eval_kernel(self, X, Y):
        if np.isscalar(X) and np.isscalar(Y):
            return np.exp(-(X - Y)**2 / (2 * 0.5 ** 2)) + 1 + 0 * (X * Y) + 3 * (X * Y) ** 2
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        pairwise_dists = np.sum(X**2, axis=0)[:, np.newaxis] + np.sum(Y**2, axis=0) - 2 * np.dot(X.T, Y)
        gaussPart = np.exp(-pairwise_dists / (2 * 0.5 ** 2))
        polyPart = self.polyParams[0] + self.polyParams[1] * np.dot(X.T, Y) + self.polyParams[2] * (np.dot(X.T, Y) ** 2)
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
