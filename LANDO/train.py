import numpy as np
import time

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

def trainLANDO(X, Y, nu, kernel, online=0, xScl=1, backslash=1, psinv=0, displ=0):
    start_time = time.time()
    evalKernel = kernel[0]
    sX = xScl * X
    ktt = evalKernel(sX[:, 0], sX[:, 0])
    C = np.sqrt(ktt).reshape(1, 1)
    wr = 1
    if online:
        Pt = 1
        Wtilde = Y[:, 0] / ktt
    m = 1
    sXdic = sX[:, 0][:, np.newaxis]
    for t in range(1, X.shape[1]):
        sXt = sX[:, t]
        kTildeT = evalKernel(sXdic, sXt)
        pit = np.linalg.solve(C.T, np.linalg.solve(C, kTildeT))
        ktt = evalKernel(sXt, sXt)
        delta = ktt - kTildeT.T @ pit
        if abs(delta) > nu:
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
            if online:
                Pt = np.block([[Pt, np.zeros((m, 1))], [np.zeros((1, m)), 1]])
                YminWKD = (Y[:, t] - Wtilde @ kTildeT) / delta
                Wtilde = np.column_stack((Wtilde - YminWKD * pit.T, YminWKD))
            m += 1
        else:
            if online:
                ht = (pit.T @ Pt) / (1 + pit.T @ Pt @ pit)
                Pt = Pt - (Pt @ pit)[:, np.newaxis] @ ht[np.newaxis, :]
                Wtilde = Wtilde + ((Y[:, t] - Wtilde @ kTildeT) * ht) @ np.linalg.inv(C).T @ np.linalg.inv(C)
    Xdic = sXdic / xScl
    if not online:
        if backslash:
            Wtilde = np.linalg.lstsq(evalKernel(sXdic, sX), Y.T, rcond=None)[0].T
        elif psinv:
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


wtest  = np.load(f"{self.path}/{self.dom}_flucs.npy")

X = wtest[:, :-1]
Y = wtest[:, 1:]

k = [evalKernel]
model, Xdic, Wtilde, recErr = trainLANDO(X, Y, 0.01, k, displ=True)

model, Xdic.shape, Wtilde.shape, recErr