import numpy as np

import os
case = "test/up"
# case = "0.001/16"

data = np.load(f"{os.getcwd()}/data/{case}/data/body_flucs.npy")
nx, ny, nt = np.load(f"{os.getcwd()}/data/{case}/data/body_nxyt.npy")
data.resize((3, nx, ny, nt))
body_p = data[2, :, :, :]

np.save(f"{os.getcwd()}/data/{case}/data/body_p.npy", body_p)
body_p.shape
