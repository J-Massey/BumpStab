import numpy as np

class LoadData:
    def __init__(self, path):
        self.path = path

    def load(self):
        data = np.load(self.path)
        data = np.einsum("ijk -> kji", data)
        return data #[::2, ::2, :]