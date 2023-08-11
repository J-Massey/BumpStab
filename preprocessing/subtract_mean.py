from load_data import LoadData

class SubtractMean(LoadData):
    def __init__(self, path, T=1):
        super(SubtractMean, self).__init__(path)
        self.T = T
