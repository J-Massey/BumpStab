import unittest
import numpy as np
from preprocessing.load_data import LoadData

class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.data_loader = LoadData("tests/testdata.npy")

    def test_data_loading(self):
        data = self.data_loader.flat_subdomain('wake')
        self.assertIsNotNone(data)
        self.assertIsInstance(data, np.ndarray)

    def test_unwarping(self):
        warped_data = self.data_loader.flat_subdomain('body')
        self.assertIsNotNone(warped_data)
        self.assertIsInstance(warped_data, np.ndarray)

if __name__ == '__main__':
    unittest.main()
