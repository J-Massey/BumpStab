import unittest
import numpy as np
import os
import h5py
from preprocessing.load_data import LoadData

class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.data_loader = LoadData("tests/testdata.hdf5")

    def test_data_loading(self):
        data = self.data_loader._load_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, np.ndarray)

    def test_unwarping(self):
        warped_data = self.data_loader.unwarp_velocity_field()
        self.assertIsNotNone(warped_data)
        self.assertIsInstance(warped_data, np.ndarray)

if __name__ == '__main__':
    unittest.main()
