import unittest
import os
import read_bruker_sdt as rb
import numpy as np


class TestReadBrukerSdt(unittest.TestCase):
    def setUp(self):
        ## 2-channel test data from Swab-Ult 2025
        self.test_sdt_file = r"C:\dev\test_data\Images FLIM i3S\coumarin6Image-03202025-1307-001\LifetimeData_Cycle00001_000001.sdt"

        if not os.path.exists(self.test_sdt_file):
            raise FileNotFoundError(f"Test SDT file not found: {self.test_sdt_file}")

    def test_read_sdt150(self):
        data = rb.read_sdt150(self.test_sdt_file)
        self.assertIsInstance(data, np.ndarray)
        data = rb.read_bruker_sdt(self.test_sdt_file)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertTrue(hasattr(data[0], "shape"))
        self.assertGreater(len(data[0].shape), 0)


if __name__ == "__main__":
    unittest.main()
