import unittest
import os
import read_bruker_sdt as rb
import numpy as np
import pathlib

class TestReadBrukerSdt(unittest.TestCase):
    def setUp(self):

        self.test_sdt_files = list(pathlib.Path(r'C:\dev\test_data').rglob('*.sdt'))
        
        if len(self.test_sdt_files)<1:
            raise FileNotFoundError(f"Test SDT file not found: {self.test_sdt_file}")

    def test_read_sdt150(self):
        for test_sdt_file in self.test_sdt_files:
            data = rb.read_sdt150(test_sdt_file)
            self.assertIsInstance(data, np.ndarray)        
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            self.assertTrue(hasattr(data[0], "shape"))
            self.assertGreater(len(data[0].shape), 0)

if __name__ == "__main__":
    unittest.main()
