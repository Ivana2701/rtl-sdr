import os
import sys
import unittest

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from waterfall_app import FileIQSource, compute_power_db, MIN_NFFT


class TestFileIQSource(unittest.TestCase):
    def test_reads_and_wraps(self):
        sample_path = os.path.join(ROOT_DIR, "test", "sample_iq.csv")
        src = FileIQSource(sample_path, sample_rate_hz=2.4e6, center_freq_hz=94.9e6, fmt="csv")
        data = src.read_samples(64)
        self.assertEqual(data.shape[0], 64)
        self.assertTrue(np.iscomplexobj(data))


class TestComputePower(unittest.TestCase):
    def test_dynamic_range_clipping(self):
        sample_path = os.path.join(ROOT_DIR, "test", "sample_iq.csv")
        src = FileIQSource(sample_path, sample_rate_hz=2.4e6, center_freq_hz=94.9e6, fmt="csv")
        iq = src.read_samples(MIN_NFFT)
        window = np.hanning(MIN_NFFT).astype(np.float32)

        power_db, db_min, db_max = compute_power_db(iq, MIN_NFFT, window, 60.0, None)
        self.assertEqual(power_db.shape[0], MIN_NFFT)
        self.assertGreaterEqual(db_max - db_min, 60.0)
        self.assertLessEqual(float(power_db.max()), db_max + 1e-6)
        self.assertGreaterEqual(float(power_db.min()), db_min - 1e-6)


if __name__ == "__main__":
    unittest.main()
