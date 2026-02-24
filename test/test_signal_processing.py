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
    def setUp(self):
        self.sample_path = os.path.join(ROOT_DIR, "test", "sample_iq.csv")
        if not os.path.exists(self.sample_path):
            self.skipTest(f"Missing test file: {self.sample_path}")

    def test_reads_complex(self):
        src = FileIQSource(self.sample_path, sample_rate_hz=2.4e6, center_freq_hz=94.9e6, fmt="csv")
        data = src.read_samples(64)
        self.assertEqual(data.shape, (64,))
        self.assertTrue(np.iscomplexobj(data))

    def test_wraps_when_reading_past_end(self):
        # Read a chunk larger than the file length to force wrap/repeat behavior
        src = FileIQSource(self.sample_path, sample_rate_hz=2.4e6, center_freq_hz=94.9e6, fmt="csv")
        # Read twice MIN_NFFT to ensure wrap happens for most small test files
        data = src.read_samples(MIN_NFFT * 2)
        self.assertEqual(data.shape, (MIN_NFFT * 2,))
        self.assertTrue(np.iscomplexobj(data))
        #sanity check: not all zeros
        self.assertGreater(float(np.abs(data).mean()), 0.0)


class TestComputePower(unittest.TestCase):
    def setUp(self):
        self.sample_path = os.path.join(ROOT_DIR, "test", "sample_iq.csv")
        if not os.path.exists(self.sample_path):
            self.skipTest(f"Missing test file: {self.sample_path}")

    def test_dynamic_range_clipping_exact(self):
        src = FileIQSource(self.sample_path, sample_rate_hz=2.4e6, center_freq_hz=94.9e6, fmt="csv")
        iq = src.read_samples(MIN_NFFT)
        window = np.hanning(MIN_NFFT).astype(np.float32)

        power_db, db_min, db_max = compute_power_db(
            iq, MIN_NFFT, window, dynamic_range_db=60.0, db_max=None, clip=True
        )

        self.assertEqual(power_db.shape, (MIN_NFFT,))
        self.assertAlmostEqual(db_max - db_min, 60.0, places=5)
        self.assertLessEqual(float(power_db.max()), db_max + 1e-6)
        self.assertGreaterEqual(float(power_db.min()), db_min - 1e-6)

    def test_fft_shift_symmetry_for_tone(self):
        # Synthetic IQ: tone at +200 kHz (should appear off-center after fftshift)
        fs = 2.4e6
        n = MIN_NFFT
        f_tone = 200e3
        t = np.arange(n) / fs
        iq = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)

        window = np.hanning(n).astype(np.float32)
        power_db, _, _ = compute_power_db(iq, n, window, dynamic_range_db=80.0, db_max=None, clip=False)

        peak_bin = int(np.argmax(power_db))
        # After fftshift, DC is at n//2, so +200kHz should be on the right side
        self.assertGreater(peak_bin, n // 2)


if __name__ == "__main__":
    unittest.main()