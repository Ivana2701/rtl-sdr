import os
import sys
from rtlsdr import RtlSdr

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    dll_dir1 = root_dir
    dll_dir2 = os.path.join(root_dir, "rtlsdr_dll")

    if os.path.isdir(dll_dir1):
        os.add_dll_directory(dll_dir1)
    if os.path.isdir(dll_dir2):
        os.add_dll_directory(dll_dir2)

    sdr = RtlSdr()
    try:
        # configure
        sdr.sample_rate = 2.048e6
        sdr.center_freq = 87.6e6
        sdr.gain = 35

        print("OK - device opened")
        print("Center freq:", sdr.get_center_freq())
        print("Sample rate:", sdr.get_sample_rate())
        print("Gain:", sdr.get_gain())

        # real test: read samples
        x = sdr.read_samples(2048)
        print("Read samples:", len(x), "complex")

    finally:
        sdr.close()
        print("Closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("SDR test failed:", repr(e))
        sys.exit(1)