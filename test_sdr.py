import os

# tell Windows where to look for DLLs
root_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(root_dir)
os.add_dll_directory(os.path.join(root_dir, "rtlsdr_dll"))

from rtlsdr import RtlSdr

sdr = RtlSdr()
print("OK", sdr.get_center_freq())
sdr.close()
