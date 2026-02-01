import os, sys

# tell Windows where to look for DLLs
os.add_dll_directory(os.path.join(os.getcwd(), ".venv", "Scripts"))

from rtlsdr import RtlSdr

sdr = RtlSdr()
print("OK", sdr.get_center_freq())
sdr.close()
