import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---- Qt settings for stability on Windows ----
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

# ---- RTL-SDR DLL path (local, avoids user-specific paths) ----
try:
    rtl_dll_dir = r"C:\Users\Ivana\Downloads\Release\x64"
    if os.path.isdir(rtl_dll_dir):
        os.add_dll_directory(rtl_dll_dir)
except Exception:
    pass

from waterfall_app import main


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")

