# RTL-SDR Waterfall
-Standalone application to visualize RF spectrum in real-time using RTL-SDR hardware. 

## Features
 
• Read IQ (complex) samples directly from RTL-SDR dongle 
• Implement FFT-based spectrum analysis (minimum 1024-point FFT) 
• Display waterfall plot with time on Y-axis, frequency on X-axis, power in color scale 
• Update rate: minimum 10 Hz refresh 
• Configurable center frequency and sample rate 
• Dynamic range: display at least 60 dB Deliverables Source code, executable, screenshot of captured spectrum, brief technical report (max 2 pages).

## Requirements
- Windows 10/11
- Python 3.10+ recommended
- RTL-SDR drivers installed (Zadig/WinUSB)
- RTL-SDR DLL available on disk

## Setup
Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install numpy PySide6 pyrtlsdr
```

Point the app to your RTL-SDR DLL folder in [main.py](main.py) (see `rtl_dll_dir`).

## Usage
Run with RTL-SDR hardware:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Run with mock data (no hardware required):

```powershell
.\.venv\Scripts\python.exe .\main.py --mock
```

Run with an IQ file:

```powershell
.\.venv\Scripts\python.exe .\main.py --iq-file .\test\sample_iq.csv
```

## Notes
- If the app shows "Mock IQ", hardware init failed or `--mock` was used.
- Make sure the RTL-SDR DLL folder is reachable and the dongle is plugged in.
- If you see driver errors, reinstall WinUSB via Zadig and retry.
