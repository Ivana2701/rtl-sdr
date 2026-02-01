import os
import sys

# ---- Qt plugin paths (fix "Could not load the Qt platform plugin windows") ----
venv_root = os.path.dirname(os.path.dirname(sys.executable))  # ...\.venv
qt_plugins = os.path.join(venv_root, "Lib", "site-packages", "PyQt5", "Qt5", "plugins")
os.environ["QT_PLUGIN_PATH"] = qt_plugins
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugins, "platforms")

# ---- RTL-SDR DLL path (isolated to avoid conflicts with Qt) ----
rtlsdr_dll_dir = os.path.join(os.getcwd(), "rtlsdr_dll")
os.add_dll_directory(rtlsdr_dll_dir)

from rtlsdr import RtlSdr
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np



class WaterfallApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTL-SDR Real-Time Waterfall / Spectrogram")

        # ---------- Defaults (good for FM broadcast) ----------
        self.center_freq_hz = 94.9e6
        self.sample_rate_hz = 2.4e6
        self.gain_db = "auto"     # try 30-40 if auto looks weak
        self.nfft = 2048          # >= 1024 requirement
        self.fps = 15             # >= 10 Hz requirement
        self.waterfall_rows = 300
        self.dynamic_range_db = 60.0  # >= 60 dB requirement

        # ---------- SDR ----------
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rate_hz
        self.sdr.center_freq = self.center_freq_hz
        self.sdr.gain = self.gain_db

        # Window for FFT
        self.window = np.hanning(self.nfft).astype(np.float32)

        # Waterfall buffer (rows x bins)
        self.waterfall = np.full((self.waterfall_rows, self.nfft), -120.0, dtype=np.float32)

        # For stable scaling (optional smoothing)
        self._db_max_smooth = None

        # ---------- UI ----------
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        # Center frequency input (MHz)
        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(1.0, 2000.0)
        self.freq_input.setDecimals(6)
        self.freq_input.setValue(self.center_freq_hz / 1e6)
        self.freq_input.setSuffix(" MHz")

        # Sample rate input (MS/s)
        self.sr_input = QtWidgets.QDoubleSpinBox()
        self.sr_input.setRange(0.25, 3.2)
        self.sr_input.setDecimals(3)
        self.sr_input.setValue(self.sample_rate_hz / 1e6)
        self.sr_input.setSuffix(" MS/s")

        # Gain input
        self.gain_input = QtWidgets.QLineEdit(str(self.gain_db))
        self.gain_input.setFixedWidth(80)
        self.gain_input.setToolTip('Use "auto" or a number like 35')

        # FFT size
        self.fft_input = QtWidgets.QComboBox()
        self.fft_input.addItems(["1024", "2048", "4096", "8192"])
        self.fft_input.setCurrentText(str(self.nfft))

        # FPS
        self.fps_input = QtWidgets.QSpinBox()
        self.fps_input.setRange(10, 60)
        self.fps_input.setValue(int(self.fps))

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)

        controls.addWidget(QtWidgets.QLabel("Center:"))
        controls.addWidget(self.freq_input)
        controls.addWidget(QtWidgets.QLabel("Sample rate:"))
        controls.addWidget(self.sr_input)
        controls.addWidget(QtWidgets.QLabel("Gain:"))
        controls.addWidget(self.gain_input)
        controls.addWidget(QtWidgets.QLabel("FFT:"))
        controls.addWidget(self.fft_input)
        controls.addWidget(QtWidgets.QLabel("FPS:"))
        controls.addWidget(self.fps_input)
        controls.addWidget(self.apply_btn)
        controls.addStretch(1)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency (MHz)")
        self.plot.setLabel("left", "Time (newest at bottom)")
        self.plot.setMouseEnabled(x=True, y=False)
        layout.addWidget(self.plot)

        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # Nice colormap
        cmap = pg.colormap.get("viridis")
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        # Frequency axis mapping
        self.update_freq_axis()

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_once)
        self.timer.start(int(1000.0 / self.fps))

        self._closing = False

    def update_freq_axis(self):
        # Frequency bins in MHz centered around sdr.center_freq
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1.0 / self.sdr.sample_rate))
        freqs_mhz = (freqs + self.sdr.center_freq) / 1e6
        self.freqs_mhz = freqs_mhz

        x_min = float(freqs_mhz[0])
        x_max = float(freqs_mhz[-1])

        # Set image rectangle so x-axis corresponds to frequency span
        self.img.setRect(pg.QtCore.QRectF(x_min, 0, (x_max - x_min), self.waterfall_rows))

    def apply_settings(self):
        # Update SDR parameters
        self.sdr.center_freq = float(self.freq_input.value()) * 1e6
        self.sdr.sample_rate = float(self.sr_input.value()) * 1e6

        # Gain: "auto" or number
        gain_txt = self.gain_input.text().strip().lower()
        if gain_txt == "auto":
            self.sdr.gain = "auto"
        else:
            try:
                self.sdr.gain = float(gain_txt)
            except ValueError:
                # if invalid, keep previous
                pass

        # FFT size
        new_nfft = int(self.fft_input.currentText())
        if new_nfft != self.nfft:
            self.nfft = new_nfft
            self.window = np.hanning(self.nfft).astype(np.float32)
            self.waterfall = np.full((self.waterfall_rows, self.nfft), -120.0, dtype=np.float32)
            self._db_max_smooth = None

        # FPS
        self.fps = int(self.fps_input.value())
        self.timer.start(int(1000.0 / self.fps))

        # Update axis mapping
        self.update_freq_axis()

    def update_once(self):
        if self._closing:
            return

        # Read IQ samples
        iq = self.sdr.read_samples(self.nfft).astype(np.complex64)

        # Remove DC offset a bit (helps with center spike)
        iq = iq - np.mean(iq)

        # Windowed FFT
        x = iq * self.window
        X = np.fft.fftshift(np.fft.fft(x, n=self.nfft))

        # Power in dB
        mag = np.abs(X)
        power_db = 20.0 * np.log10(mag + 1e-12)

        # Choose a robust "top" and enforce 60 dB range
        new_db_max = float(np.percentile(power_db, 99))

        if self._db_max_smooth is None:
            self._db_max_smooth = new_db_max
        else:
            # smooth to avoid pumping colors
            self._db_max_smooth = 0.9 * self._db_max_smooth + 0.1 * new_db_max

        db_max = self._db_max_smooth
        db_min = db_max - self.dynamic_range_db

        power_db = np.clip(power_db, db_min, db_max).astype(np.float32)

        # Update waterfall (scroll up, newest at bottom)
        self.waterfall[:-1, :] = self.waterfall[1:, :]
        self.waterfall[-1, :] = power_db

        # Show image
        self.img.setImage(self.waterfall, autoLevels=False)

    def closeEvent(self, event):
        self._closing = True
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.sdr.close()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = WaterfallApp()
    w.resize(1200, 650)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

