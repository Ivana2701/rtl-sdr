import argparse
import faulthandler
import logging
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("QT_DWRITE_NO_DIRECTWRITE", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
os.environ.setdefault("QT_SCALE_FACTOR", "1")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


MIN_NFFT = 1024
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def make_colormap() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    r = np.clip(1.6 * t, 0.0, 1.0)
    g = np.clip(1.6 * (1.0 - np.abs(t - 0.5) * 2.0), 0.0, 1.0)
    b = np.clip(1.6 * (1.0 - t), 0.0, 1.0)
    cmap = np.stack([r, g, b], axis=1)
    return (cmap * 255).astype(np.uint8)


class AxisLabel(QtWidgets.QWidget):
    def __init__(self, text: str, vertical: bool = False):
        super().__init__()
        self._text = text
        self._vertical = vertical
        if self._vertical:
            self.setFixedWidth(24)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setPen(self.palette().color(QtGui.QPalette.WindowText))

        if self._vertical:
            painter.translate(0, self.height())
            painter.rotate(-90)
            rect = QtCore.QRect(0, 0, self.height(), self.width())
        else:
            rect = self.rect()

        painter.drawText(rect, QtCore.Qt.AlignCenter, self._text)


class AxisScale(QtWidgets.QWidget):
    def __init__(self, orientation: str, unit: str = "", tick_count: int = 5):
        super().__init__()
        self._orientation = orientation
        self._unit = unit
        self._tick_count = max(2, int(tick_count))
        self._min_val = 0.0
        self._max_val = 1.0

        if self._orientation == "vertical":
            self.setFixedWidth(56)
        else:
            self.setFixedHeight(28)

    def set_range(self, min_val: float, max_val: float) -> None:
        self._min_val = float(min_val)
        self._max_val = float(max_val)
        self.update()

    def _format_value(self, value: float) -> str:
        if self._unit == "MHz":
            return f"{value:.3f}"
        if self._unit == "s":
            return f"{value:.1f}"
        return f"{value:.2f}"

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setPen(self.palette().color(QtGui.QPalette.WindowText))

        span = self._max_val - self._min_val
        if span <= 0.0:
            return

        if self._orientation == "vertical":
            x_axis = self.width() - 1
            painter.drawLine(x_axis, 0, x_axis, self.height())
            for i in range(self._tick_count):
                t = i / (self._tick_count - 1)
                y = int(t * (self.height() - 1))
                value = self._min_val + t * span
                painter.drawLine(x_axis - 6, y, x_axis, y)
                label = self._format_value(value)
                painter.drawText(0, y - 8, x_axis - 8, 16, QtCore.Qt.AlignRight, label)
        else:
            y_axis = 0
            painter.drawLine(0, y_axis, self.width(), y_axis)
            for i in range(self._tick_count):
                t = i / (self._tick_count - 1)
                x = int(t * (self.width() - 1))
                value = self._min_val + t * span
                painter.drawLine(x, y_axis, x, y_axis + 6)
                label = self._format_value(value)
                painter.drawText(x - 24, y_axis + 8, 48, 16, QtCore.Qt.AlignHCenter, label)


class WaterfallCanvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self._colormap = make_colormap()
        self._rgb = None
        self.setMinimumSize(600, 400)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)

    def set_waterfall(self, waterfall: np.ndarray, db_min: float, db_max: float) -> None:
        if waterfall.size == 0:
            return
        if db_max <= db_min:
            return

        norm = (waterfall - db_min) / (db_max - db_min)
        norm = np.clip(norm, 0.0, 1.0)
        idx = (norm * 255.0).astype(np.uint8)
        self._rgb = self._colormap[idx]

        h, w, _ = self._rgb.shape
        qimg = QtGui.QImage(self._rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))


def setup_logging() -> str:
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    def _qt_message_handler(mode, context, message):
        logging.info("Qt: %s", message)

    QtCore.qInstallMessageHandler(_qt_message_handler)

    try:
        log_handle = open(log_path, "a", encoding="utf-8")
        faulthandler.enable(file=log_handle)
    except Exception:
        pass

    return log_path


@dataclass
class AppConfig:
    center_freq_hz: float
    sample_rate_hz: float
    gain: str
    nfft: int
    fps: int
    waterfall_rows: int
    dynamic_range_db: float


class BaseIQSource:
    def __init__(self, sample_rate_hz: float, center_freq_hz: float):
        self.sample_rate_hz = float(sample_rate_hz)
        self.center_freq_hz = float(center_freq_hz)

    def read_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError()

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        self.sample_rate_hz = float(sample_rate_hz)

    def set_center_freq(self, center_freq_hz: float) -> None:
        self.center_freq_hz = float(center_freq_hz)

    def set_gain(self, gain) -> None:
        _ = gain

    def close(self) -> None:
        pass

    @property
    def description(self) -> str:
        return self.__class__.__name__


class SdrIQSource(BaseIQSource):
    def __init__(self, sample_rate_hz: float, center_freq_hz: float, gain):
        from rtlsdr import RtlSdr

        super().__init__(sample_rate_hz, center_freq_hz)
        self._sdr = RtlSdr()
        self._sdr.sample_rate = self.sample_rate_hz
        self._sdr.center_freq = self.center_freq_hz
        self.set_gain(gain)

    def read_samples(self, num_samples: int) -> np.ndarray:
        return self._sdr.read_samples(num_samples).astype(np.complex64, copy=False)

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        super().set_sample_rate(sample_rate_hz)
        self._sdr.sample_rate = self.sample_rate_hz

    def set_center_freq(self, center_freq_hz: float) -> None:
        super().set_center_freq(center_freq_hz)
        self._sdr.center_freq = self.center_freq_hz

    def set_gain(self, gain) -> None:
        self._sdr.gain = gain

    def close(self) -> None:
        try:
            self._sdr.close()
        except Exception:
            pass

    @property
    def description(self) -> str:
        return "RTL-SDR"


class FileIQSource(BaseIQSource):
    def __init__(self, file_path: str, sample_rate_hz: float, center_freq_hz: float, fmt: str):
        super().__init__(sample_rate_hz, center_freq_hz)
        self._iq = load_iq_file(file_path, fmt)
        if self._iq.size == 0:
            raise ValueError("IQ file has no samples")
        self._pos = 0
        self._file_path = file_path

    def read_samples(self, num_samples: int) -> np.ndarray:
        num_samples = int(num_samples)
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        if num_samples <= self._iq.size:
            end = self._pos + num_samples
            if end <= self._iq.size:
                chunk = self._iq[self._pos:end]
                self._pos = end
            else:
                tail = self._iq[self._pos:]
                head = self._iq[: end % self._iq.size]
                chunk = np.concatenate([tail, head])
                self._pos = end % self._iq.size
        else:
            repeats = int(np.ceil(num_samples / self._iq.size))
            expanded = np.tile(self._iq, repeats)
            chunk = expanded[:num_samples]
            self._pos = num_samples % self._iq.size

        return chunk.astype(np.complex64, copy=False)

    @property
    def description(self) -> str:
        return f"IQ File ({os.path.basename(self._file_path)})"


class MockIQSource(BaseIQSource):
    def __init__(self, sample_rate_hz: float, center_freq_hz: float, tone_hz: float = 200e3):
        super().__init__(sample_rate_hz, center_freq_hz)
        self._tone_hz = float(tone_hz)
        self._sample_index = 0

    def read_samples(self, num_samples: int) -> np.ndarray:
        num_samples = int(num_samples)
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        idx = np.arange(num_samples, dtype=np.float32) + self._sample_index
        phase = 2.0 * np.pi * self._tone_hz * (idx / self.sample_rate_hz)
        tone = np.exp(1j * phase)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.2
        self._sample_index += num_samples
        return (tone + noise).astype(np.complex64, copy=False)

    @property
    def description(self) -> str:
        return "Mock IQ"


def load_iq_file(file_path: str, fmt: str) -> np.ndarray:
    fmt = (fmt or "auto").lower()
    if fmt == "auto":
        _, ext = os.path.splitext(file_path)
        fmt = "npy" if ext.lower() == ".npy" else "csv"

    if fmt == "npy":
        data = np.load(file_path)
        if not np.iscomplexobj(data):
            raise ValueError("NPY file must contain complex64/complex128 IQ data")
        return np.asarray(data, dtype=np.complex64)

    data = np.loadtxt(file_path, delimiter=",")
    if data.ndim == 1:
        if np.iscomplexobj(data):
            return np.asarray(data, dtype=np.complex64)
        raise ValueError("CSV IQ file must have two columns: I,Q")
    if data.shape[1] < 2:
        raise ValueError("CSV IQ file must have two columns: I,Q")

    iq = data[:, 0] + 1j * data[:, 1]
    return np.asarray(iq, dtype=np.complex64)


def compute_power_db(iq, nfft, window, dynamic_range_db, db_max=None, clip=True):
    iq = iq - np.mean(iq)
    x = iq * window
    spectrum = np.fft.fftshift(np.fft.fft(x, n=nfft))

    p_db = 10.0 * np.log10((np.abs(spectrum) ** 2) + 1e-20).astype(np.float32)

    noise = float(np.median(p_db))
    p_db = p_db - noise  # noise ~ 0 dB

    if db_max is None:
        db_max = float(np.percentile(p_db, 99))
    else:
        db_max = float(db_max)

    db_min = db_max - float(dynamic_range_db)
    if clip:
        p_db = np.clip(p_db, db_min, db_max).astype(np.float32)

    return p_db, db_min, db_max


class WaterfallApp(QtWidgets.QMainWindow):
    def __init__(self, source: BaseIQSource, config: AppConfig):
        super().__init__()
        self.setWindowTitle("RTL-SDR Real-Time Waterfall / Spectrogram")

        self.source = source
        self.center_freq_hz = config.center_freq_hz
        self.sample_rate_hz = config.sample_rate_hz
        self.gain = config.gain
        self.nfft = max(MIN_NFFT, int(config.nfft))
        self.fps = max(10, int(config.fps))
        self.waterfall_rows = int(config.waterfall_rows)
        self.dynamic_range_db = float(config.dynamic_range_db)

        self.window = np.hanning(self.nfft).astype(np.float32)
        init_db_max = 10.0
        init_db_min = init_db_max - self.dynamic_range_db
        self.waterfall = np.full((self.waterfall_rows, self.nfft), init_db_min, dtype=np.float32)

        self._db_max_smooth = None
        self._closing = False
        self._read_error = False

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(1.0, 2000.0)
        self.freq_input.setDecimals(6)
        self.freq_input.setValue(self.center_freq_hz / 1e6)
        self.freq_input.setSuffix(" MHz")

        self.sr_input = QtWidgets.QDoubleSpinBox()
        self.sr_input.setRange(0.25, 3.2)
        self.sr_input.setDecimals(3)
        self.sr_input.setValue(self.sample_rate_hz / 1e6)
        self.sr_input.setSuffix(" MS/s")

        self.gain_input = QtWidgets.QLineEdit(str(self.gain))
        self.gain_input.setFixedWidth(80)
        self.gain_input.setToolTip('Use "auto" or a number like 35')

        self.fft_input = QtWidgets.QComboBox()
        self.fft_input.addItems(["1024", "2048", "4096", "8192"])
        self.fft_input.setCurrentText(str(self.nfft))

        self.fps_input = QtWidgets.QSpinBox()
        self.fps_input.setRange(10, 60)
        self.fps_input.setValue(int(self.fps))

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)

        self.source_label = QtWidgets.QLabel(f"Source: {self.source.description}")

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
        controls.addWidget(self.source_label)
        controls.addStretch(1)

        plot_row = QtWidgets.QHBoxLayout()
        layout.addLayout(plot_row, 1)

        self.y_axis_label = AxisLabel("Time (newest at top)", vertical=True)
        plot_row.addWidget(self.y_axis_label)

        self.time_scale = AxisScale("vertical", unit="s", tick_count=6)
        plot_row.addWidget(self.time_scale)

        self.image_widget = WaterfallCanvas()
        plot_row.addWidget(self.image_widget, 1)

        self.freq_scale = AxisScale("horizontal", unit="MHz", tick_count=6)
        layout.addWidget(self.freq_scale)

        self.freq_label = QtWidgets.QLabel("Frequency (MHz)")
        self.freq_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.freq_label)

        self.update_freq_axis()
        self.update_time_axis()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_once)
        self.timer.start(int(1000.0 / self.fps))

    def update_freq_axis(self) -> None:
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1.0 / self.source.sample_rate_hz))
        freqs_mhz = (freqs + self.source.center_freq_hz) / 1e6
        x_min = float(freqs_mhz[0])
        x_max = float(freqs_mhz[-1])
        self.freq_scale.set_range(x_min, x_max)
        self.freq_label.setText("Frequency (MHz)")

    def update_time_axis(self) -> None:
        if self.fps <= 0:
            return
        time_span_s = float(self.waterfall_rows) / float(self.fps)
        self.time_scale.set_range(0.0, time_span_s)

    def apply_settings(self) -> None:
        self.center_freq_hz = float(self.freq_input.value()) * 1e6
        self.sample_rate_hz = float(self.sr_input.value()) * 1e6
        self.source.set_center_freq(self.center_freq_hz)
        self.source.set_sample_rate(self.sample_rate_hz)

        gain_txt = self.gain_input.text().strip().lower()
        if gain_txt == "auto":
            self.gain = "auto"
        else:
            try:
                self.gain = float(gain_txt)
            except ValueError:
                pass
        try:
            self.source.set_gain(self.gain)
        except Exception:
            pass

        new_nfft = int(self.fft_input.currentText())
        if new_nfft != self.nfft:
            self.nfft = max(MIN_NFFT, new_nfft)
            self.window = np.hanning(self.nfft).astype(np.float32)
            self.waterfall = np.full((self.waterfall_rows, self.nfft), -120.0, dtype=np.float32)
            self._db_max_smooth = None

        self.fps = int(self.fps_input.value())
        self.timer.start(int(1000.0 / self.fps))

        self.update_freq_axis()
        self.update_time_axis()

    def update_once(self) -> None:
        if self._closing:
            return

        try:
            iq = self.source.read_samples(self.nfft)
        except Exception as exc:
            if not self._read_error:
                self._read_error = True
                self.statusBar().showMessage(f"IQ read error: {exc}")
            return

        if iq.size < self.nfft:
            iq = np.pad(iq, (0, self.nfft - iq.size), mode="wrap")

        power_db, _, db_max_raw = compute_power_db(
            iq,
            self.nfft,
            self.window,
            self.dynamic_range_db,
            db_max=None,
            clip=False,
        )

        if self._db_max_smooth is None:
            self._db_max_smooth = db_max_raw
        else:
            self._db_max_smooth = 0.9 * self._db_max_smooth + 0.1 * db_max_raw

        db_max = self._db_max_smooth
        db_min = db_max - self.dynamic_range_db
        power_db = np.clip(power_db, db_min, db_max).astype(np.float32)

        self.waterfall[1:, :] = self.waterfall[:-1, :]
        self.waterfall[0, :] = power_db
        self.image_widget.set_waterfall(self.waterfall, db_min, db_max)


    def closeEvent(self, event) -> None:
        self._closing = True
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.source.close()
        except Exception:
            pass
        event.accept()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTL-SDR real-time waterfall/spectrogram")
    parser.add_argument("--center-freq", type=float, default=94.9e6, help="Center frequency in Hz")
    parser.add_argument("--sample-rate", type=float, default=2.4e6, help="Sample rate in samples/sec")
    parser.add_argument("--gain", default="auto", help="Gain in dB or 'auto'")
    parser.add_argument("--fft", type=int, default=2048, help="FFT size (>=1024)")
    parser.add_argument("--fps", type=int, default=15, help="Update rate in Hz (>=10)")
    parser.add_argument("--waterfall-rows", type=int, default=300, help="Waterfall history in rows")
    parser.add_argument("--dynamic-range", type=float, default=60.0, help="Dynamic range in dB")
    parser.add_argument("--iq-file", type=str, help="Path to IQ capture file (.csv or .npy)")
    parser.add_argument("--iq-format", type=str, default="auto", choices=["auto", "csv", "npy"])
    parser.add_argument("--mock", action="store_true", help="Use synthetic IQ source")
    parser.add_argument("--no-hardware", action="store_true", help="Do not attempt RTL-SDR hardware")
    return parser.parse_args(argv)


def create_iq_source(args: argparse.Namespace) -> BaseIQSource:
    if args.iq_file:
        return FileIQSource(args.iq_file, args.sample_rate, args.center_freq, args.iq_format)
    if args.mock or args.no_hardware:
        return MockIQSource(args.sample_rate, args.center_freq)

    try:
        return SdrIQSource(args.sample_rate, args.center_freq, args.gain)
    except Exception as exc:
        raise RuntimeError(
            f"RTL-SDR unavailable ({exc}). Ensure drivers are installed and the DLL is in rtlsdr_dll."
        )


def main(argv=None) -> None:
    log_path = setup_logging()
    args = parse_args(argv)

    config = AppConfig(
        center_freq_hz=args.center_freq,
        sample_rate_hz=args.sample_rate,
        gain=args.gain,
        nfft=max(MIN_NFFT, args.fft),
        fps=max(10, args.fps),
        waterfall_rows=args.waterfall_rows,
        dynamic_range_db=max(60.0, args.dynamic_range),
    )

    try:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_Use96Dpi, True)
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    logging.info("App starting, log file: %s", log_path)
    source = create_iq_source(args)
    window = WaterfallApp(source, config)
    window.resize(1200, 650)
    window.show()
    if hasattr(app, "exec"):
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())
