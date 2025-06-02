import sys
import os
import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# -------------------------------
# Check for OpenGL / GPU support
# -------------------------------
try:
    import OpenGL
    USE_OPENGL = True
except ImportError:
    USE_OPENGL = False

# -------------------------------
# Audio Device Selection
# -------------------------------
p = pyaudio.PyAudio()
input_devices = []
print("Available audio input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev["maxInputChannels"] > 0:
        input_devices.append(i)
        print(f"[{i}] {dev['name']} ({int(dev['maxInputChannels'])} ch @ {int(dev['defaultSampleRate'])} Hz)")

if not input_devices:
    raise RuntimeError("No input devices found.")

device_index = int(input("Select input device index: "))
if device_index not in input_devices:
    raise ValueError("Invalid device index.")

# -------------------------------
# Audio Parameters
# -------------------------------
CHUNK = 2048 # Size of each audio chunk
RATE = int(p.get_device_info_by_index(device_index)["defaultSampleRate"])
CHANNELS = 1
FORMAT = pyaudio.paInt16

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

class AudioVisualizer(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__(show=True, title="Real-Time Audio Visualizer")
        self.setWindowTitle("Audio Visualizer")
        self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            QtWidgets.QApplication.quit()

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

logo_path = resource_path("Logo VIVES Hogeschool - Smile.png")
# logo_img = pg.ImageItem(pg.imread(logo_path))

# -------------------------------
# GUI Setup
# -------------------------------
app = QtWidgets.QApplication(sys.argv)
win = AudioVisualizer()
win.setWindowTitle("Audio Visualizer")
win.showFullScreen()
font_style = {'color': '#FFF', 'font-size': '14pt', 'bold': True}

# Add this after win.showFullScreen()
logo_label = QLabel(win)
logo_pixmap = QPixmap(logo_path)
logo_pixmap = logo_pixmap.scaledToWidth(400, Qt.SmoothTransformation)
logo_label.setPixmap(logo_pixmap)
logo_label.setAttribute(Qt.WA_TranslucentBackground)
logo_label.setStyleSheet("background: transparent;")
logo_label.adjustSize()
logo_label.show()

# Position the logo in the top right corner with some margin
margin = 20
def position_logo():
    logo_label.move(
        win.width() - logo_label.width() - margin,
        margin
    )
# Call once and also on resize
position_logo()
win.resizeEvent = lambda event: (position_logo(), super(type(win), win).resizeEvent(event))


# Set OpenGL acceleration
if USE_OPENGL:
    pg.setConfigOptions(useOpenGL=True, antialias=True)
else:
    print("⚠️ OpenGL not available. Falling back to CPU rendering.")

# Time Domain Plot
plot_time = win.addPlot(title="Time Domain", **font_style)
plot_time.setYRange(-10000, 10000) # (-32768, 32767)
curve_time = plot_time.plot(pen=pg.mkPen('c', width=2))
plot_time.setLabel('left', "Amplitude", **font_style)
plot_time.setLabel('bottom', "Time", units='s', **font_style)
plot_time.setXRange(0, CHUNK / RATE)

# Frequency Spectrum
win.nextRow()
plot_fft = win.addPlot(title="Frequency Spectrum (FFT)", **font_style)
curve_fft = plot_fft.plot(pen=pg.mkPen('m', width=2))
plot_fft.setLabel('left', "Magnitude", **font_style)
plot_fft.setLabel('bottom', "Frequency", units='Hz', **font_style)
plot_fft.setXRange(0, RATE / 2)
plot_fft.setLogMode(x=False, y=True)
plot_fft.setYRange(-2, 4)

# Scrolling Spectrogram
win.nextRow()
spec_len = 200
fft_bins = CHUNK // 2 + 1
spec_data = np.zeros((spec_len, fft_bins))
img_spec = pg.ImageItem()
plot_spec = win.addPlot(title="Scrolling Spectrogram", **font_style)
plot_spec.addItem(img_spec)

# Set the image scale so that the y-axis matches the frequency range
img_spec.setImage(spec_data, autoLevels=False, rect=pg.QtCore.QRectF(0, 0, spec_len, RATE/2))
img_spec.setLookupTable(pg.colormap.get('inferno').getLookupTable())
img_spec.setLevels([0, 50])  # adjusted for log scaling

# Set the plot's y-limits to match the frequency range
plot_spec.setLimits(xMin=0, xMax=fft_bins, yMin=0, yMax=RATE / 4)
# plot_spec.setYRange(0, RATE / 2)

plot_spec.setLabel('bottom', "Time", units='s', **font_style)
plot_spec.setLabel('left', "Frequency", units='Hz', **font_style)


# -------------------------------
# Update Function
# -------------------------------
def update():
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Time-domain
    x_time = np.arange(CHUNK) / RATE
    curve_time.setData(x_time, audio_data)

    # FFT
    windowed = audio_data * np.hanning(len(audio_data))
    fft_data = np.abs(np.fft.rfft(windowed)) / CHUNK
    fft_freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    curve_fft.setData(fft_freqs, fft_data)

    # Spectrogram
    global spec_data
    spec_data[:-1] = spec_data[1:]
    # spec_data[-1] = fft_data
    spec_data[-1] = 20 * np.log10(np.clip(fft_data, 1e-10, None))
    img_spec.setImage(spec_data, autoLevels=False, rect=pg.QtCore.QRectF(0, 0, spec_len, RATE/2))
    

# Timer for 30fps
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

# Exit cleanly
def close():
    stream.stop_stream()
    stream.close()
    p.terminate()

app.aboutToQuit.connect(close)
sys.exit(app.exec_())
