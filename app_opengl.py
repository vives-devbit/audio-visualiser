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

# device_index = int(input("Select input device index: "))
device_index = 0  # Default to the first device
if device_index not in input_devices:
    raise ValueError("Invalid device index.")

# -------------------------------
# Audio Parameters
# -------------------------------
CHUNK = 1024 # Size of each audio chunk
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

logo_path = resource_path("Logo VIVES Hogeschool - Smile - wit.png")
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
        margin * 2 + win.height() // 2 - logo_label.width() // 2  # Adjust vertical position
    )
# Call once and also on resize
position_logo()
win.resizeEvent = lambda event: (position_logo(), super(type(win), win).resizeEvent(event))


# Set OpenGL acceleration
if USE_OPENGL:
    pg.setConfigOptions(useOpenGL=True, antialias=True)
else:
    print("⚠️ OpenGL not available. Falling back to CPU rendering.")

# scrolling Time Domain Plot of 10 seconds
plot_time1 = win.addPlot(title="Tijds-domein (10 seconden)", **font_style)
plot_time1.setYRange(-10000, 10000) # (-32768, 32767)
curve_time1 = plot_time1.plot(pen=pg.mkPen('c', width=2))
plot_time1.setLabel('left', "Geluidssterkte", **font_style)
plot_time1.setLabel('bottom', "Tijd", units='s', **font_style)

# Number of points to display in the scrolling plot
display_points = 20000
scroll_time = 10  # seconds
plot_time1.setXRange(0, scroll_time)  # Show 10 seconds of data

# Buffer for the interpolated display (initialize with zeros)
scroll_x = np.linspace(0, scroll_time, display_points)
scroll_y = np.zeros(display_points, dtype=np.float32)

# Time Domain Plot
win.nextCol()
plot_time = win.addPlot(title="Tijds-domein (%d milliseconden)"%(int(CHUNK / RATE*1000)), **font_style)
plot_time.setYRange(-10000, 10000) # (-32768, 32767)
curve_time = plot_time.plot(pen=pg.mkPen('c', width=2))
plot_time.setLabel('left', "Geluidssterkte", **font_style)
plot_time.setLabel('bottom', "Tijd", units='s', **font_style)
plot_time.setXRange(0, CHUNK / RATE)

# Frequency Spectrum
win.nextRow()
plot_fft = win.addPlot(title="Frequentie-domein (spectrum)", **font_style, colspan=3)
curve_fft = plot_fft.plot(pen=pg.mkPen('m', width=2))
plot_fft.setLabel('left', "Geluidssterkte", **font_style)
plot_fft.setLabel('bottom', "Frequency (aantal golven per seconde)", units='Hz', **font_style)
plot_fft.setXRange(0, RATE / 2)
plot_fft.setLogMode(x=False, y=False)
# plot_fft.setYRange(-2, 4)
plot_fft.setYRange(0, 500)

# Scrolling Spectrogram
win.nextRow()
spec_len = int(scroll_time * (RATE / 2) / (CHUNK * 2))  # Number of columns in the spectrogram
fft_bins = CHUNK // 2 + 1
spec_data = np.zeros((spec_len, fft_bins))
img_spec = pg.ImageItem()
plot_spec = win.addPlot(title="Scrolling Spectrogram", **font_style, colspan=3)
plot_spec.addItem(img_spec)
plot_spec.setXRange(0, scroll_time) # Show 10 seconds of data

# Set the image scale so that the y-axis matches the frequency range and the x-axis matches the time range
img_spec.setImage(spec_data, autoLevels=False, rect=pg.QtCore.QRectF(0, 0, scroll_time, RATE/2))
img_spec.setLookupTable(pg.colormap.get('inferno').getLookupTable())
img_spec.setLevels([0, 100])  # adjusted for log scaling

# Set the plot's y-limits to match the frequency range
plot_spec.setLimits(xMin=0, xMax=fft_bins, yMin=0, yMax=RATE / 4)
# plot_spec.setYRange(0, RATE / 2)

plot_spec.setLabel('bottom', "Tijd", units='s', **font_style)
plot_spec.setLabel('left', "Frequency", units='Hz', **font_style)


# -------------------------------
# Update Function
# -------------------------------
def update():
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

     # --- Scrolling Time Domain Plot (10s, interpolated, incremental) ---
    global scroll_y

    # Interpolate only the new chunk to the display resolution
    # chunk_time = np.linspace(0, scroll_time, len(curve_time1))
    new_time = np.linspace(scroll_time - (CHUNK / RATE), scroll_time, CHUNK, endpoint=False)
    new_interp_x = np.linspace(scroll_time - (CHUNK / RATE), scroll_time, int(display_points * CHUNK / (CHUNK * 100)), endpoint=False)
    new_interp_y = np.interp(new_interp_x, new_time, audio_data)

    # Shift the buffer and append new points
    n_new = len(new_interp_y)
    scroll_y = np.roll(scroll_y, -n_new)
    scroll_y[-n_new:] = new_interp_y

    # Update the plot
    curve_time1.setData(scroll_x, scroll_y)

    # Time-domain
    x_time = np.arange(CHUNK) / RATE
    curve_time.setData(x_time, audio_data)

    # FFT
    windowed = audio_data * np.hamming(len(audio_data))
    fft_data = np.abs(np.fft.rfft(windowed)) / CHUNK
    fft_freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    curve_fft.setData(fft_freqs, fft_data)

    # Spectrogram
    global spec_data
    spec_data[:-1] = spec_data[1:]
    spec_data[-1] = fft_data
    # spec_data[-1] = 20 * np.log10(np.clip(fft_data, 1e-10, None))
    img_spec.setImage(spec_data, autoLevels=False, rect=pg.QtCore.QRectF(0, 0, scroll_time, RATE/2))
    

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
