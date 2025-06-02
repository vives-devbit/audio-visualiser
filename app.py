import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import atexit

# -------------------------------
# Audio Device Selection
# -------------------------------
p = pyaudio.PyAudio()

print("Available audio input devices:")
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev["maxInputChannels"] > 0:
        input_devices.append(i)
        print(f"[{i}] {dev['name']} ({int(dev['maxInputChannels'])} ch @ {int(dev['defaultSampleRate'])} Hz)")

if not input_devices:
    raise RuntimeError("No input devices found.")

device_index = int(input(f"Select input device index from the list above: "))
if device_index not in input_devices:
    raise ValueError("Invalid input device selected.")

# -------------------------------
# Audio Parameters
# -------------------------------
CHUNK = 1024
RATE = int(p.get_device_info_by_index(device_index)['defaultSampleRate'])
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Open Stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

# Clean-up on exit
def cleanup():
    stream.stop_stream()
    stream.close()
    p.terminate()
atexit.register(cleanup)

# -------------------------------
# Plot Setup
# -------------------------------
plt.ion()
fig = plt.figure(figsize=(16, 9))
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2])
ax_time = plt.subplot(gs[0])
ax_freq = plt.subplot(gs[1])
ax_spec = plt.subplot(gs[2])

# Time domain
x_time = np.arange(0, CHUNK) / RATE
line_time, = ax_time.plot(x_time, np.zeros(CHUNK))
ax_time.set_title("Time Domain")
ax_time.set_ylim(-32768, 32767)
ax_time.set_xlim(0, CHUNK / RATE)

# Frequency spectrum
x_freq = np.fft.rfftfreq(CHUNK, d=1. / RATE)
line_freq, = ax_freq.semilogx(x_freq, np.zeros(len(x_freq)))
ax_freq.set_title("Frequency Spectrum")
ax_freq.set_xlim(20, RATE / 2)
ax_freq.set_ylim(0, 1000)

# Scrolling Spectrogram
spec_data = np.zeros((100, len(x_freq)))
spec_img = ax_spec.imshow(spec_data,
                          aspect='auto',
                          extent=[x_freq[0], x_freq[-1], 0, 100],
                          origin='lower',
                          interpolation='nearest',
                          cmap='inferno')
ax_spec.set_title("Scrolling Spectrogram")
ax_spec.set_ylabel("Time (scrolling)")
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_yticks([])

# -------------------------------
# Animation Function
# -------------------------------
def update_plot(frame):
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Time-domain
    line_time.set_ydata(audio_data)

    # Frequency domain
    windowed = audio_data * np.hanning(len(audio_data))
    fft_data = np.abs(np.fft.rfft(windowed)) / CHUNK
    line_freq.set_ydata(fft_data)

    # Spectrogram
    spec_data[:-1] = spec_data[1:]
    spec_data[-1] = fft_data
    spec_img.set_data(spec_data)

    return line_time, line_freq, spec_img

ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=False, cache_frame_data=False)
plt.show()
