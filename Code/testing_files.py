import numpy as np
import matplotlib.pyplot as plt
import librosa

# Load the audio file
file_path = "/home/dsi/ilaiz/output_original_audio_1.wav"
y, sr = librosa.load(file_path, sr=None)  # `sr=None` keeps the original sampling rate

# Create the time axis
t = np.linspace(0, len(y) / sr, len(y))

# Plot the waveform
plt.figure(figsize=(12, 6))
plt.plot(t, y, label="Waveform Original")
plt.title("Waveform in Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Save the plot to a file
output_path = "/home/dsi/ilaiz/waveform_original_plot.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")

# Load the audio file
file_path = "/home/dsi/ilaiz/output_clean_audio_1.wav"
y, sr = librosa.load(file_path, sr=None)  # `sr=None` keeps the original sampling rate

# Create the time axis
t = np.linspace(0, len(y) / sr, len(y))

# Plot the waveform
plt.figure(figsize=(12, 6))
plt.plot(t, y, label="Waveform Clean")
plt.title("Waveform in Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Save the plot to a file
output_path = "/home/dsi/ilaiz/waveform_clean_plot.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")