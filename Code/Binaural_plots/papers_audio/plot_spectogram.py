import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

AUDIO_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Binaural_plots/papers_audio"

def plot_spectrogram(wav_path, out_path, title):
    """Plot and save high-quality spectrogram for a single WAV file."""
    x, fs = sf.read(wav_path)
    if x.ndim > 1:
        x = x[:, 0]

    # ===== STFT =====
    f, t, Zxx = stft(x, fs=fs, nperseg=1024, noverlap=768)
    Sxx_db = 20 * np.log10(np.abs(Zxx) + 1e-8)
    Sxx_db = np.clip(Sxx_db, -80, 0)

    # ===== Plot =====
    plt.figure(figsize=(10, 4), dpi=800)
    plt.imshow(Sxx_db, origin='lower', aspect='auto',
               extent=[t[0], t[-1], 0, fs/2000],
               cmap='inferno', vmin=-80, vmax=0, interpolation='gaussian')
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# ===== Generate all spectrograms =====
pairs = [
    ("clean_4.wav",     "Clean Speech – Babble 6 dB"),
    ("estimated_4.wav", "Enhanced Speech – Babble 6 dB"),
    ("mixture_4.wav",   "Mixture – Babble 6 dB"),
    ("clean_6.wav",     "Clean Speech – Babble 3 dB"),
    ("estimated_6.wav", "Enhanced Speech – Babble 3 dB"),
    ("mixture_6.wav",   "Mixture – Babble 3 dB"),
]

for fname, title in pairs:
    wav_path = os.path.join(AUDIO_DIR, fname)
    spec_path = os.path.join(AUDIO_DIR, f"{os.path.splitext(fname)[0]}_spectrogram.png")
    print(f"Processing {fname} → {os.path.basename(spec_path)}")
    plot_spectrogram(wav_path, spec_path, title)
