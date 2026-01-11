import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve, resample_poly

# audio I/O: prefer soundfile, fallback to scipy
try:
    import soundfile as sf
    def read_wav(path):
        x, fs = sf.read(path, always_2d=False)
        return x, fs
    def write_wav(path, x, fs):
        sf.write(path, x, fs)
except ImportError:
    from scipy.io import wavfile
    def read_wav(path):
        fs, x = wavfile.read(path)
        # convert int to float32 in [-1,1]
        if np.issubdtype(x.dtype, np.integer):
            maxv = np.iinfo(x.dtype).max
            x = x.astype(np.float32) / maxv
        else:
            x = x.astype(np.float32)
        return x, fs
    def write_wav(path, x, fs):
        # write as float32 WAV if scipy supports, else int16
        x32 = x.astype(np.float32)
        wavfile.write(path, fs, x32)

# Your SOFA reader
import sys
sys.path.insert(0, '../../src')
import sofa


# -----------------------
# User paths / settings
# -----------------------
SOFA_PATH = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/SOFA/ari/hrtf_nh16.sofa"
WAV_IN    = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28240/5105-28240-0021.wav"
OUT_DIR   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA"

# Choose which measurement (direction) to use
# You can change this to any valid 0..(HRTF.Dimensions.M-1)
MEASUREMENT_M =25
EMITTER_E = 1 # usually 0 for single-emitter datasets

OUT_WAV  = os.path.join(OUT_DIR, f"binaural_convolved_M{MEASUREMENT_M}.wav")
OUT_PNG  = os.path.join(OUT_DIR, f"hrir_plot_M{MEASUREMENT_M}.png")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_mono(x):
    # if input is stereo, average channels
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def peak_normalize_stereo(y, peak=0.99):
    m = np.max(np.abs(y))
    if m < 1e-12:
        return y
    return y * (peak / m)


def main():
    ensure_dir(OUT_DIR)

    # -----------------------
    # Load SOFA / HRIR
    # -----------------------
    H = sofa.Database.open(SOFA_PATH)

    # Sampling rate in SOFA
    sofa_fs = float(H.Data.SamplingRate.get_values())
    N = int(H.Dimensions.N)
    R = int(H.Dimensions.R)

    if R < 2:
        raise RuntimeError(f"SOFA has R={R} receivers; need at least 2 for binaural.")

    # HRIRs: shape (N,) for each ear
    hL = H.Data.IR.get_values(indices={"M": MEASUREMENT_M, "R": 0, "E": EMITTER_E}).astype(np.float64)
    hR = H.Data.IR.get_values(indices={"M": MEASUREMENT_M, "R": 1, "E": EMITTER_E}).astype(np.float64)
    pos = H.Source.Position.get_values(system="spherical", angle_unit="degree")
    print(pos[MEASUREMENT_M])  # [az, elev, r]
    H.close()

    # -----------------------
    # Plot HRIR (time domain)
    # -----------------------
    t = np.arange(N) / sofa_fs
    plt.figure(figsize=(14, 4))
    plt.plot(t, hL, label="Left HRIR (R=0)")
    plt.plot(t, hR, label="Right HRIR (R=1)")
    plt.title(f"HRIR from {os.path.basename(SOFA_PATH)} at M={MEASUREMENT_M}, E={EMITTER_E}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    # -----------------------
    # Load audio and resample if needed
    # -----------------------
    x, fs_in = read_wav(WAV_IN)
    x = to_mono(x).astype(np.float64)

    if fs_in != int(sofa_fs):
        # Rational resampling: good quality, fast
        # Use integer ratio fs_out/fs_in
        from math import gcd
        fs_out = int(round(sofa_fs))
        g = gcd(fs_in, fs_out)
        up = fs_out // g
        down = fs_in // g
        print(f"[INFO] Resampling: {fs_in} -> {fs_out} Hz (up={up}, down={down})")
        x = resample_poly(x, up, down)
        fs = fs_out
    else:
        fs = fs_in

    # -----------------------
    # Convolve (binaural)
    # -----------------------
    yL = fftconvolve(x, hL, mode="full")
    yR = fftconvolve(x, hR, mode="full")

    y = np.stack([yL, yR], axis=1)  # shape (T, 2)
    # y = peak_normalize_stereo(y, peak=0.99).astype(np.float32)

    # -----------------------
    # Save stereo WAV
    # -----------------------
    write_wav(OUT_WAV, y, fs)

    print(f"[OK] Saved WAV: {OUT_WAV}")
    print(f"[OK] Saved HRIR plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
