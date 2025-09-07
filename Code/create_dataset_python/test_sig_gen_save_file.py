#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf
import scipy.io as sio

# ==== USER SETTINGS ====
MAT_PATH = "/dsi/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen/clean_example_0000000.mat"
FS = 16000  # Hz (update if your data used a different fs)
OUT_SUFFIX = "_stereo.wav"
# =======================

# --- Load .mat ---
data = sio.loadmat(MAT_PATH)
if "clean" not in data:
    raise KeyError(f"'clean' variable not found in {MAT_PATH}")
clean = np.array(data["clean"], dtype=np.float64)  # (samples, mics) expected

# --- Heuristic: fix orientation if needed (mics x samples) ---
if clean.shape[0] < clean.shape[1] and clean.shape[1] > 8:
    # Likely (mics, samples) -> transpose to (samples, mics)
    clean = clean.T

# --- Build stereo from outermost mics (or duplicate if mono) ---
num_mics = clean.shape[1] if clean.ndim == 2 else 1
if num_mics >= 2:
    left = clean[:, 0]
    right = clean[:, -1]
    stereo = np.stack([left, right], axis=1)
else:
    stereo = np.repeat(clean, 2, axis=1)

# --- Peak normalize (avoid clipping) ---
peak = np.max(np.abs(stereo))
if np.isfinite(peak) and peak > 0:
    stereo = stereo / peak

# --- Save WAV (float32) ---
out_name = os.path.splitext(os.path.basename(MAT_PATH))[0] + OUT_SUFFIX
out_path = os.path.join(os.getcwd(), out_name)
sf.write(out_path, stereo.astype(np.float32), FS)

dur = stereo.shape[0] / FS
print(f"Saved: {out_path}  |  fs={FS} Hz  |  duration={dur:.2f}s  |  shape={stereo.shape}")
