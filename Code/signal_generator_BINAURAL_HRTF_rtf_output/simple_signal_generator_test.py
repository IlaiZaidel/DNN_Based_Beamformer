#!/usr/bin/env python3
# simple_signal_generator_test.py
#
# Minimal end-to-end test for the HRTF-based SignalGenerator extension:
# 1) Load HRIRs + (az,elev) grid from a SOFA file using your existing wrappers
# 2) Create a simple circular source path around a fixed head position
# 3) Run C++ generate_hrtf() (piecewise-constant HRIR updates)
# 4) Save binaural WAV next to this script

import os
import sys
import math
import numpy as np
import soundfile as sf

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Your compiled pybind module lives in this directory (same as this script)
SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_BINAURAL_HRTF_rtf_output"
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)

from signal_generator import SignalGenerator  # pybind module

# Import your SOFA wrappers from parent directory
# /home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA/hrtf_convolve.py
SOFA_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA"
if SOFA_DIR not in sys.path:
    sys.path.insert(0, SOFA_DIR)

import sys

sys.path.insert(0, "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA")

from hrtf_convolve import SOFA_HRTF_wrapper

SOFA_PATH = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/SOFA/riec/RIEC_hrir_subject_001.sofa"

# ---------------- Test config ----------------
FS = 16000
T_SEC = 4.0
T = int(FS * T_SEC)

STFT_WIN = 512
STFT_HOP = 128
UPDATE_EVERY = STFT_HOP  # best default: update at hop boundaries

RADIUS = 1.2      # meters
ELEV_DEG = 0.0    # keep horizontal plane
AZ_START = 0.0
AZ_END = 360.0    # full circle
HEAD_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # fixed head center
SRC_Z = 0.0       # same height as head (simple)

OUT_WAV = os.path.join(SCRIPT_DIR, "simple_hrtf_circle.wav")

# ---------------- Helpers ----------------
def deg2rad(d):
    return d * np.pi / 180.0

def make_test_signal(fs, T):
    # A simple "click train" + short noise burst (robust and easy to hear direction changes)
    x = np.zeros(T, dtype=np.float64)
    click_every = int(0.25 * fs)
    click_len = int(0.002 * fs)
    for n in range(0, T, click_every):
        end = min(T, n + click_len)
        x[n:end] += 0.9 * np.hanning(end - n)
    # Add small wideband noise
    rng = np.random.default_rng(0)
    x += 0.02 * rng.standard_normal(T)
    # Normalize
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return x

def circular_source_path(T, fs, radius, az_start, az_end, head_pos, z=0.0):
    # az sweeps linearly from az_start to az_end over T samples
    az = np.linspace(az_start, az_end, T, endpoint=False)
    azr = deg2rad(az)
    x = head_pos[0] + radius * np.cos(azr)
    y = head_pos[1] + radius * np.sin(azr)
    z = np.full(T, head_pos[2] + z, dtype=np.float64)
    return np.stack([x, y, z], axis=1)  # [T,3]

# ---------------- Main ----------------
def main():
    # 1) Load SOFA HRIR database through wrapper
    hrtf = SOFA_HRTF_wrapper(SOFA_PATH)

    # Collect all measured directions
    azelevs = hrtf.azelevs.detach().cpu().numpy().astype(np.float64)  # [N,2] in degrees
    N_dirs = azelevs.shape[0]

    # Build HRIR tensor [N,2,K] by asking wrapper for each (az,elev)
    # NOTE: wrapper returns h shape [K,2] (left/right). We convert to [2,K].
    hrirs_list = []
    K = None
    for i in range(N_dirs):
        az_i = float(azelevs[i, 0])
        el_i = float(azelevs[i, 1])
        h = hrtf.get_hrtf(az_i, el_i, fs=FS)  # expected shape [K,2]
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != 2:
            raise RuntimeError(f"Unexpected HRIR shape at (az={az_i}, el={el_i}): {h.shape}")
        if K is None:
            K = h.shape[0]
        else:
            if h.shape[0] != K:
                raise RuntimeError("HRIR length K is inconsistent across directions.")
        hrirs_list.append(h.T)  # [2,K]
    hrirs = np.stack(hrirs_list, axis=0)  # [N,2,K]

    print(f"[OK] Loaded SOFA HRIRs: N={N_dirs}, K={K}, fs={FS}")

    # 2) Create test signal and paths
    x = make_test_signal(FS, T)
    r_head_path = np.tile(HEAD_POS[None, :], (T, 1))     # [T,3] fixed
    s_path = circular_source_path(T, FS, RADIUS, AZ_START, AZ_END, HEAD_POS, z=SRC_Z)  # [T,3]

    # 3) Run C++ generator
    gen = SignalGenerator()
    result = gen.generate_hrtf(
        list(x),
        s_path.tolist(),
        r_head_path.tolist(),
        hrirs.tolist(),
        azelevs.tolist(),
        STFT_WIN,
        STFT_HOP,
        UPDATE_EVERY
    )

    y = np.array(result.output, dtype=np.float64)  # [2,T]
    y = y.T  # [T,2]

    # Normalize and save
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    sf.write(OUT_WAV, y.astype(np.float32), FS)
    print(f"[OK] Saved binaural WAV: {OUT_WAV}")

    # Optional: also save a quick mono reference signal
    # sf.write(os.path.join(SCRIPT_DIR, "simple_hrtf_input_mono.wav"), x.astype(np.float32), FS)

if __name__ == "__main__":
    main()
