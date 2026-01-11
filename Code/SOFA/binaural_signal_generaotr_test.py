#!/usr/bin/env python3
# simple_signal_generator_test_librispeech.py
#
# End-to-end test for HRTF-based SignalGenerator using REAL speech input:
# 1) Load HRIRs + (az,elev) grid from SOFA
# 2) Load LibriSpeech waveform as input
# 3) Create circular moving source
# 4) Run C++ generate_hrtf()
# 5) Save binaural WAV

import os
import sys
import numpy as np
import soundfile as sf

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_BINAURAL_HRTF_rtf_output"
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)

from signal_generator import SignalGenerator  # pybind module

SOFA_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA"
if SOFA_DIR not in sys.path:
    sys.path.insert(0, SOFA_DIR)

from hrtf_convolve import SOFA_HRTF_wrapper

# ---------------- Files ----------------
SOFA_PATH = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/SOFA/riec/RIEC_hrir_subject_001.sofa"

INPUT_WAV = (
    "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/"
    "LibriSpeech/Test/5105/28241/5105-28241-0011.wav"
)

OUT_WAV = os.path.join(SCRIPT_DIR, "librispeech_hrtf_circle_2.wav")

# ---------------- Config ----------------
FS = 16000

STFT_WIN = 512
STFT_HOP = 128
UPDATE_EVERY = STFT_HOP

RADIUS = 1.2       # meters
AZ_START = -90.0
AZ_END =90.0
HEAD_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SRC_Z = 0.0

# ---------------- Helpers ----------------
def deg2rad(d):
    return d * np.pi / 180.0

def circular_source_path(T, radius, az_start, az_end, head_pos, z=0.0):
    az = np.linspace(az_start, az_end, T, endpoint=False)
    azr = deg2rad(az)
    x = head_pos[0] + radius * np.cos(azr)
    y = head_pos[1] + radius * np.sin(azr)
    z = np.full(T, head_pos[2] + z, dtype=np.float64)
    return np.stack([x, y, z], axis=1)  # [T,3]

# ---------------- Main ----------------
def main():
    # 1) Load speech signal
    x, fs = sf.read(INPUT_WAV, dtype="float64")

    if fs != FS:
        raise RuntimeError(f"Expected fs={FS}, got fs={fs}")

    if x.ndim > 1:
        x = x[:, 0]  # force mono

    # Normalize safely
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    x = x[:64000]
    T = len(x)
    print(f"[OK] Loaded input WAV: {INPUT_WAV}")
    print(f"     T = {T} samples ({T / FS:.2f} sec)")

    # 2) Load SOFA HRIR database
    hrtf = SOFA_HRTF_wrapper(SOFA_PATH)

    azelevs = hrtf.azelevs.detach().cpu().numpy().astype(np.float64)  # [N,2]
    N_dirs = azelevs.shape[0]

    hrirs_list = []
    K = None

    for i in range(N_dirs):
        az_i = float(azelevs[i, 0])
        el_i = float(azelevs[i, 1])

        h = hrtf.get_hrtf(az_i, el_i, fs=FS)  # [K,2]
        h = np.asarray(h, dtype=np.float64)

        if h.ndim != 2 or h.shape[1] != 2:
            raise RuntimeError(f"Bad HRIR shape at az={az_i}, el={el_i}: {h.shape}")

        if K is None:
            K = h.shape[0]
        elif h.shape[0] != K:
            raise RuntimeError("HRIR length mismatch")

        hrirs_list.append(h.T)  # [2,K]

    hrirs = np.stack(hrirs_list, axis=0)  # [N,2,K]

    print(f"[OK] Loaded SOFA HRIRs: N={N_dirs}, K={K}")

    # 3) Create motion paths
    r_head_path = np.tile(HEAD_POS[None, :], (T, 1))  # fixed head
    s_path = circular_source_path(
        T, RADIUS, AZ_START, AZ_END, HEAD_POS, z=SRC_Z
    )

    # 4) Run C++ generator
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

    y = np.asarray(result.output, dtype=np.float64).T  # [T,2]

    # Normalize output
    peak = np.max(np.abs(y))
    if peak > 0:
        y /= peak

    # 5) Save result
    sf.write(OUT_WAV, y.astype(np.float32), FS)
    print(f"[OK] Saved binaural WAV: {OUT_WAV}")

if __name__ == "__main__":
    main()
