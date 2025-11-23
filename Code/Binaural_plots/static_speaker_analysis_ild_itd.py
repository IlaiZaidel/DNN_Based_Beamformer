#!/usr/bin/env python3
"""
TRUE vs. PASTd-estimated time-varying RTF comparison.

Generates a synthetic multichannel mixture using the SignalGenerator,
adds either babble or pink noise, estimates RTFs using PASTd, and
compares them against the true generator RTFs (time-domain view).

Author: Ilai Zaidel
"""

import os, sys, math, ast, random
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sg
import matplotlib.pyplot as plt
import torch
import os, sys, math, ast, time, re
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import scipy.io as sio
import torch


# ===== Project imports =====
from omegaconf import OmegaConf
cfg = OmegaConf.load("DNN_Based_Beamformer/Code/conf/config.yaml")
from rir_generator import generate as rir_generate
import sys, os
sys.path.append("/home/dsi/ilaiz/DNN_Based_Beamformer/Code")
from LoadPreTrainedModel import loadPreTrainedModel
from utils import Preprocesing, Postprocessing, return_as_complex

# =====================================================
# ====================== CONFIG =======================
# =====================================================
CSV_PATH   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_rtf_output"
INDEX      =  3

# Room & signal generator
C          = 343.0
NSAMPLES   = 1024
ORDER      = 0
HOP_AIR    = 32
M_TYPE     = "o"

# STFT / RTF
FS         = 16000
WIN_LENGTH = 512
HOP        = 128          # must match RIR snapshot stride
REF_MIC    = 0
NOISE_HEAD_S = 0.5
BETA_PAST  = 0.98

# Analysis
MIC_TO_PLOT    = None
FRAMES_TO_SHOW = None
OUT_FIG        = "rtf_time_domain_true_vs_past.png"

# Truncation (MATLAB-style)
F_L, F_R = 256, 256

# Experiment toggles
NOISE_TYPE      = "babble"    # or "pink" babble
MOVING_SPEAKER  = False        # if False → static source
DELTA_ANGLE     = 50          # movement delta in degrees

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
OUT_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Binaural_plots/Paper"
MODEL_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Trained_Models/28_10_2025_BABBLE_ATTEN_DUAL_TRUE_RIRS/trained_model_dataset_withoutRev.pt"
def stamp_now():
    return time.strftime("%Y%m%d_%H%M%S")

model_folder = os.path.basename(os.path.dirname(MODEL_PATH))
model_date   = "_".join(model_folder.split("_")[:2])  # e.g. "13_08"
RUN_DIR = os.path.join(OUT_DIR, f"run_{model_date}_{stamp_now()}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# ================ INITIALIZATION =====================
# =====================================================
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)
from signal_generator import SignalGenerator


# =====================================================
# ====================== HELPERS ======================
# =====================================================

# =====================================================
# ================ ROOM + SIGNAL GEN ==================
# =====================================================
df = pd.read_csv(CSV_PATH)
row = df.iloc[INDEX].to_dict()

wav_path = row["speaker_path"]
if wav_path.startswith("/dsi/gannot-lab1/"):
    wav_path = wav_path.replace("/dsi/gannot-lab1/", "/dsi/gannot-lab/gannot-lab1/", 1)

fs = int(row.get("fs", FS))
T_sec = float(row.get("T", 4.0))
L_room = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
beta = [0.1] * 6

x, fs_file = sf.read(wav_path)
if x.ndim > 1:
    x = x[:, 0]
if fs_file != fs:
    raise RuntimeError(f"fs mismatch: wav={fs_file}, csv={fs}")

N = int(T_sec * fs)
if len(x) < N:
    x = np.tile(x, math.ceil(N / len(x)))
x = x[:N].astype(np.float64)

# --- Microphone array ---
angleOrientation = random.choice(np.arange(-45, 46))
mic_height = random.uniform(1.0, 1.5)
mic_x = random.uniform(1.0, L_room[0] - 1.0)
mic_y = random.uniform(1.0, L_room[1] - 1.0)
center = np.array([mic_x, mic_y])

mic_offsets = np.array([
    [-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
    [ 0.04, 0], [ 0.07, 0], [ 0.12, 0], [ 0.17, 0]
])
rot = np.array([[np.cos(np.radians(angleOrientation)), -np.sin(np.radians(angleOrientation))],
                [np.sin(np.radians(angleOrientation)),  np.cos(np.radians(angleOrientation))]])
mic_rotated = mic_offsets @ rot.T + center
mic_pos = np.column_stack([mic_rotated, np.full(len(mic_rotated), mic_height)])
M = mic_pos.shape[0]

# --- Source path ---
radius = random.uniform(1.0, 1.5)
angle_x = random.randint(0, 180)
speaker_start = np.array([
    mic_x + radius * np.cos(np.radians(angle_x + angleOrientation)),
    mic_y + radius * np.sin(np.radians(angle_x + angleOrientation)),
    random.uniform(1.2, 1.8)
])
if MOVING_SPEAKER:
    speaker_stop = np.array([
        mic_x + radius * np.cos(np.radians(angle_x + DELTA_ANGLE + angleOrientation)),
        mic_y + radius * np.sin(np.radians(angle_x + DELTA_ANGLE + angleOrientation)),
        random.uniform(1.2, 1.8)
    ])
else:
    speaker_stop = speaker_start.copy()

# --- Build paths ---
sp_path = np.zeros((N, 3))
rp_path = np.zeros((N, 3, M))
for i in range(0, N, HOP_AIR):
    alpha = i / max(1, N - 1)
    sp = speaker_start + alpha * (speaker_stop - speaker_start)
    end = min(i + HOP_AIR, N)
    sp_path[i:end] = sp
    for m in range(M):
        rp_path[i:end, :, m] = mic_pos[m]

# --- Run generator ---
gen = SignalGenerator()
result = gen.generate(list(x), C, fs, rp_path.tolist(), sp_path.tolist(),
                      L_room, beta, NSAMPLES, M_TYPE, ORDER, 3, [], True)
all_rirs = np.array(result.all_rirs, dtype=np.float64)
clean    = np.array(result.output, dtype=np.float64).T  # (N, M)


# =====================================================
# ================== NOISE ADDITION ===================
# =====================================================
ref_idx = REF_MIC
N_clean, M = clean.shape
d_power = float(np.sum(clean[:, ref_idx] ** 2) + 1e-12)

# --- White noise ---
SNR_WHITE_DB = 30.0
v = np.random.randn(N_clean, M).astype(np.float32)
scale_v = np.sqrt(d_power * 10.0 ** (-SNR_WHITE_DB / 10.0) /
                  (np.sum(v[:, ref_idx] ** 2) + 1e-12)).astype(np.float32)
v *= scale_v

if NOISE_TYPE == "babble":
    BABBLE_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Babble_Noise/Test"
    babble_path = os.path.join(BABBLE_DIR, f"babble_{INDEX:07d}.wav")
    if not os.path.exists(babble_path):
        raise FileNotFoundError(f"Babble file not found: {babble_path}")
    babble, sr_b = sf.read(babble_path, always_2d=True)
    if sr_b != fs:
        raise RuntimeError(f"Sample rate mismatch: babble={sr_b}, fs={fs}")
    if babble.shape[0] < N_clean:
        babble = np.tile(babble, (math.ceil(N_clean / babble.shape[0]), 1))
    babble = babble[:N_clean, :M]
    b_power = float(np.sum(babble[:, ref_idx] ** 2) + 1e-12)
    SNR_BABBLE_DB = 10.0
    babble *= np.sqrt(d_power * 10.0 ** (-SNR_BABBLE_DB / 10.0) / b_power).astype(np.float32)
    mixture = clean + babble + v
    print(f"[OK] Added BABBEL noise @ {SNR_BABBLE_DB:.1f} dB (ref mic {ref_idx})")


# =====================================================
# ================== RUN MODEL ON MIXTURE ==============
# =====================================================
print("\n[STEP] Loading pretrained model...")
model = loadPreTrainedModel(cfg).to(DEVICE).eval()
print(f"[OK] Model loaded on {DEVICE}")

# --- Prepare mixture for inference ---
print("[STEP] Preparing STFT input...")
T_SEC = N_clean / FS
HOP_STFT = WIN_LENGTH // 4
X_t = torch.from_numpy(mixture[None, ...]).float().to(DEVICE)   # (1, N, M)
Y = Preprocesing(X_t, WIN_LENGTH, FS, T_SEC, HOP_STFT, DEVICE)
# Convert RIRs to torch tensor (float32 for GPU FFT)
all_rirs_t = torch.from_numpy(all_rirs).float().unsqueeze(0).to(DEVICE)



# --- Run model (ExNet-BF+PF dual-branch) ---
print("[STEP] Running model inference...")
with torch.no_grad():
    # Pass to model
    W_Stage1_left, W_Stage1_right, X_hat_Stage1_C_left, X_hat_Stage1_C_right, _ = model(Y, all_rirs_t, DEVICE)

# --- ISTFT reconstruction and normalization ---
print("[STEP] Reconstructing left/right binaural signals...")
x_hat_left_B = Postprocessing(X_hat_Stage1_C_left, HOP_STFT, WIN_LENGTH, DEVICE)
x_hat_right_B = Postprocessing(X_hat_Stage1_C_right, HOP_STFT, WIN_LENGTH, DEVICE)

# Normalize each channel independently to avoid clipping
max_left = torch.max(torch.abs(x_hat_left_B))
max_right = torch.max(torch.abs(x_hat_right_B))
x_hat_left = (x_hat_left_B / (max_left + 1e-8)).cpu().numpy().flatten()
x_hat_right = (x_hat_right_B / (max_right + 1e-8)).cpu().numpy().flatten()

# Combine into stereo waveform (L,R)
x_hat_stereo = np.stack([x_hat_left, x_hat_right], axis=-1).astype(np.float32)

# =====================================================
# ================== SAVE RESULTS =====================
# =====================================================
os.makedirs(RUN_DIR, exist_ok=True)
out_wav = os.path.join(RUN_DIR, f"beamformed_binaural_mix_{INDEX:03d}.wav")
sf.write(out_wav, x_hat_stereo, FS)
print(f"[DONE] Binaural beamformed mixture saved to:\n  {out_wav}")