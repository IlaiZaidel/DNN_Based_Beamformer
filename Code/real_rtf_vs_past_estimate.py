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

# -------- Project utils (must exist) --------
from utils import Preprocesing, return_as_complex
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking


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
NOISE_TYPE      = "pink"    # or "pink" babble
MOVING_SPEAKER  = True        # if False → static source
DELTA_ANGLE     = 50          # movement delta in degrees

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# --- Full deterministic setup (important for CUDA) ---
# instead, allow non-deterministic operations
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # can speed things up a bit
# =====================================================
# ================ INITIALIZATION =====================
# =====================================================
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)
from signal_generator import SignalGenerator


# =====================================================
# ====================== HELPERS ======================
# =====================================================
def onesided_to_full_spectrum(H_pos):
    Fpos = H_pos.shape[0]
    nfft = (Fpos - 1) * 2
    H_conj = np.conj(H_pos[1:-1])[::-1] if nfft % 2 == 0 else np.conj(H_pos[1:])[::-1]
    return np.concatenate([H_pos, H_conj], axis=0)


def rtf_time_shift_and_truncate(H_pos, F_L, F_R, nfft):
    H_full = onesided_to_full_spectrum(H_pos)
    g = np.fft.ifft(H_full, n=nfft)
    g_trc = np.zeros_like(g)
    g_trc[:F_R], g_trc[-F_L:] = g[:F_R], g[-F_L:]
    g_plot = np.fft.ifftshift(g_trc).real.astype(np.float32)
    G_full_trc = np.fft.fft(g_trc, n=nfft)
    Fpos = nfft // 2 + 1
    H_pos_trc = G_full_trc[:Fpos]
    return H_pos_trc, g_plot


def true_rtf_from_all_rirs(all_rirs, win_len, ref_mic, window="hann", eps=1e-12):
    M, L, K = all_rirs.shape
    A_true = np.zeros((M, win_len, L), dtype=np.complex64)
    for li in range(L):
        h_mk = all_rirs[:, li, :]
        h_seg = h_mk[:, :win_len] if K >= win_len else np.pad(h_mk, ((0, 0), (0, win_len - K)))
        H_mf = np.fft.fft(h_seg, n=win_len, axis=1)
        denom = H_mf[ref_mic, :] + eps
        A_true[:, :, li] = (H_mf / denom[None, :]).astype(np.complex64)
    return A_true


def estimate_rtf_past(clean, fs, win_len, hop, ref_mic, noise_head_s, beta=0.95, device=torch.device("cpu")):
    N, M = clean.shape
    x = torch.from_numpy(clean.astype(np.float32)).to(device)[None, :, :]
    Y_stfts = []
    for m in range(M):
        Y = Preprocesing(x[:, :, m:m+1], win_len, fs, N/fs, hop, device)
        Y_stfts.append(return_as_complex(Y))
    Yc = torch.cat(Y_stfts, dim=1)
    W_past, eigvals, eigvecs = pastd_rank1_whitened(Yc, noise_head_s, beta=beta)
    a_hat = rtf_from_subspace_tracking(W_past, eigvals, eigvecs, noise_head_s, ref_mic)
    Ln = int(noise_head_s * fs // hop)
    return Yc, a_hat, Ln


def generate_pink(N):
    b = [0.049922035, 0.050612699, 0.050979644, 0.048882058]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return sg.lfilter(b, a, np.random.randn(N).astype(np.float32)).astype(np.float32)


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

elif NOISE_TYPE == "pink":
    from rir_generator import generate as rir_generate
    pink_base = generate_pink(N_clean)
    L_room = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
    beta = [float(row["beta"])] * 6
    n_taps = int(row["n"])
    mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
    noise1_pos = [float(row["noise1_x"]), float(row["noise1_y"]), float(row["noise1_z"])]
    rir_n1 = np.array(rir_generate(C, fs, mic_pos, noise1_pos, L_room, beta, n_taps, order=0)).T # Change ORDER
    n1 = np.stack([sg.lfilter(rir_n1[m], [1.0], pink_base) for m in range(M)], axis=1)[:N_clean, :]
    SNR_PINK_DB = 8.0
    n_power = float(np.sum(n1[:, ref_idx] ** 2) + 1e-12)
    n1 *= np.sqrt(d_power * 10.0 ** (-SNR_PINK_DB / 10.0) / n_power).astype(np.float32)
    mixture = clean + n1 + v
    print(f"[OK] Added PINK noise @ {SNR_PINK_DB:.1f} dB (ref mic {ref_idx})")

else:
    raise ValueError("NOISE_TYPE must be 'babble' or 'pink'")

# Normalize mixture
peak = float(np.max(np.abs(mixture)) + 1e-12)
mixture, clean = mixture / peak, clean / peak


# =====================================================
# ================== RTF ESTIMATION ===================
# =====================================================
A_true_full = true_rtf_from_all_rirs(all_rirs, WIN_LENGTH, REF_MIC)
_, a_hat, Ln = estimate_rtf_past(mixture, FS, WIN_LENGTH, HOP, REF_MIC,
                                 NOISE_HEAD_S, beta=BETA_PAST, device=DEVICE)

Mics, Ffull, L_true = A_true_full.shape
Fpos = WIN_LENGTH // 2 + 1
A_true_pos = A_true_full[:, :Fpos, :]
A_hat_np = a_hat[0].detach().cpu().numpy()
A_hat_mfl = np.transpose(A_hat_np, (1, 0, 2))

L_hat = A_hat_mfl.shape[2]
L_true_s = max(0, L_true - Ln)
L_use = min(L_true_s, L_hat)
if L_use <= 0:
    raise RuntimeError("No overlap between TRUE and PASTd after removing noise head.")

A_true_use = A_true_pos[:, :, Ln:Ln+L_use]
A_hat_use  = A_hat_mfl[:, :, :L_use]


# =====================================================
# ======================= PLOTS =======================
# =====================================================
mic = MIC_TO_PLOT if MIC_TO_PLOT is not None else (1 if REF_MIC == 0 else 0)
frames = (
    sorted(set([min(10, L_use-1), L_use//2, max(0, L_use-10)]))
    if FRAMES_TO_SHOW is None else
    [f for f in FRAMES_TO_SHOW if 0 <= f < L_use]
)

plt.figure(figsize=(4.6*len(frames), 3.3), dpi=130)
t_ms = np.arange(WIN_LENGTH) / FS * 1e3

for i, t in enumerate(frames, 1):
    H_true, H_hat = A_true_use[mic, :, t], A_hat_use[mic, :, t]
    _, g_true = rtf_time_shift_and_truncate(H_true, F_L, F_R, WIN_LENGTH)
    _, g_hat  = rtf_time_shift_and_truncate(H_hat,  F_L, F_R, WIN_LENGTH)
    ax = plt.subplot(1, len(frames), i)
    ax.plot(t_ms, g_true, '-', label='TRUE (shift+gate)')
    ax.plot(t_ms, g_hat,  '--', label='PAST (shift+gate)')
    ax.set_title(f"Mic {mic} | Frame {t}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    if i == 1:
        ax.legend()

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=160)
plt.close()
print(f"[OK] Saved {OUT_FIG}")
