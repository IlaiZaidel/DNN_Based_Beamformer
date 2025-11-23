#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

# ==== Your project imports ====
# Make sure PYTHONPATH points to your project root so these resolve.
from rir_generator import generate as rir_generate
from utils import Preprocesing, return_as_complex, Postprocessing  # Postprocessing not strictly needed here
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking

# ================== CONFIG ==================
SPEECH_WAV   = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28241/5105-28241-0011.wav"
OUT_DIR      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/rtf_directpath_demo"
os.makedirs(OUT_DIR, exist_ok=True)

FS           = 16000
T_SEC        = 4.0
CK           = 343.0                # speed of sound (m/s)
WIN_LENGTH   = 512
HOP          = WIN_LENGTH // 4
MICS         = 8
ROOM_Z       = 3.0
MIC_REF_1B   = 4                    # 1-based reference mic for RTF
NOISE_HEAD_S = 0.5                  # noise-only head (sec) for PASTd
SNR_WHITE_DB = 100.0                 # white vs clean @ ref mic (dB)
RIR_ORDER    = 0                    # DIRECT PATH ONLY
RIR_TAPS     = int(0.6 * FS)        # long enough but only direct path is used for order=0

# ============ Random room + array + single source ============
np.random.seed(int(time.time()) & 0xFFFFFFFF)

# Room size
x_lim = np.random.uniform(6.0, 9.0)
y_lim = np.random.uniform(6.0, 9.0)
L = [x_lim, y_lim, ROOM_Z]
beta6 = [0.0]*6  # irrelevant for order=0 (no reflections), but keep API happy

# Array pose
angle_orientation = np.random.randint(-45, 46)  # degrees
theta = np.deg2rad(angle_orientation)
mic_center_x = np.random.uniform(2.0, x_lim - 2.0)
mic_center_y = np.random.uniform(2.0, y_lim - 2.0)
mic_height   = np.random.uniform(1.1, 1.6)

# Linear 8-mic array (aperture ~0.34 m), rotate by theta
mic_offsets = np.array([[-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
                        [ 0.04, 0], [ 0.07, 0], [ 0.12, 0], [ 0.17, 0]])
Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)]])
mic_xy = mic_offsets @ Rmat.T + np.array([mic_center_x, mic_center_y])
mic_positions = np.column_stack([mic_xy, np.full((MICS,), mic_height)])

# One speech source at radius 1.2–1.6 m & random DOA in front-half plane
radius   = np.random.uniform(1.2, 1.6)
angle_src = np.random.randint(0, 181)
sx = mic_center_x + radius*np.cos(np.deg2rad(angle_src + angle_orientation))
sy = mic_center_y + radius*np.sin(np.deg2rad(angle_src + angle_orientation))
sz = np.random.uniform(1.2, 1.8)
src_pos = [float(sx), float(sy), float(sz)]

# ============ Load speech (mono), trim/tile to N ============
N = int(FS*T_SEC)
x, fs_file = sf.read(SPEECH_WAV)
if x.ndim > 1: x = x[:, 0]
if fs_file != FS:
    raise RuntimeError(f"Sample-rate mismatch: wav={fs_file}, expected {FS}")
if len(x) < N:
    reps = int(math.ceil(N/len(x)))
    x = np.tile(x, reps)
x = x[:N].astype(np.float64)

# ============ RIRs (order=0 -> direct path only) ============
# rir_generate returns (taps, M) -> transpose to (M, taps)
h_speech = np.array(rir_generate(CK, FS, mic_positions, src_pos, L, beta6,
                                 RIR_TAPS, order=RIR_ORDER)).T  # (M, taps)

# ============ Direct speech at mics via FFT-conv ============
xM = np.tile(x[:, None], (1, MICS))  # (N, M)
d_full = np.stack(
    [signal.fftconvolve(xM[:, m], h_speech[m], mode="full")[:N] for m in range(MICS)],
    axis=1
)  # (N, M)

# Optional noise-only head to satisfy "noise_only_time" for PASTd
nlead = int(NOISE_HEAD_S*FS)
keep  = N - nlead
d = np.vstack([np.zeros((nlead, MICS), dtype=np.float64), d_full[:keep, :]])  # (N, M)

# ============ Additive white noise only ============
ref = MIC_REF_1B - 1
d_pow = np.sum(d[:, ref]**2) + 1e-12

v = np.random.randn(N, MICS)
v_pow = np.sum(v[:, ref]**2) + 1e-12
scale_v = np.sqrt(d_pow * 10**(-SNR_WHITE_DB/10.0) / v_pow)
v *= scale_v

# Mixture
y = d + v

# Peak normalize mixture/components together (keeps SNR)
peak = np.max(np.abs(y))
if peak < 1e-12: peak = 1.0
y /= peak; d /= peak; v /= peak

# Optionally save WAVs for sanity check
sf.write(os.path.join(OUT_DIR, "mix.wav"),   y.astype(np.float32), FS)
sf.write(os.path.join(OUT_DIR, "clean.wav"), d.astype(np.float32), FS)
sf.write(os.path.join(OUT_DIR, "white.wav"), v.astype(np.float32), FS)

# ============ STFT preprocessing -> [B, M, 2F, L] ============
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y_t = torch.from_numpy(y[None, ...]).float().to(device)              # (1, N, M)
Y   = Preprocesing(y_t, WIN_LENGTH, FS, T_SEC, HOP, device)          # (1, M, 2F, L)
Yc  = return_as_complex(Y)                                           # (1, M,  F, L)

B, M, F, L = Yc.shape
print("Yc shape:", Yc.shape, "(B,M,F,L)")

# ============ PASTd RTF estimation ============
# noise_only_time is needed by your rtf_from_subspace_tracking (for its padding convention)
noise_only_time = NOISE_HEAD_S
W, eigvals, eigvecs = pastd_rank1_whitened(Yc, noise_only_time, beta=0.995)

# Single-head RTF w.r.t. MIC_REF_1B
a_hat = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time, MIC_REF_1B)
# a_hat: (B, F, M, T_eff) — frequency-domain RTFs across time (after the noise-only head)
print("a_hat shape (B,F,M,T):", a_hat.shape)

# ============ Make a time-aligned RTF tensor like you do in the model ============
R = HOP
Ln = int(noise_only_time * FS // R)   # number of STFT frames in noise-only head
B1, F1, M1, T_eff = a_hat.shape
L_total = T_eff + Ln
zeros_pad = torch.zeros((B1, F1, M1, Ln), dtype=a_hat.dtype, device=a_hat.device)
rtf_tracking = torch.cat([zeros_pad, a_hat], dim=-1)   # (B, F, M, L_total)
# Reorder to (B, M, F, L_total) and stack real/imag like your UNet expects (not strictly needed for plotting)
rtf_trk_BMFL = rtf_tracking.permute(0, 2, 1, 3).contiguous()          # (B, M, F, L)
rtf_realimag = torch.cat([rtf_trk_BMFL.real, rtf_trk_BMFL.imag], 2)   # (B, M, 2F, L)

# ============ Convert frequency-domain RTF -> time domain (per mic) ============
# We'll collapse time and take an average across the active speech region for a "static" RTF,
# then IFFT across frequency to get a relative impulse response per mic.
# Choose a time window after the noise-only head:
t_start = Ln + 5
t_end   = min(L_total, t_start + 50)  # short window
a_stat  = rtf_tracking[:, :, :, t_end]  # (B, F, M)
a_stat  = a_stat.permute(0, 2, 1).contiguous()                      # (B, M, F)

# IFFT across frequency (works with one-sided spectrum as produced by your pipeline)
# We'll follow your earlier pattern: simple ifft over F, then ifftshift.
ifft_results = torch.fft.ifftshift(torch.fft.ifft(a_stat.detach().cpu(), dim=-1), dim=-1)  # (B, M, F)
h_rel = ifft_results[0].numpy()  # (M, F) complex -> time-like index (not true taps length, but useful)

# ============ Plot time-domain RTF (relative IR per mic) ============
def irtf_from_one_sided(H_half: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    H_half: (..., F) one-sided complex spectrum (DC..Nyquist), F = n_fft//2 + 1
    returns: (..., n_fft) time-domain IR via Hermitian reconstruction + IFFT
    """
    assert H_half.size(-1) == n_fft // 2 + 1, "F must be n_fft//2 + 1"
    # Split out DC and Nyquist (scalar along last dim)
    H_dc      = H_half[..., :1]                  # k = 0
    H_nyquist = H_half[..., -1:]                 # k = N/2
    # Interior positive freqs (1..N/2-1)
    H_pos_int = H_half[..., 1:-1]                # shape (..., F-2)

    # Build the negative-frequency part by conjugate reverse of interior
    H_neg = torch.conj(torch.flip(H_pos_int, dims=[-1]))

    # Concatenate: [DC] [1..N/2-1] [Nyquist] [N/2+1..N-1]
    H_full = torch.cat([H_dc, H_pos_int, H_nyquist, H_neg], dim=-1)  # (..., n_fft)

    # IFFT over last dim. Result should be (approximately) real-valued if H_full is Hermitian.
    h = torch.fft.ifft(H_full, n=n_fft, dim=-1)
    return h

# ---- pick a short time window (after noise head) and average/collapse if you like ----
# Your a_stat currently: (B, M, F). If instead you want to average across a time span:
# a_stat_win = rtf_tracking[:, :, :, t_start:t_end].mean(dim=-1).permute(0,2,1)  # (B, M, F)

n_fft = WIN_LENGTH  # 512
# a_stat: (B, M, F) one-sided complex RTF
h_rel_td = irtf_from_one_sided(a_stat, n_fft=n_fft)  # (B, M, N_fft), complex dtype

# Convert to numpy for plotting. Take the first batch.
h_rel = h_rel_td[0].detach().cpu().numpy()  # (M, N_fft)
# (Optional) If tiny imaginary residues remain due to non-perfect Hermitian symmetry:
h_rel = np.real_if_close(h_rel, tol=1000)

# ============ Plot time-domain RTF (magnitude only) ============
t_idx = np.arange(h_rel.shape[1])
for m in range(MICS):
    plt.figure(figsize=(9, 4))
    plt.plot(t_idx, h_rel[m], label="|RTF|")
    plt.title(f"RTF (time-domain magnitude) – Mic {m+1} w.r.t Ref {MIC_REF_1B}")
    plt.xlabel("Sample index (IR from irfft of [0..π])")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"rtf_time_mag_mic{m+1}.png"), dpi=200)
    plt.close()