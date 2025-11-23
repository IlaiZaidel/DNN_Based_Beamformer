#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSE-over-time: TRUE vs. PASTd-estimated time-varying RTFs (time-domain RIR view)

- Loads clean + all_rirs (.mat) and multichannel babble (.wav) for INDEX
- Mixes at requested SNRs (babble + white) while preserving noise-only head
- Estimates RTF via PASTd
- Converts to time-domain relative impulse responses (shift+gate)
- Plots MSE(frame) between TRUE and EST RIRs vs time

Author: Ilai Zaidel
"""

import os, math
import numpy as np
import soundfile as sf
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

# ======== USER CONFIG (edit here) ========
INDEX            =  10
TEST_PATH        = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Static_Signal_Gen_with_rir"
BUBBLE_PATH      = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Babble_Noise/Test"   # multichannel babble_{INDEX}.wav
OUT_PNG          = "rtf_mse_over_time.png"

# STFT / RTF
FS               = 16000
WIN_LENGTH       = 512
HOP              = 128
REF_MIC          = 0                 # 0-based
NOISE_HEAD_S     = 0.5               # seconds of noise-only head present in clean
BETA_PAST        = 0.98

# Noise SNRs @ ref mic (dB)
SNR_BABBLE_DB    =10.0
SNR_WHITE_DB     = 30.0

# Time-domain RIR gating (MATLAB-style)
F_L, F_R         = 256, 256          # left/right taps kept; WIN_LENGTH must be >= F_L+F_R

# Plotting / sampling
MAX_POINTS       = 120               # max frames plotted (uniform sample)
MIC_TO_COMPARE   = None              # if None → first non-ref mic
SEED             = 0                 # for reproducibility (white noise)
# ========================================


OUT_DIR   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/figs_rtf_eval"
OUT_STEM  = f"rtf_mse_idx{INDEX}_snrB{SNR_BABBLE_DB:g}_snrW{SNR_WHITE_DB:g}_beta{BETA_PAST:g}_ref{REF_MIC}"
OUT_PNG   = os.path.join(OUT_DIR, OUT_STEM + ".png")
# -------- Project utils (must exist/importable) --------
from utils import Preprocesing, return_as_complex
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking


# ====================== HELPERS ======================
def onesided_to_full_spectrum(H_pos: np.ndarray) -> np.ndarray:
    """Mirror positive-frequency spectrum to full FFT bins (complex)."""
    Fpos = H_pos.shape[0]
    nfft = (Fpos - 1) * 2
    if nfft % 2 == 0:
        H_conj = np.conj(H_pos[1:-1])[::-1]
    else:
        H_conj = np.conj(H_pos[1:])[::-1]
    return np.concatenate([H_pos, H_conj], axis=0)


def rtf_time_shift_and_truncate(H_pos: np.ndarray, F_L: int, F_R: int, nfft: int):
    """
    Given one-sided RTF H_pos (Fpos,), make time-domain relative impulse response:
      - IFFT to time
      - Gate F_L (tail) + F_R (head)
      - ifftshift for plotting alignment
    Returns:
        H_pos_trc: complex (Fpos,) — re-FFT of gated time RIR
        g_plot: float (nfft,) — centered (ifftshift) time sequence for comparison
    """
    H_full = onesided_to_full_spectrum(H_pos)
    g = np.fft.ifft(H_full, n=nfft)
    g_trc = np.zeros_like(g)
    g_trc  = g
    g_trc= g
    g_plot = np.fft.ifftshift(g_trc).real.astype(np.float32)
    G_full_trc = np.fft.fft(g_trc, n=nfft)
    Fpos = nfft // 2 + 1
    H_pos_trc = G_full_trc[:Fpos]
    return H_pos_trc, g_plot

def evaluate_single_snr(SNR_BABBLE_DB):
    """Run the whole pipeline for one SNR and return mean MSE."""
    # (same code as before inside main, until after mse_per_frame computation)
    # ------------------------------------------------------------
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Load clean + all_rirs ---
    mat_path = os.path.join(TEST_PATH, f"clean_example_{INDEX:07d}.mat")
    if not os.path.exists(mat_path):
        alt = mat_path.replace("/dsi/gannot-lab1/", "/dsi/gannot-lab/gannot-lab1/", 1)
        if os.path.exists(alt):
            mat_path = alt
        else:
            raise FileNotFoundError(mat_path)
    data = sio.loadmat(mat_path)
    d_full   = np.asarray(data["clean"], dtype=np.float32)
    all_rirs = np.asarray(data["all_rirs"], dtype=np.float32)
    N, M = d_full.shape
    fs = FS

    # --- Build d with noise-only head ---
    nlead = int(NOISE_HEAD_S * fs)
    d = np.vstack([np.zeros((nlead, M), dtype=np.float32), d_full])
    N_full = d.shape[0]

    # --- Load babble ---
    b_path = os.path.join(BUBBLE_PATH, f"babble_{INDEX:07d}.wav")
    if not os.path.exists(b_path):
        alt = b_path.replace("/dsi/gannot-lab1/", "/dsi/gannot-lab/gannot-lab1/", 1)
        if os.path.exists(alt):
            b_path = alt
        else:
            raise FileNotFoundError(b_path)
    babble, sr = sf.read(b_path, always_2d=True)
    if sr != fs:
        raise RuntimeError(f"Samp. rate mismatch: babble={sr}, expected fs={fs}")
    if babble.shape[0] < N_full:
        babble = np.tile(babble, (math.ceil(N_full / babble.shape[0]), 1))
    babble = babble[:N_full, :M].astype(np.float32, copy=False)

    # --- Mix ---
    ref_idx = REF_MIC
    d_pow = float(np.sum(d[:, ref_idx] ** 2) + 1e-12)
    b_pow = float(np.sum(babble[:, ref_idx] ** 2) + 1e-12)
    scale_b = np.sqrt(d_pow * 10.0 ** (-SNR_BABBLE_DB / 10.0) / b_pow).astype(np.float32)
    babble *= scale_b

    v = np.random.randn(N_full, M).astype(np.float32)
    v_pow = float(np.sum(v[:, ref_idx] ** 2) + 1e-12)
    scale_v = np.sqrt(d_pow * 10.0 ** (-SNR_WHITE_DB / 10.0) / v_pow).astype(np.float32)
    v *= scale_v

    y = d + babble + v
    peak = float(np.max(np.abs(y)) + 1e-12)
    y /= peak; d /= peak; babble /= peak

    # --- TRUE RTFs + PASTd ---
    A_true_pos = true_rtf_from_all_rirs(all_rirs, WIN_LENGTH, REF_MIC)
    a_hat, Ln = estimate_rtf_past(y, FS, WIN_LENGTH, HOP, REF_MIC, NOISE_HEAD_S,
                                  beta=BETA_PAST, device=None)
    A_hat_np = a_hat[0].detach().cpu().numpy()
    A_hat_mfl = np.transpose(A_hat_np, (1, 0, 2))

    # Align lengths
    Mics, Fpos, L_true = A_true_pos.shape
    L_hat = A_hat_mfl.shape[2]
    L_use = min(L_true, L_hat)
    A_true_use = A_true_pos
    A_hat_use  = A_hat_mfl

    mic = 1 if REF_MIC == 0 else 0
    frame_idx = np.arange(L_use)
    mse_per_frame = np.zeros(L_use)
    center = WIN_LENGTH // 2
    support = F_L + F_R
    lo = center - support // 2
    hi = lo + support

    for t in frame_idx:
        H_true = A_true_use[mic, :, t]
        H_hat  = A_hat_use[mic,  :, t]
        _, g_true = rtf_time_shift_and_truncate(H_true, F_L, F_R, WIN_LENGTH)
        _, g_hat  = rtf_time_shift_and_truncate(H_hat,  F_L, F_R, WIN_LENGTH)
        norm_true =np.sqrt(np.sum(g_true**2))
        norm_hat =np.sqrt(np.sum(g_hat**2))
        g_true /= np.sqrt(np.sum(g_true**2))
        g_hat  /= np.sqrt(np.sum(g_hat**2))
        nmse = np.mean((g_true[lo:hi] - g_hat[lo:hi]) ** 2)/(norm_hat ** 2)
        mse_per_frame[t] = 10 * np.log10(nmse + 1e-12)
    mean_mse = np.mean(mse_per_frame)
    return mean_mse

def true_rtf_from_all_rirs(all_rirs: np.ndarray, win_len: int, ref_mic: int, eps=1e-12):
    """
    all_rirs: shape (M, L, K) — from generator .mat
    Returns A_true: (M, Fpos, L) complex64, one-sided
    """
    M, L, K = all_rirs.shape
    Fpos = win_len // 2 + 1
    A_true = np.zeros((M, Fpos, L), dtype=np.complex64)
    for li in range(L):
        h_mk = all_rirs[:, li, :]
        if K >= win_len:
            h_seg = h_mk[:, :win_len]
        else:
            pad = np.zeros((M, win_len - K), dtype=h_mk.dtype)
            h_seg = np.concatenate([h_mk, pad], axis=1)
        H_mf = np.fft.rfft(h_seg, n=win_len, axis=1)  # (M, Fpos)
        denom = H_mf[ref_mic, :] + eps
        A_true[:, :, li] = (H_mf / denom[None, :]).astype(np.complex64)
    return A_true


@torch.no_grad()
def estimate_rtf_past(y: np.ndarray, fs: int, win_len: int, hop: int,
                      ref_mic: int, noise_head_s: float, beta=0.98,
                      device=None):
    """
    y: (N, M) mixture
    Returns:
        a_hat: torch.complex64, shape (B=1, Fpos, M, L_hat)
        Ln: int, number of noise-only frames
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N, M = y.shape
    x = torch.from_numpy(y.astype(np.float32)).to(device)[None, :, :]  # (1, N, M)

    Y_stfts = []
    for m in range(M):
        Y = Preprocesing(x[:, :, m:m+1], win_len, fs, N / fs, hop, device)
        Y_stfts.append(return_as_complex(Y))                            # (1, Fpos, 1, L)
    Yc = torch.cat(Y_stfts, dim=1)                                      # (1, Fpos, M, L)

    W_past, eigvals, eigvecs = pastd_rank1_whitened(Yc, noise_head_s, beta=beta)
    a_hat = rtf_from_subspace_tracking(W_past, eigvals, eigvecs, noise_head_s, ref_mic)
    Ln = int(noise_head_s * fs // hop)
    return a_hat, Ln


def uniform_sample_indices(L: int, max_points: int):
    if L <= max_points:
        return np.arange(L)
    xs = np.linspace(0, L - 1, num=max_points)
    return np.unique(np.round(xs).astype(int))


# ====================== MAIN (no CLI) ======================
def main():
    # seeds (white noise)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Load clean + all_rirs (.mat) ---
    mat_path = os.path.join(TEST_PATH, f"clean_example_{INDEX:07d}.mat")
    if not os.path.exists(mat_path):
        alt = mat_path.replace("/dsi/gannot-lab1/", "/dsi/gannot-lab/gannot-lab1/", 1)
        if os.path.exists(alt):
            mat_path = alt
        else:
            raise FileNotFoundError(mat_path)

    data = sio.loadmat(mat_path)
    d_full   = np.asarray(data["clean"], dtype=np.float32)      # (samples, M)
    all_rirs = np.asarray(data["all_rirs"], dtype=np.float32)   # (M, L, K)

    N, M = d_full.shape
    fs = FS

    # Build d with noise-only head
    nlead = int(NOISE_HEAD_S * fs)
    d = np.vstack([np.zeros((nlead, M), dtype=np.float32), d_full])    # (N+nlead, M)
    N_full = d.shape[0]

    # --- Load multichannel babble ---
    b_path = os.path.join(BUBBLE_PATH, f"babble_{INDEX:07d}.wav")
    if not os.path.exists(b_path):
        alt = b_path.replace("/dsi/gannot-lab1/", "/dsi/gannot-lab/gannot-lab1/", 1)
        if os.path.exists(alt):
            b_path = alt
        else:
            raise FileNotFoundError(b_path)
    babble, sr = sf.read(b_path, always_2d=True)
    if sr != fs:
        raise RuntimeError(f"Samp. rate mismatch: babble={sr}, expected fs={fs}")

    # Ensure length/channels
    if babble.shape[0] < N_full:
        reps = math.ceil(N_full / babble.shape[0])
        babble = np.tile(babble, (reps, 1))
    babble = babble[:N_full, :M].astype(np.float32, copy=False)

    # --- SNR scaling (ref mic) ---
    ref_idx = REF_MIC
    d_pow = float(np.sum(d[:, ref_idx] ** 2) + 1e-12)
    b_pow = float(np.sum(babble[:, ref_idx] ** 2) + 1e-12)
    scale_b = np.sqrt(d_pow * 10.0 ** (-SNR_BABBLE_DB / 10.0) / b_pow).astype(np.float32)
    babble *= scale_b

    v = np.random.randn(N_full, M).astype(np.float32)
    v_pow = float(np.sum(v[:, ref_idx] ** 2) + 1e-12)
    scale_v = np.sqrt(d_pow * 10.0 ** (-SNR_WHITE_DB / 10.0) / v_pow).astype(np.float32)
    v *= scale_v

    # --- Final mix + peak normalize ---
    y = d  + v  + babble
    peak = float(np.max(np.abs(y)) + 1e-12)
    y /= peak; d /= peak; babble /= peak; v /= peak

    # --- TRUE RTFs (one-sided) ---
    A_true_pos = true_rtf_from_all_rirs(all_rirs, WIN_LENGTH, REF_MIC)   # (M, Fpos, L_true)

    # --- PASTd estimate ---
    a_hat, Ln = estimate_rtf_past(y, FS, WIN_LENGTH, HOP, REF_MIC, NOISE_HEAD_S,
                                  beta=BETA_PAST, device=None)            # (1, Fpos, M, L_hat)
    A_hat_np = a_hat[0].detach().cpu().numpy()                            # (Fpos, M, L_hat)
    A_hat_mfl = np.transpose(A_hat_np, (1, 0, 2))                         # (M, Fpos, L_hat)

    Mics, Fpos, L_true = A_true_pos.shape
    L_hat = A_hat_mfl.shape[2]
    L_true_s = max(0, L_true - Ln)                                        # drop noise-only head
    L_use = min(L_true_s, L_hat)
    if L_use <= 0:
        raise RuntimeError("No overlap after removing noise-only head.")

    A_true_use = A_true_pos                       # (M, Fpos, L_use)
    A_hat_use  = A_hat_mfl

    # --- Which mic to compare (non-ref) ---
    mic = MIC_TO_COMPARE if MIC_TO_COMPARE is not None else (1 if REF_MIC == 0 else 0)
    if mic == REF_MIC:
        mic = 0 if REF_MIC != 0 else 1

    # --- Frame subsampling for readability ---
    def uniform_sample_indices(L: int, max_points: int):
        if L <= max_points:
            return np.arange(L)
        xs = np.linspace(0, L - 1, num=max_points)
        return np.unique(np.round(xs).astype(int))

    frame_idx = uniform_sample_indices(434, MAX_POINTS)                 # indices
    t_sec = frame_idx * (HOP / FS)

    # --- Compute per-frame MSE in time-domain RIR (shift+gate) ---
    mse_per_frame = np.zeros_like(t_sec, dtype=np.float64)
    center = WIN_LENGTH // 2
    support = F_L + F_R
    half = support // 2
    lo = center - half
    hi = lo + support

    for i, t in enumerate(frame_idx):
        H_true = A_true_use[mic, :, t]                                    # (Fpos,)
        H_hat  = A_hat_use[mic,  :, t]                                    # (Fpos,)
        _, g_true = rtf_time_shift_and_truncate(H_true, F_L, F_R, WIN_LENGTH)
        _, g_hat  = rtf_time_shift_and_truncate(H_hat,  F_L, F_R, WIN_LENGTH)
        norm_true =np.sqrt(np.sum(g_true**2))
        norm_hat =np.sqrt(np.sum(g_hat**2))
        g_true /= np.sqrt(np.sum(g_true**2))
        g_hat  /= np.sqrt(np.sum(g_hat**2))
        tmp= np.mean((g_true[lo:hi] - g_hat[lo:hi]) ** 2, dtype=np.float64)/(norm_hat ** 2)
        mse_per_frame[i] = 10 * np.log10(tmp + 1e-12)
    # --- Plot ---
    plt.figure(figsize=(8.5, 3.8), dpi=140)
    plt.plot(t_sec, mse_per_frame, lw=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel(f"RIR MSE (mic {mic} vs ref {REF_MIC})")
    plt.title("RTF Estimation Error over Time (PASTd vs TRUE)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180)
    print(f"[OK] Saved plot → {OUT_PNG}")

    # =====================================================
    # === 2) Example of the LAST FRAME's time-domain RIR ===
    # =====================================================
    H_true_last = A_true_use[mic, :, 400]
    H_hat_last  = A_hat_use[mic,  :, 400]
    _, g_true_last = rtf_time_shift_and_truncate(H_true_last, F_L, F_R, WIN_LENGTH)
    _, g_hat_last  = rtf_time_shift_and_truncate(H_hat_last,  F_L, F_R, WIN_LENGTH)

    t_ms = np.arange(WIN_LENGTH) / FS * 1e3  # x-axis in milliseconds

    plt.figure(figsize=(6, 3.5), dpi=140)
    plt.plot(t_ms, g_true_last, '-', lw=1.5, label='TRUE (shift+gate)')
    plt.plot(t_ms, g_hat_last,  '--', lw=1.5, label='PASTd (shift+gate)')
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Last Frame RTF (Mic {mic} vs Ref {REF_MIC})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_last = OUT_PNG.replace(".png", "_lastFrame.png")
    plt.savefig(out_last, dpi=180)
    print(f"[OK] Saved last-frame RIR comparison → {out_last}")



    snr_values = np.arange(-10, 41, 2)  # -10, -9, ..., 19, 20
    mean_mse_values = []

    for snr in snr_values:
        print(f"Evaluating SNR = {snr:.1f} dB ...")
        mean_mse = evaluate_single_snr(snr)
        mean_mse_values.append(mean_mse)
        print(f"  → Mean MSE = {mean_mse:.4e}")

    # --- Plot SNR vs Mean MSE ---
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(snr_values, mean_mse_values, 'o-', lw=1.8, markersize=4)  # smaller circle)
    plt.xlabel("Babble SNR (dB)")
    plt.ylabel("Mean RTF MSE (averaged over time)")
    plt.title(f"RTF Estimation: Accuracy vs SNR ")
    plt.grid(alpha=0.4)
    plt.yscale("log")
    plt.tight_layout()

    out_snr_plot = os.path.join(OUT_DIR, f"rtf_snr_vs_mse_idx{INDEX}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_snr_plot, dpi=180)
    print(f"[OK] Saved SNR–MSE plot → {out_snr_plot}")
if __name__ == "__main__":
    main()
