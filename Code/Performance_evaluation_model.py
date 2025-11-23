#!/usr/bin/env python3
# Performance evaluation (SI-SDR + WAV export)

import os
import numpy as np
import scipy.io as sio
import torch
import soundfile as sf
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# ================== CONFIG ==================
MAT_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/15_10_2025_CORRELATION/TEST_time_domain_results_16_10_2025__02_48_36_0.mat"
# MAT_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/12_10_2025_MIX/TEST_time_domain_results_12_10_2025__08_59_42_0.mat"
CLEAN_DIR    = "/dsi/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen_with_rir"
CLEAN_PREFIX = "clean_example_"
OUT_ROOT     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Performance_evaluation"
INDEX        = 5
FS           = 16000
NOISE_HEAD_SEC = 0.5
NOISE_SAMPLES   = int(NOISE_HEAD_SEC * FS)
# ============================================


def si_sdr_torch(estimate: torch.Tensor,
                 reference: torch.Tensor,
                 eps: float = 1e-10,
                 zero_mean: bool = True,
                 reduction: str = "mean"):
    """
    Scale-Invariant SDR in dB (PyTorch, differentiable).
    Accepts (T,) or (B, T) tensors; returns:
      - scalar if reduction='mean'
      - tensor of shape (B,) if reduction='none'

    estimate: torch.Tensor
    reference: torch.Tensor
    """
    x = estimate
    s = reference
    if x.ndim == 1: x = x.unsqueeze(0)
    if s.ndim == 1: s = s.unsqueeze(0)
    if x.shape != s.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {s.shape}")
    if zero_mean:
        x = x - x.mean(dim=1, keepdim=True)
        s = s - s.mean(dim=1, keepdim=True)

    s_energy = torch.sum(s * s, dim=1, keepdim=True) + eps
    alpha = torch.sum(x * s, dim=1, keepdim=True) / s_energy
    s_target = alpha * s
    e_noise = x - s_target

    num = torch.sum(s_target * s_target, dim=1) + eps
    den = torch.sum(e_noise * e_noise, dim=1) + eps
    sisdr = 10.0 * torch.log10(num / den)

    if reduction == "mean":
        return sisdr.mean()
    elif reduction == "none":
        return sisdr
    else:
        raise ValueError("reduction must be 'mean' or 'none'")
    

    
import pesq
from pesq import pesq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Load MAT ====
mat = sio.loadmat(MAT_FILE)
if "fs" in mat:
    FS = int(np.array(mat["fs"]).squeeze())

# Directory naming from model path
model_dir = os.path.basename(os.path.dirname(MAT_FILE))
out_dir = os.path.join(OUT_ROOT, model_dir)
os.makedirs(out_dir, exist_ok=True)

# ==== Load signals ====
x_hat_stage1_left  = torch.from_numpy(mat["x_hat_stage1_left"])[INDEX, NOISE_SAMPLES:].to(device).float()
x_hat_stage1_right = torch.from_numpy(mat["x_hat_stage1_right"])[INDEX, NOISE_SAMPLES:].to(device).float()
y                  = torch.from_numpy(mat["y"])[INDEX, NOISE_SAMPLES:].to(device).float()

# ==== Load clean ====
clean_mat_path = os.path.join(CLEAN_DIR, f"{CLEAN_PREFIX}{INDEX:07d}.mat")
clean_mat = sio.loadmat(clean_mat_path)
clean = clean_mat["clean"]  # (T, 8)
clean_left  = torch.tensor(clean[:, 0], dtype=torch.float32, device=device)
clean_right = torch.tensor(clean[:, 7], dtype=torch.float32, device=device)




# ==== SI-SDR computation ====
si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
y_left = y[:,0]
y_right = y[:,7]
sisdr_in_left  = si_sdr_torch(y_left, clean_left)
sisdr_in_right = si_sdr_torch(y_right, clean_right)
sisdr_left     = si_sdr_torch(x_hat_stage1_left, clean_left)
sisdr_right    = si_sdr_torch(x_hat_stage1_right, clean_right)





print(f"[Model: {model_dir}] [Index {INDEX}]")
print(f"  Baseline SI-SDR (y vs clean L):  {sisdr_in_left:+.2f} dB")
print(f"  Model    SI-SDR (x̂L vs clean L): {sisdr_left:+.2f} dB  (Δ {sisdr_left - sisdr_in_left:+.2f} dB)")
print(f"  Baseline SI-SDR (y vs clean R):  {sisdr_in_right:+.2f} dB")
print(f"  Model    SI-SDR (x̂R vs clean R): {sisdr_right:+.2f} dB  (Δ {sisdr_right - sisdr_in_right:+.2f} dB)")



# PESQ between y / x_hat / clean
pesq_in_left   = pesq(FS, clean_left.cpu().numpy(),  y_left.cpu().numpy(),  'wb')
pesq_in_right  = pesq(FS, clean_right.cpu().numpy(), y_right.cpu().numpy(), 'wb')
pesq_left      = pesq(FS, clean_left.cpu().numpy(),  x_hat_stage1_left.cpu().numpy(),  'wb')
pesq_right     = pesq(FS, clean_right.cpu().numpy(), x_hat_stage1_right.cpu().numpy(), 'wb')

print(f"[Model: {model_dir}] [Index {INDEX}]")
print(f"PESQ L_in:  {pesq_in_left:.2f}")
print(f"PESQ L_out: {pesq_left:.2f}")
print(f"PESQ R_in:  {pesq_in_right:.2f}")
print(f"PESQ R_out: {pesq_right:.2f}")

# ==== WAV Saving (mono + stereo) ====
wav_dir = os.path.join(out_dir, f"SI_SDR_EVAL_INDEX_{INDEX:02d}")
os.makedirs(wav_dir, exist_ok=True)

def _to_np_f32(sig):
    if hasattr(sig, "detach"):
        sig = sig.detach().cpu()
    a = np.asarray(sig).squeeze().astype(np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:
        a = a / (peak + 1e-8)
    return np.ascontiguousarray(a)

def save_wav(path, sig, fs):
    a = _to_np_f32(sig)
    if a.ndim > 2 or a.size == 0:
        raise ValueError(f"Bad shape {a.shape} for {path}")
    sf.write(path, a, fs, subtype="PCM_16", format="WAV")

def save_stereo(path, left, right, fs):
    L = _to_np_f32(left).reshape(-1)
    R = _to_np_f32(right).reshape(-1)
    if L.shape[0] != R.shape[0]:
        raise ValueError(f"Stereo length mismatch: L={L.shape[0]}, R={R.shape[0]}")
    stereo = np.column_stack([L, R])
    sf.write(path, stereo, fs, subtype="PCM_16", format="WAV")

# --- Mono ---
save_wav(os.path.join(wav_dir, "y.wav"),            y, FS)
save_wav(os.path.join(wav_dir, "xhat_left.wav"),    x_hat_stage1_left,  FS)
save_wav(os.path.join(wav_dir, "xhat_right.wav"),   x_hat_stage1_right, FS)
save_wav(os.path.join(wav_dir, "clean_left.wav"),   clean_left, FS)
save_wav(os.path.join(wav_dir, "clean_right.wav"),  clean_right, FS)

# --- Stereo ---
save_stereo(os.path.join(wav_dir, "clean_stereo.wav"), clean_left,         clean_right,        FS)
save_stereo(os.path.join(wav_dir, "xhat_stereo.wav"),  x_hat_stage1_left,  x_hat_stage1_right, FS)
save_stereo(os.path.join(wav_dir, "y_stereo_dup.wav"), y, y, FS)  # duplicate mono mixture

print(f"\n[OK] WAVs (mono + stereo) saved in:\n  {wav_dir}")
