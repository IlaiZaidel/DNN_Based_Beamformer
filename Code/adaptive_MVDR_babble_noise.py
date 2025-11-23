#!/usr/bin/env python3
# Time-varying MVDR using PASTd RTF (a_hat) on saved dataset (no-args)

import os
import numpy as np
import scipy.io as sio
import soundfile as sf
import torch

# --- Your project utilities ---
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking
from RTF_covariance_whitening import noise_estimation
from utils import Postprocessing  # ISTFT wrapper

# ====================== CONFIG ======================
MAT_FILE   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/05_10_2025/TEST_STFT_domain_results_05_10_2025__09_15_43_0.mat"
OUT_DIR    = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/mvdr_pastd"
INDEX      = 5           # batch example to process
REF_MIC    = 3           # mixture reference mic to export
WIN_LENGTH = 512
HOP        = WIN_LENGTH // 4
NOISE_HEAD_SEC = 0.5     # same as your dataset (used to derive Ln)
BETA_PAST  = 0.95       # PASTd forgetting factor
EPS_DIAG   = 0        # diagonal loading for Rnn
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ====================================================

os.makedirs(OUT_DIR, exist_ok=True)

def to_torch_complex(arr, device):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        t = torch.from_numpy(arr)
    else:
        t = torch.from_numpy(arr.astype(np.complex64))
    return t.to(device)

# ---------- Load saved STFTs ----------
mat = sio.loadmat(MAT_FILE)
if "X_STFT" not in mat or "Y_STFT" not in mat:
    raise RuntimeError("MAT must contain 'X_STFT' (clean) and 'Y_STFT' (mixture).")

fs = int(np.array(mat.get("fs", 16000)).squeeze())

X_all = to_torch_complex(mat["X_STFT"], DEVICE)   # [B,M,F,L]
Y_all = to_torch_complex(mat["Y_STFT"], DEVICE)   # [B,M,F,L]

if X_all.dim() != 4 or Y_all.dim() != 4:
    raise RuntimeError(f"Expected [B,M,F,L]; got X:{tuple(X_all.shape)}, Y:{tuple(Y_all.shape)}")

B, M, F, L = Y_all.shape
if not (0 <= INDEX < B):
    raise IndexError(f"INDEX={INDEX} out of range [0..{B-1}]")
if not (0 <= REF_MIC < M):
    raise IndexError(f"REF_MIC={REF_MIC} out of range [0..{M-1}]")

# Slice the example
X = X_all[INDEX:INDEX+1, :, :, :]    # [1,M,F,L] clean
Y = Y_all[INDEX:INDEX+1, :, :, :]    # [1,M,F,L] mixture
N = Y - X                             # [1,M,F,L] babble

# Derive Ln from STFT geometry
# L ≈ floor((T*fs - win)/hop) + 1 ; but we only need Ln that matches your dataset head
Ln = int(NOISE_HEAD_SEC * fs // HOP)  # number of STFT frames in the noise-only head
Ln = max(0, min(Ln, L))               # clamp

# ---------- PASTd subspace tracking on mixture ----------
# pastd_rank1_whitened expects complex STFT [B,M,F,L]
W_past, eigvals, eigvecs = pastd_rank1_whitened(Y, NOISE_HEAD_SEC, beta=BETA_PAST)
# a_hat shape: [B, F, M, L-Ln]  (per your function)
a_hat = rtf_from_subspace_tracking(W_past, eigvals, eigvecs, NOISE_HEAD_SEC)

if a_hat.dim() != 4 or a_hat.shape[0] != 1 or a_hat.shape[1] != F or a_hat.shape[2] != M:
    raise RuntimeError(f"Unexpected a_hat shape {tuple(a_hat.shape)}; expected [1,F,M,L-Ln].")
T_hat = a_hat.shape[-1]  # should be L - Ln (typically)

# ---------- Noise covariance Rnn from noise-only head ----------
# Use your provided estimator; returns [B,F,M,M]
Rnn = noise_estimation(Y, Ln)  # [1,F,M,M]
# Diagonal loading for stability
eye = torch.eye(M, dtype=Rnn.dtype, device=Rnn.device).view(1, 1, M, M)
Rnn = Rnn + EPS_DIAG * eye
Rnn_inv = torch.linalg.inv(Rnn)  # [1,F,M,M]

# ---------- Build time-varying MVDR weights W(f,t) ----------
# W_mvdr_time: [1,F,M,T_hat]
W_mvdr_time = torch.zeros((1, F, M, T_hat), dtype=torch.cfloat, device=DEVICE)

# Vectorized over frequency: do it frame-by-frame to keep code clear
for t in range(T_hat):
    a_t = a_hat[:, :, :, t]                        # [1,F,M]
    a_t_col = a_t.unsqueeze(-1)                    # [1,F,M,1]
    Ra = torch.matmul(Rnn_inv, a_t_col)            # [1,F,M,1]
    denom = torch.matmul(a_t.conj().unsqueeze(-2), Ra).squeeze(-1).squeeze(-1)  # [1,F]
    denom = denom + 1e-12
    W_t = (Ra / denom.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)  # [1,F,M]
    W_mvdr_time[:, :, :, t] = W_t

# ---------- Pad the first Ln frames with zeros to match L ----------
if Ln + T_hat < L:
    pad_needed = L - (Ln + T_hat)
    # rare, but if exists, pad zeros at the end as well
else:
    pad_needed = 0

zeros_head = torch.zeros((1, F, M, Ln),     dtype=torch.cfloat, device=DEVICE)
zeros_tail = torch.zeros((1, F, M, pad_needed), dtype=torch.cfloat, device=DEVICE)
W_mvdr_tv = torch.cat([zeros_head, W_mvdr_time, zeros_tail], dim=3)  # [1,F, M, L]
W_mvdr_tv = W_mvdr_tv[..., :L]  # exact length guard

# ---------- Apply time-varying MVDR to mixture ----------
# Align mixture to [1,F,M,L]
Y_fml = Y.permute(0, 2, 1, 3).contiguous()  # [1,F,M,L]
# Z(f,t) = sum_m W^*(f,t,m) * Y(f,t,m)
Z = torch.sum(W_mvdr_tv.conj() * Y_fml, dim=2)  # [1,F,L]

# ---------- Reconstruct time-domain ----------
y_ref_time = Postprocessing(Y[:, REF_MIC, :, :], HOP, WIN_LENGTH, DEVICE)  # [1,N]
z_time     = Postprocessing(Z,                       HOP, WIN_LENGTH, DEVICE)  # [1,N]

# Align for saving
Nmin = min(y_ref_time.shape[-1], z_time.shape[-1])
y_ref_time = y_ref_time[..., :Nmin]
z_time     = z_time    [..., :Nmin]

base = f"mvdr_pastd_idx{INDEX}_fs{fs}"

# ---------- Save noise track ----------
wav_noise = os.path.join(OUT_DIR, f"{base}_noise_refmic{REF_MIC}.wav")
# N: [1, M, F, L] → take reference mic and ISTFT back
N_ref = N[:, REF_MIC, :, :]  # [1, F, L]
n_time = Postprocessing(N_ref, HOP, WIN_LENGTH, DEVICE)  # [1, time]
n_time = n_time[..., :Nmin]  # align lengths
sf.write(wav_noise, n_time.squeeze(0).detach().cpu().numpy(), fs)

# ---------- Save WAVs ----------
base = f"mvdr_pastd_idx{INDEX}_fs{fs}"
wav_mix = os.path.join(OUT_DIR, f"{base}_mixture_refmic{REF_MIC}.wav")
wav_bf  = os.path.join(OUT_DIR, f"{base}_mvdr_tv.wav")

sf.write(wav_mix, y_ref_time.squeeze(0).detach().cpu().numpy(), fs)
sf.write(wav_bf,  z_time.squeeze(0).detach().cpu().numpy(), fs)

print("[OK] Wrote:")
print("  Mixture (ref mic):", wav_mix)
print("  MVDR (time-varying):", wav_bf)
print(f"Frames: L={L}, noise-head Ln={Ln}, a_hat frames={T_hat}, padded={W_mvdr_tv.shape[-1]}")



# ================== Beampattern + Spectrogram from W_mvdr_tv ==================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd

from utils import Preprocesing, Postprocessing, return_as_complex

# --- Config (adjust paths if needed) ---
SURROUND_DIR  = "/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
OUT_ROOT      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gifs"
FRAME_STRIDE  = 10
MAX_FRAMES    = 60
VIDEO_FPS     = 8
SPEC_DB_MIN   = -60.0
SPEC_DB_MAX   = 0.0
BP_DB_MIN     = -20.0
BP_DB_MAX     = 0.0

# Optional: overlay a known interferer angle from your CSV (comment out if not needed)
try:
    CSV_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
    angle_n1 = int(pd.read_csv(CSV_FILE).iloc[INDEX]["angle_n1"])
except Exception:
    angle_n1 = None

# Output subdir by date folder (same style as your other tool)
model_dir = os.path.basename(os.path.dirname(MAT_FILE))  # e.g., "05_10_2025"
out_dir = os.path.join(OUT_ROOT, f"{model_dir}_mvdr")
os.makedirs(out_dir, exist_ok=True)

# Ensure weights are [1, M, F, L] to match probe STFT multiplication
Wm = W_mvdr_tv.permute(0, 2, 1, 3).contiguous()  # [1, M, F, L]
_, M, F, Lw = Wm.shape

# Frames to render
frames = list(range(0, Lw, max(1, FRAME_STRIDE)))
if MAX_FRAMES is not None:
    frames = frames[:MAX_FRAMES]

# Polar geometry
angles = np.arange(181)                    # 0..180 deg
ang_rad = np.deg2rad(angles)
mapped  = np.pi/2 - ang_rad                # map to polar (0deg = North)

# Preload surround probes once (as time series → STFT complex for each DOA)
surround_stfts = []
for doa in range(181):
    m = sio.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
    feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(DEVICE)   # [N, M] time-domain probes
    feat = feat.unsqueeze(0)  # [1, N, M]
    Yp   = Preprocesing(feat, WIN_LENGTH, fs, feat.shape[1]/fs, HOP, DEVICE)  # [1,M,2F,Lt]
    Ypc  = return_as_complex(Yp).contiguous()                                  # [1,M,F,Lt]
    surround_stfts.append(Ypc)

frames_beam, frames_spec = [], []

for k, fidx in enumerate(frames):
    # Current frame weights: [1,M,F,1]
    Wf = Wm[:, :, :, fidx:fidx+1]

    pow_list = []
    spec_acc = torch.zeros((F, 181), dtype=torch.float32, device=DEVICE)  # [F, DOA]

    for doa in range(181):
        Yc = surround_stfts[doa]                    # [1,M,F,Lt]
        Z  = torch.sum(torch.conj(Wf) * Yc, dim=1)  # [1,F,Lt]
        zt = Postprocessing(Z, HOP, WIN_LENGTH, DEVICE)  # [1, Nt]

        # scalar power for the polar beampattern
        pow_list.append(torch.sum(torch.abs(zt)**2))

        # freq-wise (time-avg) beam power for the 2D "spectrogram"
        spec_acc[:, doa] = torch.mean(torch.abs(Z)**2, dim=-1).squeeze(0)

    # ---- normalize & clamp for polar plot ----
    a = torch.stack(pow_list)  # [181]
    a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a) + 1e-12), min=1e-12)).detach().cpu().numpy()
    a_db = np.clip(a_db, BP_DB_MIN, BP_DB_MAX)
    r = a_db - BP_DB_MIN

    # ---- beam-power "spectrogram" (F x DOA) ----
    spec_db = 10.0 * torch.log10(torch.clamp(spec_acc, min=1e-12)).detach().cpu().numpy()
    spec_db = np.clip(spec_db, SPEC_DB_MIN, SPEC_DB_MAX)

    # ----- render beampattern frame (polar) -----
    fig_bp, ax_bp = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.25, 5.0), dpi=128)
    ax_bp.plot(mapped, r)

    if angle_n1 is not None:
        ax_bp.plot([np.pi/2 - np.deg2rad(angle_n1)], [BP_DB_MAX - BP_DB_MIN], "bo", label="Noise 1")

    ax_bp.set_theta_zero_location('N'); ax_bp.set_theta_direction(-1)
    ax_bp.set_thetamin(-90); ax_bp.set_thetamax(90)
    ax_bp.set_rlim(0, BP_DB_MAX - BP_DB_MIN)
    rticks = np.arange(0, (BP_DB_MAX - BP_DB_MIN) + 0.1, 5)
    ax_bp.set_rticks(rticks)
    ax_bp.set_yticklabels([f"{BP_DB_MIN + t:.0f} dB" for t in rticks])
    ax_bp.set_rlabel_position(135)
    ax_bp.grid(True)
    ax_bp.set_title(f"MVDR Beampattern (frame {fidx}/{Lw-1})")
    fig_bp.tight_layout()
    fig_bp.canvas.draw()
    w_bp, h_bp = fig_bp.canvas.get_width_height()
    img_bp = np.frombuffer(fig_bp.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_bp, w_bp, 4))[:, :, :3]
    frames_beam.append(img_bp)
    plt.close(fig_bp)

    # ----- render beam-power spectrogram frame -----
    freqs = np.linspace(0.0, fs/2, F)
    fig_sp, ax_sp = plt.subplots(figsize=(6.25, 5.0), dpi=128)
    im = ax_sp.imshow(spec_db, aspect="auto", origin="lower",
                      extent=[0, 180, freqs[0], freqs[-1]],
                      vmin=SPEC_DB_MIN, vmax=SPEC_DB_MAX, cmap="viridis")
    cbar = fig_sp.colorbar(im, ax=ax_sp)
    cbar.set_label("Power (dB)")
    ax_sp.set_xlabel("DOA (deg)")
    ax_sp.set_ylabel("Frequency (Hz)")
    ax_sp.set_title(f"MVDR Beam-power (frame {fidx}/{Lw-1})")
    if angle_n1 is not None:
        ax_sp.axvline(x=angle_n1, color="b", linestyle="--", linewidth=1.5, label="Noise 1")
        ax_sp.legend(loc="upper right", fontsize=8)

    fig_sp.tight_layout()
    fig_sp.canvas.draw()
    w_sp, h_sp = fig_sp.canvas.get_width_height()
    img_sp = np.frombuffer(fig_sp.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_sp, w_sp, 4))[:, :, :3]
    frames_spec.append(img_sp)
    plt.close(fig_sp)
import time
# === write videos ===
stamp = time.strftime("%Y%m%d_%H%M%S")
mp4_beam = os.path.join(out_dir, f"mvdr_beampattern_INDEX_{INDEX}_{stamp}.mp4")
mp4_spec = os.path.join(out_dir, f"mvdr_beam_spectrogram_INDEX_{INDEX}_{stamp}.mp4")

iio.imwrite(mp4_beam, frames_beam, fps=VIDEO_FPS, codec="libx264")
print(f"[OK] Beampattern MP4: {mp4_beam}")

iio.imwrite(mp4_spec, frames_spec, fps=VIDEO_FPS, codec="libx264")
print(f"[OK] Beam-power Spectrogram MP4: {mp4_spec}")
# ==============================================================================
