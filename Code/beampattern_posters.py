#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click pipeline (no CLI flags):
- Edit CONFIG + ANGLE_X_REL / ANGLE_N1_REL below and run.
- It will:
  1) Load CSV -> synthesize a 4s multichannel mixture via Habets RIRs (speech from CSV `speaker_path`)
  2) Run your pretrained ExNet-BF+PF model (Stage-1 weights -> complex W(k))
  3) Save WAVs for inputs and beamformed outputs
  4) Compute & save beampattern plots using your surround white-noise bank
"""

import os, sys, ast, math, time
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io
from datetime import datetime

plt.rcParams.update({
    "font.size": 16,          # base font
    "axes.titlesize": 20,     # ax.set_title(...)
    "axes.labelsize": 18,     # x/y labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,   # plt.suptitle(...)
})
# ------------------ MANUAL RELATIVE ANGLES (degrees, −90..+90) ------------------
# 0 = in front (broadside); + = to the array's RIGHT; − = to the LEFT
ANGLE_X_REL  = 0     # Speaker relative DOA (−90..+90)
ANGLE_N1_REL = -30    # Directional noise relative DOA (−90..+90)
SOURCE_RADIUS_M = 1.2  # radial distance from mic-center for both sources
# -------------------------------------------------------------------------------

MY_MODEL_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Trained_Models/17_05_2025/trained_model_dataset_withoutRev.pt"
# MY_MODEL_PATH ="/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Models/trained_model_dataset_withoutRev.pt"
# ======== CONFIG (EDIT HERE) ========
CONFIG = {
    # Input
    "CSV_PATH": "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv",
    "CSV_INDEX": 0,

    # Model
    "CFG_PATH": "DNN_Based_Beamformer/Code/conf/config.yaml",
    "MODEL_PATH": "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Models/trained_model_dataset_withoutRev.pt",

    # Output
    "OUT_DIR": "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Test_Model_Files/one_click_runs",

    # Audio / STFT
    "FS": 16000,
    "T_SEC": 4.0,
    "WIN_LEN": 512,
    "HOP_LEN": 512 // 4,

    # Array / scene
    "MIC_REF_1B": 4,          # ref mic index (1-based)
    "NOISE_HEAD_S": 0.5,      # seconds of noise-only head
    "SNR_NOISE_DB": 3.0,      # directional colored noise SNR (vs clean @ ref)
    "SNR_WHITE_DB": 30.0,     # white sensor noise SNR (vs clean @ ref)
    "AR_COEF": -0.7,          # AR(1) coefficient for colored noise
    "C_K": 343.0,             # speed of sound (m/s)

    # Beampattern settings
    "W_PICK_FRAME": None,     # None -> average across time; or int frame index
    "NORMALIZE_WAVS": True,

    # Surround bank path (angle_0..180.mat files)
    "SURROUND_BASE": "/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround/",
}
# ====================================

# ---- Project imports ----
from utils import Preprocesing, Postprocessing, return_as_complex
from rir_generator import generate as rir_generate
from omegaconf import OmegaConf
from old_ExNetBFPFModel import old_ExNetBFPF
from old_UnetModel import UNET  # if your old model references it


# ===================== Surround-based Beampattern =====================
@torch.no_grad()
def compute_beampattern_from_surround_bank(
    W_k,                          # complex, [1,M,F,1] or [M,F,1]
    base_path,
    win_len=512,
    hop_len=None,
    sample_rate=16000,
    device=None
):
    """
    Sweep 0..180 deg using your surround features:
      - loads my_surround_feature_vector_angle_{i}.mat
      - STFT -> apply fixed W_k -> ISTFT -> power
      - accumulates |Z(f,l)|^2 per freq for DOA–freq heatmap
    """
    if hop_len is None:
        hop_len = win_len // 4

    if device is None:
        device = W_k.device if torch.is_tensor(W_k) else torch.device("cpu")
    if not torch.is_tensor(W_k):
        raise ValueError("W_k must be a complex torch tensor.")
    if W_k.dim() == 3:
        W_k = W_k.unsqueeze(0)        # -> [1,M,F,1]
    if W_k.size(-1) != 1:
        raise ValueError(f"W_k last dim must be 1 (time-fixed). Got {tuple(W_k.shape)}")
    if not torch.is_complex(W_k):
        raise ValueError("W_k must be complex (convert 2F real/imag to complex first).")

    W_k = W_k.to(device)

    # infer F from first file
    first_mat = scipy.io.loadmat(os.path.join(base_path, f"my_surround_feature_vector_angle_0.mat"))
    feat0 = torch.from_numpy(first_mat["feature"]).float().to(device).unsqueeze(0)   # [1, T] or [1, C, T]
    Y0 = Preprocesing(feat0, win_len, sample_rate, 4, hop_len, device)               # [1, C, 2F, L]
    Y0 = return_as_complex(Y0)                                                       # [1, C, F,  L]
    F = Y0.shape[2]
    del feat0, Y0

    amplitudes_sq  = torch.zeros(1, 181, device=device)
    amplitudes_spec = torch.zeros(181, F, device=device)

    for i in range(181):
        data = scipy.io.loadmat(os.path.join(base_path, f"my_surround_feature_vector_angle_{i}.mat"))
        feature_vector = torch.from_numpy(data["feature"]).float().to(device).unsqueeze(0)  # [1,T] or [1,C,T]

        Y = Preprocesing(feature_vector, win_len, sample_rate, 4, hop_len, device)  # [1, C, 2F, L]
        Y = return_as_complex(Y)                                                    # [1, C, F,  L]

        # Match mic channels with W_k
        B, My, Fbank, L = Y.shape
        Mw = W_k.shape[1]
        if My == Mw:
            Ym = Y
        elif My == 1 and Mw > 1:
            Ym = Y.repeat(1, Mw, 1, 1)
        else:
            raise ValueError(f"Channel mismatch: Y has {My} mics but W_k expects {Mw}.")

        wy = torch.conj(W_k) * Ym                   # [1, M, F, L]
        Z_STFT = torch.sum(wy, dim=1)               # [1, F, L]

        z = Postprocessing(Z_STFT, hop_len, win_len, device)  # [1, T]
        amplitudes_sq[0, i] = torch.sum(torch.abs(z) ** 2)

        Z_energy_f = torch.sum(torch.abs(Z_STFT) ** 2, dim=-1).squeeze(0)  # [F]
        amplitudes_spec[i, :] = Z_energy_f

    # dB scaling
    maximum = torch.max(amplitudes_sq).clamp_min(1e-12)
    amplitudes_sq_db  = 10.0 * torch.log10((amplitudes_sq / maximum).clamp_min(1e-12))
    amplitudes_spec_db = 10.0 * torch.log10(amplitudes_spec.clamp_min(1e-12))
    return amplitudes_sq_db, amplitudes_spec_db


def plot_beampattern_from_surround(
    amplitudes_sq_db, angle_x_rel, angle_n1_rel, index, output_dir,
    title_prefix="Beamforming Beampattern"
):
    """
    Plot 'as is':
      - angles: -90 (left) .. 0 (front) .. +90 (right)
      - NO rotation/reindexing
      - radial (dB) limits fixed to [-20, 0]
    """
    import os, numpy as np, torch, matplotlib.pyplot as plt

    # data: 181 samples, assumed to correspond to -90..+90 in order
    rel_angles = np.arange(-90, 91)                                   # [-90..+90]
    amps = amplitudes_sq_db.squeeze(0).detach().cpu().numpy()         # (181,)

    # correct theta mapping for this axis config:
    #   theta = rel_angle (deg) because 0° is at 'N' and clockwise is positive
    theta = np.deg2rad(rel_angles.astype(float))

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 6))

    # axis: front hemisphere, right=+90, left=-90
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    # fixed dB range
    ax.set_ylim([-15, 0.0])

    # curve
    ax.plot(theta, amps, linewidth=2, label="Power (dB)")
    real_x_angle = -angle_x_rel
    real_n_angle = -angle_n1_rel
    # markers at outer ring (0 dB) — USE θ = angle (deg) directly
    r_mark = 0.0
    ax.plot([np.deg2rad(-angle_x_rel)],  [r_mark], 'ro', label=f"Speaker {real_x_angle}°")
    ax.plot([np.deg2rad(-angle_n1_rel)], [r_mark], 'bo', label=f"Noise {real_n_angle}°")
    # 3) Above the plot, centered
#ValueError: 'bottom right' is not a valid value for loc; supported values are 'best', 'upper right', 
# 'upper left', 'lower left', 'lower right', 'right',
#  'center left', 'center right', 'lower center', 'upper center', 'center'
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=13)
    # ax.legend(loc='upper right')
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  # move below the circle
        ncol=3,
        frameon=True
    )
    ax.set_title("Beampower vs DOA, MWF Loss", fontsize=20, y=0.87)

    out_png = os.path.join(
        output_dir, f"beampattern_index_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(out_png, dpi=300); plt.close()
    print(f"Beampattern plot saved to {out_png}")
    return out_png





def plot_doa_frequency_from_surround(
    amplitudes_spec_db, angle_x_rel=None, angle_n1_rel=None, index=0, output_dir=".", sample_rate=16000, title_prefix="Spectrogram: Frequency vs DOA"
):
    """
    X-axis shown as −90..+90. Internally amplitudes_spec_db is [181,F] for sweep 0..180,
    so we just display with extent −90..+90.
    """
    rel_angles = np.arange(-90, 91)   # 181 points
    F = amplitudes_spec_db.shape[1]
    freqs = np.linspace(0, sample_rate / 2, F)
    A = amplitudes_spec_db.detach().cpu().numpy().T  # [F, 181]

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    plt.imshow(A, aspect="auto",
               extent=[rel_angles.min(), rel_angles.max(), freqs.min(), freqs.max()],
               origin="lower", cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.xlabel("DOA (degrees, relative)", fontsize=20); plt.ylabel("Frequency (Hz)", fontsize=20)
    plt.title("Beampower: Frequency vs DOA, MWF Loss", fontsize=30, pad = 14)
    real_x_angle = -angle_x_rel
    real_n_angle = -angle_n1_rel
    if angle_x_rel is not None:
        plt.axvline(x=-angle_x_rel, color='r', linestyle='--', linewidth=4, label=f"Speaker {real_x_angle}°")
    if angle_n1_rel is not None:
        plt.axvline(x=-angle_n1_rel, color='b', linestyle='--', linewidth=4, label=f"Noise {real_n_angle}°")
    if (angle_x_rel is not None) or (angle_n1_rel is not None):
        plt.legend(loc='upper right', fontsize=20)

    out_png = os.path.join(output_dir, f"spectrogram_index_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(out_png, dpi=300); plt.close()
    print(f"Spectrogram plot saved to {out_png}")
    return out_png


# ===================== Model loader (always old class) =====================
def _strip_module(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def loadPreTrainedModel(cfg):
    PATH = MY_MODEL_PATH
    # Try weights-only first (PyTorch 2.6 default)
    try:
        sd = torch.load(PATH, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise RuntimeError("Not a state_dict")
    except Exception:
        # Fallback: old full pickled model -> extract state_dict ONLY
        from torch.serialization import add_safe_globals
        try:
            from ExNetBFPFModel import ExNetBFPF
            add_safe_globals([ExNetBFPF])
        except Exception:
            pass
        obj = torch.load(PATH, map_location="cpu", weights_only=False)
        sd = obj.state_dict()
        del obj
    sd = _strip_module(sd)

    model = old_ExNetBFPF(cfg.modelParams)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("load_state_dict -> missing:", missing, " unexpected:", unexpected)
    return model

def stamp_now():
    return time.strftime("%Y%m%d_%H%M%S")

def to_stereo(arr):
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.stack([arr[:, 0], arr[:, -1]], axis=-1).astype(np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    return arr.astype(np.float32)


# ===================== CSV + scene (speaker/noise positioned by RELATIVE ANGLES) =====================
def read_scene_from_csv(CFG, fs_expected):
    df = pd.read_csv(CFG["CSV_PATH"])
    if CFG["CSV_INDEX"] < 0 or CFG["CSV_INDEX"] >= len(df):
        raise IndexError(f"CSV index {CFG['CSV_INDEX']} out of range [0,{len(df)-1}]")
    row = df.iloc[CFG["CSV_INDEX"]]

    wav_path = row["speaker_path"]
    mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=float)  # (M,3)
    room_dim = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
    beta_val = float(row["beta"]); beta6 = [beta_val]*6
    n_taps   = int(row.get("n", max(256, int(CFG["FS"]*0.4))))

    # Compute positions from relative angles (−90..+90) in array-local frame
    sp_pos = _pos_from_relative_angle(ANGLE_X_REL,  mic_positions, SOURCE_RADIUS_M)
    n1_pos = _pos_from_relative_angle(ANGLE_N1_REL, mic_positions, SOURCE_RADIUS_M)

    # speech
    x, fs_wav = sf.read(wav_path)
    if x.ndim > 1: x = x[:, 0]
    if fs_wav != fs_expected:
        raise RuntimeError(f"Sample-rate mismatch (wav:{fs_wav}, expected:{fs_expected})")
    N = int(CFG["T_SEC"]*fs_expected)
    if len(x) < N:
        reps = int(np.ceil(N/len(x)))
        x = np.tile(x, reps)
    x = x[:N].astype(np.float64)

    return x, mic_positions, room_dim, beta6, n_taps, sp_pos, n1_pos


def synthesize_scene(CFG, x, mic_positions, room_dim, beta6, n_taps, sp_pos, n1_pos):
    FS = CFG["FS"]; NOISE_HEAD_S = CFG["NOISE_HEAD_S"]; AR_COEF = CFG["AR_COEF"]
    SNR_NOISE_DB = CFG["SNR_NOISE_DB"]; SNR_WHITE_DB = CFG["SNR_WHITE_DB"]
    C_K = CFG["C_K"]; MIC_REF_1B = CFG["MIC_REF_1B"]

    M = mic_positions.shape[0]; N = len(x)
    h_s = np.array(rir_generate(C_K, FS, mic_positions, sp_pos,  room_dim, beta6, n_taps, order=0)).T
    xM  = np.tile(x[:, None], (1, M))
    d_full = np.stack([signal.fftconvolve(xM[:, m], h_s[m], mode="full")[:N] for m in range(M)], axis=1)

    nlead = int(NOISE_HEAD_S*FS); keep = N - nlead
    d = np.vstack([np.zeros((nlead, M)), d_full[:keep, :]])

    h_n = np.array(rir_generate(C_K, FS, mic_positions, n1_pos, room_dim, beta6, n_taps, order=0)).T
    w = np.random.randn(N)
    drv = signal.lfilter([1.0], [1.0, -AR_COEF], w)
    n1 = np.stack([signal.lfilter(h_n[m], [1.0], drv) for m in range(M)], axis=1)

    v = np.random.randn(N, M)

    ref = MIC_REF_1B - 1
    d_pow = np.sum(d[:, ref]**2) + 1e-12
    n_pow = np.sum(n1[:, ref]**2) + 1e-12
    v_pow = np.sum(v[:,  ref]**2) + 1e-12
    scale_n = math.sqrt(d_pow * 10**(-SNR_NOISE_DB/10.0) / n_pow)
    scale_v = math.sqrt(d_pow * 10**(-SNR_WHITE_DB/10.0) / v_pow)
    n1 *= scale_n; v *= scale_v

    y = d + n1 + v
    maxY = np.max(np.abs(y)) + 1e-12
    return y/maxY, d/maxY, n1/maxY, v/maxY


# --- helpers: axes from mic geometry and RELATIVE angle -> position mapping ---
def _array_axes_from_mics(mic_positions):
    """
    Returns:
      u_lr  : unit vector along the array's left->right axis in XY plane
      n_fwd : unit vector pointing 'forward' (broadside) in XY plane
      center: mic centroid (3,)
    """
    mp = np.asarray(mic_positions, dtype=float)
    center = mp.mean(axis=0)

    XY = mp[:, :2] - center[:2]
    U, S, Vt = np.linalg.svd(XY, full_matrices=False)
    lr2 = Vt[0]
    if lr2[0] < 0:
        lr2 = -lr2

    u_lr2  = lr2 / (np.linalg.norm(lr2) + 1e-12)           # left→right
    n_fwd2 = np.array([-u_lr2[1], u_lr2[0]])               # +90° CCW = "front" (broadside)

    u_lr  = np.array([u_lr2[0],  u_lr2[1],  0.0]);  u_lr  /= (np.linalg.norm(u_lr)  + 1e-12)
    n_fwd = np.array([n_fwd2[0], n_fwd2[1], 0.0]);  n_fwd /= (np.linalg.norm(n_fwd) + 1e-12)
    return u_lr, n_fwd, center

def _pos_from_relative_angle(deg_rel, mic_positions, radius_m):
    """
    deg_rel in [-90, +90]: 0 = front (broadside), + = to the RIGHT side of array.
    """
    u_lr, n_fwd, center = _array_axes_from_mics(mic_positions)
    rad = np.deg2rad(np.clip(deg_rel, -90.0, 90.0))
    dir_vec = np.cos(rad) * n_fwd + np.sin(rad) * u_lr
    pos = center + radius_m * dir_vec
    pos[2] = center[2]
    return pos


# ===================== Model run =====================
def load_model_and_device(CFG):
    cfg = OmegaConf.load(CFG["CFG_PATH"])
    try:
        if CFG["MODEL_PATH"]:
            cfg.paths.model_path = CFG["MODEL_PATH"]
    except Exception:
        pass
    device = torch.device(f"cuda:{getattr(cfg.device, 'device_num', 0)}" if torch.cuda.is_available() else "cpu")
    model = loadPreTrainedModel(cfg).to(device).eval()
    return model, device, cfg

@torch.no_grad()
def model_forward_weights(CFG, model, device, y_time):
    y_t = torch.from_numpy(y_time[None, ...]).float().to(device)
    Y   = Preprocesing(y_t, CFG["WIN_LEN"], CFG["FS"], CFG["T_SEC"], CFG["HOP_LEN"], device)
    outs = model(Y, device)
    W = return_as_complex(outs[0])              # (1,M,F,L)
    return W

def pick_time_invariant_W(W, frame_idx=None):
    if frame_idx is None:
        return W.mean(dim=-1, keepdim=True)
    L = W.shape[-1]
    f = max(0, min(L-1, int(frame_idx)))
    return W[..., f:f+1]

@torch.no_grad()
def apply_beamformer(CFG, W_k, x_time, device):
    x_t = torch.from_numpy(x_time[None, ...]).float().to(device)
    X   = Preprocesing(x_t, CFG["WIN_LEN"], CFG["FS"], CFG["T_SEC"], CFG["HOP_LEN"], device)
    Xc  = return_as_complex(X)
    Z   = torch.sum(torch.conj(W_k) * Xc, dim=1)
    xh  = Postprocessing(Z, CFG["HOP_LEN"], CFG["WIN_LEN"], device).cpu().numpy()[0]
    return xh


# ===================== Main =====================
def main():
    CFG = dict(CONFIG)
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    run_dir = os.path.join(CFG["OUT_DIR"], f"run_{stamp_now()}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) CSV -> scene (positions from ANGLE_*_REL)
    x, Rpos, L, beta6, n_taps, sp_pos, n1_pos = read_scene_from_csv(CFG, CFG["FS"])
    y, d, n1, v = synthesize_scene(CFG, x, Rpos, L, beta6, n_taps, sp_pos, n1_pos)

    # These are already relative angles (−90..+90)
    angle_src_rel   = ANGLE_X_REL
    angle_noise_rel = ANGLE_N1_REL

    # 2) Load model & get weights
    model, device, _ = load_model_and_device(CFG)
    W = model_forward_weights(CFG, model, device, y)
    Wk = pick_time_invariant_W(W, frame_idx=CFG["W_PICK_FRAME"])  # (1,M,F,1)

    # 3) Apply beamformer & save WAVs
    xhat_mix   = apply_beamformer(CFG, Wk, y, device)
    xhat_clean = apply_beamformer(CFG, Wk, d, device)
    xhat_dir   = apply_beamformer(CFG, Wk, n1, device)
    xhat_white = apply_beamformer(CFG, Wk, v, device)

    def maybe_norm(sig):
        if CFG["NORMALIZE_WAVS"]:
            m = np.max(np.abs(sig)) + 1e-12
            return (sig / m).astype(np.float32)
        return sig.astype(np.float32)

    base = os.path.join(run_dir, "ex")
    sf.write(base + "_mix.wav",   to_stereo(y),  CFG["FS"])
    sf.write(base + "_clean.wav", to_stereo(d),  CFG["FS"])
    sf.write(base + "_dir.wav",   to_stereo(n1), CFG["FS"])
    sf.write(base + "_white.wav", to_stereo(v),  CFG["FS"])

    sf.write(base + "_mix_beam.wav",   maybe_norm(xhat_mix),   CFG["FS"])
    sf.write(base + "_clean_beam.wav", maybe_norm(xhat_clean), CFG["FS"])
    sf.write(base + "_dir_beam.wav",   maybe_norm(xhat_dir),   CFG["FS"])
    sf.write(base + "_white_beam.wav", maybe_norm(xhat_white), CFG["FS"])

    # 4) Beampattern via surround white-noise bank
    amps_db, amps_spec_db = compute_beampattern_from_surround_bank(
        Wk,
        base_path=CFG["SURROUND_BASE"],
        win_len=CFG["WIN_LEN"],
        hop_len=CFG["HOP_LEN"],
        sample_rate=CFG["FS"],
        device=device
    )

    polar_png = plot_beampattern_from_surround(
        amplitudes_sq_db=amps_db,
        angle_x_rel=angle_src_rel,
        angle_n1_rel=angle_noise_rel,
        index=CFG["CSV_INDEX"],
        output_dir=run_dir,
        title_prefix="Beamforming Beampattern"
    )

    spec_png = plot_doa_frequency_from_surround(
        amplitudes_spec_db=amps_spec_db,
        angle_x_rel=angle_src_rel,
        angle_n1_rel=angle_noise_rel,
        index=CFG["CSV_INDEX"],
        output_dir=run_dir,
        sample_rate=CFG["FS"],
        title_prefix="Spectrogram: Frequency vs DOA"
    )

    print("Saved to:", run_dir)
    print("Polar beampattern:", polar_png)
    print("DOA–freq beampattern:", spec_png)

if __name__ == "__main__":
    main()
