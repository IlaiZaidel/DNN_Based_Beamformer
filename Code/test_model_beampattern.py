#!/usr/bin/env python3
# --- minimal, single-file demo: randomize scene -> make mixture -> run model -> save WAVs ---

import os
import math
import time
import numpy as np
import soundfile as sf
import scipy.signal as signal
import torch
import scipy
# ----------------------- EDIT ME -----------------------
SPEECH_WAV   = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28241/5105-28241-0011.wav"
MODEL_PATH   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Trained_Models/19_08_2025/trained_model_dataset_withoutRev.pt"
OUT_DIR      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Beampattern_and_model_analysis"
# ---- Make unique subfolder for this run ----
stamp = time.strftime("%Y%m%d_%H%M%S")

# Extract model date from MODEL_PATH, e.g. "19_08_2025" -> "19_08"
model_folder = os.path.basename(os.path.dirname(MODEL_PATH))
model_date   = "_".join(model_folder.split("_")[:2])  # take first two parts

RUN_DIR = os.path.join(OUT_DIR, f"run_{model_date}_{stamp}")
os.makedirs(RUN_DIR, exist_ok=True)

FS           = 16000
T_SEC        = 4.0
CK           = 343.0                # speed of sound
WIN_LENGTH   = 512
HOP          = WIN_LENGTH // 4
MIC_REF_1B   = 4                    # 1-based index of ref mic for SNR calcs

USE_TWO_NOISES   = False             # if False -> only noise #1 is used
NOISE_HEAD_SEC   = 0.5              # prepend silence before clean speech
SNR_NOISE_DB     = 3.0              # colored noise(s) vs clean @ ref mic
SNR_WHITE_DB     = 30.0             # white vs clean @ ref mic

ROOM_Z        = 3.0
MICS          = 8
BETA_T60      = np.random.choice(np.arange(0.30, 0.55, 0.05))
N_TAPS        = int(FS * BETA_T60)  # rir length for rir_generator
RIR_ORDER     = 0                   # direct path only (change if wanted)
# -------------------------------------------------------
# ========== CFG LOADING (no main, no argparse) ==========
from omegaconf import OmegaConf
cfg = OmegaConf.load("DNN_Based_Beamformer/Code/conf/config.yaml")  # <- uses your existing Hydra YAML
# imports that depend on your project
from rir_generator import generate as rir_generate
from utils import Preprocesing, Postprocessing, return_as_complex
import matplotlib.pyplot as plt

# ---- surround features location + frame to analyze ----
SURROUND_DIR = "/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
FRAME_IDX    = 300  # pick a valid STFT frame index inside L

def compute_beampattern_amplitudes(W_time_fixed_complex, device, win_len, fs, T_sec, hop, mic_dim=1):
    """
    Evaluate static beampattern using surround white-noise scenes.
    W_time_fixed_complex: [1, M_or_1, F, 1] complex weights for a single time frame.
    Returns:
      amplitudes_sq_db: [1, 181] dB power vs DOA
      amplitudes_spec_db: [181, F] dB power per frequency vs DOA
    """
    amplitudes_sq   = torch.zeros(1, 181, device=device)
    amplitudes_spec = torch.zeros(181, W_time_fixed_complex.shape[2], device=device)

    for doa in range(181):
        mat = scipy.io.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
        feat = torch.from_numpy(mat["feature"]).float().to(device).unsqueeze(0)  # [1, N, M]
        Y = Preprocesing(feat, win_len, fs, T_sec, hop, device)                  # [1, M, 2F, L]
        Yc = return_as_complex(Y)                                                # [1, M, F, L]

        wy = torch.conj(W_time_fixed_complex) * Yc     # broadcast over time L
        Z  = torch.sum(wy, dim=mic_dim)                # [1, F, L]
        z  = Postprocessing(Z, hop, win_len, device)   # [1, N]
        amplitudes_sq[0, doa] = torch.sum(torch.abs(z) ** 2)

        # freq-wise energy (time-avg)
        Z_energy = torch.mean(torch.abs(Z) ** 2, dim=-1).squeeze(0)  # [F]
        amplitudes_spec[doa, :] = Z_energy

    # dB scaling
    max_pow = torch.max(amplitudes_sq)
    amplitudes_sq_db   = 10.0 * torch.log10(torch.clamp(amplitudes_sq / (max_pow + 1e-12), min=1e-12))
    amplitudes_spec_db = 10.0 * torch.log10(torch.clamp(amplitudes_spec, min=1e-12))
    return amplitudes_sq_db, amplitudes_spec_db


def plot_beampattern(amplitudes_sq_db, out_dir, angle_x=None, angle_n1=None, angle_n2=None, tag=""):
    angles  = torch.arange(181)
    a_np    = amplitudes_sq_db.squeeze(0).detach().cpu().numpy()
    ang_rad = np.deg2rad(angles.numpy())
    mapped  = np.pi/2 - ang_rad

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 6))
    ax.plot(mapped, a_np, label="Power (dB)")
    if angle_x  is not None: ax.plot([np.pi/2 - np.deg2rad(angle_x)],  [0], 'ro', label="Speaker")
    if angle_n1 is not None: ax.plot([np.pi/2 - np.deg2rad(angle_n1)], [0], 'bo', label="Noise 1")
    if angle_n2 is not None: ax.plot([np.pi/2 - np.deg2rad(angle_n2)], [0], 'go', label="Noise 2")

    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_ylim([-30, 1]); ax.legend(loc='upper right')
    plt.title("Beampattern (static weights)")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path  = os.path.join(out_dir, f"beampattern_{tag}_{stamp}.png")
    plt.savefig(path, dpi=300); plt.close()
    print("Saved beampattern:", path)


def plot_beam_spectrogram(amplitudes_spec_db, fs, out_dir, angle_x=None, angle_n1=None, angle_n2=None, tag=""):
    angles = np.arange(181)
    freqs  = np.linspace(0, fs/2, amplitudes_spec_db.shape[1])
    A      = amplitudes_spec_db.detach().cpu().numpy().T  # [F, 181]

    plt.figure(figsize=(12, 6))
    plt.imshow(A, aspect="auto",
               extent=[angles.min(), angles.max(), freqs.min(), freqs.max()],
               origin="lower", cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.xlabel("DOA (deg)"); plt.ylabel("Frequency (Hz)")
    plt.title("Beam-power Spectrogram (static weights)")
    for ang, color in [(angle_x, 'r'), (angle_n1, 'b'), (angle_n2, 'g')]:
        if ang is not None:
            plt.axvline(x=ang, color=color, linestyle='--', linewidth=1.5)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path  = os.path.join(out_dir, f"beam_spectrogram_{tag}_{stamp}.png")
    plt.savefig(path, dpi=300); plt.close()
    print("Saved beam spectrogram:", path)


# ---- helpers (local) ----
def randomize_noise_angles(src_angle_deg, min_sep=30):
    pool = [a for a in range(181) if abs(a - src_angle_deg) >= min_sep]
    a1 = np.random.choice(pool)
    pool = [a for a in pool if abs(a - a1) >= min_sep]
    a2 = np.random.choice(pool)
    return int(a1), int(a2)

# ---- 1) Randomize room + array + sources ----
np.random.seed(int(time.time()) & 0xFFFFFFFF)
os.makedirs(OUT_DIR, exist_ok=True)

x_lim = np.random.uniform(6.0, 9.0)
y_lim = np.random.uniform(6.0, 9.0)
L = [x_lim, y_lim, ROOM_Z]
beta6 = [BETA_T60]*6

angle_orientation = np.random.randint(-45, 46)
theta = np.deg2rad(angle_orientation)

mic_center_x = np.random.uniform(2.0, x_lim - 2.0)
mic_center_y = np.random.uniform(2.0, y_lim - 2.0)
mic_height   = np.random.uniform(1.0, 1.5)

# linear 8-mic array (0.34m aperture) before rotation
mic_offsets = np.array([[-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
                        [ 0.04, 0], [ 0.07, 0], [ 0.12, 0], [ 0.17, 0]])
Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)]])
mic_xy = mic_offsets @ Rmat.T + np.array([mic_center_x, mic_center_y])
mic_positions = np.column_stack([mic_xy, np.full((MICS,), mic_height)])

# speech source at random radius & angle about array center
radius = np.random.uniform(1.0, 1.5)
angle_src = np.random.randint(0, 181)  # (0..180) in front half-plane
sx = mic_center_x + radius*np.cos(np.deg2rad(angle_src + angle_orientation))
sy = mic_center_y + radius*np.sin(np.deg2rad(angle_src + angle_orientation))
sz = np.random.uniform(1.2, 1.8)
src_pos = [float(sx), float(sy), float(sz)]

# two noises (angles kept away from src and from each other)
angle_n1, angle_n2 = randomize_noise_angles(angle_src, min_sep=10)
n1x = mic_center_x + radius*np.cos(np.deg2rad(angle_n1 + angle_orientation))
n1y = mic_center_y + radius*np.sin(np.deg2rad(angle_n1 + angle_orientation))
n1z = np.random.uniform(1.2, 1.9)
noise1_pos = [float(n1x), float(n1y), float(n1z)]

n2x = mic_center_x + radius*np.cos(np.deg2rad(angle_n2 + angle_orientation))
n2y = mic_center_y + radius*np.sin(np.deg2rad(angle_n2 + angle_orientation))
n2z = np.random.uniform(1.2, 1.9)
noise2_pos = [float(n2x), float(n2y), float(n2z)]

# ---- 2) Load speech (mono), cut/tile to N ----
N = int(FS*T_SEC)
x, fs_file = sf.read(SPEECH_WAV)
if x.ndim > 1:
    x = x[:, 0]
if fs_file != FS:
    raise RuntimeError(f"Sample-rate mismatch: wav={fs_file}, expected {FS}")
if len(x) < N:
    reps = int(math.ceil(N/len(x)))
    x = np.tile(x, reps)
x = x[:N].astype(np.float64)

# ---- 3) RIRs: speech + noise1 + noise2 (order=0 here) ----
# shapes returned (Taps, M) -> we transpose to (M, taps)
h_speech = np.array(rir_generate(CK, FS, mic_positions, src_pos,   L, beta6, N_TAPS, order=RIR_ORDER)).T
h_n1     = np.array(rir_generate(CK, FS, mic_positions, noise1_pos,L, beta6, N_TAPS, order=RIR_ORDER)).T
h_n2     = np.array(rir_generate(CK, FS, mic_positions, noise2_pos,L, beta6, N_TAPS, order=RIR_ORDER)).T

# ---- 4) Direct speech per-mic via FFT-conv ----
# tile mono speech across mics for convenience during conv loop
xM = np.tile(x[:, None], (1, MICS))
d_full = np.stack(
    [signal.fftconvolve(xM[:, m], h_speech[m], mode="full")[:N] for m in range(MICS)],
    axis=1
)

# optional noise-only head
nlead = int(NOISE_HEAD_SEC*FS)
keep  = N - nlead
d = np.vstack([np.zeros((nlead, MICS), dtype=np.float64), d_full[:keep, :]])

# ---- 5) Colored noises (AR(1) -> per-mic filtering with noise RIRs) ----
w = np.random.randn(N, 2)
AR = [1.0, -0.7]  # low-pass-ish
drv1 = signal.lfilter([1.0], AR, w[:, 0])
drv2 = signal.lfilter([1.0], AR, w[:, 1])

n1 = np.stack([signal.lfilter(h_n1[m], [1.0], drv1) for m in range(MICS)], axis=1)
n2 = np.stack([signal.lfilter(h_n2[m], [1.0], drv2) for m in range(MICS)], axis=1)

# ---- 6) SNR scaling vs clean @ ref mic ----
ref = MIC_REF_1B - 1
d_pow = np.sum(d[:, ref]**2)
if USE_TWO_NOISES:
    n_pow = np.sum((n1[:, ref] + n2[:, ref])**2) + 1e-10
else:
    n_pow = np.sum(n1[:, ref]**2) + 1e-10
scale_n = np.sqrt(d_pow * 10**(-SNR_NOISE_DB/10.0) / n_pow)
n1 *= scale_n
n2 *= scale_n

# white noise
v = np.random.randn(N, MICS)
v_pow = np.sum(v[:, ref]**2) + 1e-10
scale_v = np.sqrt(d_pow * 10**(-SNR_WHITE_DB/10.0) / v_pow)
v *= scale_v

# ---- 7) Sum mixture ----
y = d + n1 + (n2 if USE_TWO_NOISES else 0) + v

# Normalize everything together by max |y|
peak = np.max(np.abs(y))

y  /= peak
d  /= peak
n1 /= peak
n2 /= peak
v  /= peak

# ---- 8) Run the model (assumes torch.save(model, path)) ----
device = torch.device(f"cuda:{cfg.device.device_num}" if torch.cuda.is_available() else "cpu")

# Load the model directly onto that device
from LoadPreTrainedModel import loadPreTrainedModel

device = torch.device(f"cuda:{cfg.device.device_num}" if torch.cuda.is_available() else "cpu")

# Load model using your existing loader
model = loadPreTrainedModel(cfg)

# Ensure parameters are on the correct device
model.to(device).eval()
# preprocessing to STFT (your utils)
y_t = torch.from_numpy(y[None, ...]).float().to(device)        # (1, N, M)
Y   = Preprocesing(y_t, WIN_LENGTH, FS, T_SEC, HOP, device)    # (B, M, 2F, L)

with torch.no_grad():
    # your model’s forward signature:
    # W_timeChange, X_hat_Stage1, Y_out, W_Stage1, X_hat_Stage2, W_Stage2, _, _
    outs = model(Y, device)
    X_hat_Stage1 = outs[1]
   

xhat1 = Postprocessing(X_hat_Stage1, HOP, WIN_LENGTH, device).cpu().numpy()[0]  # (N,)


def to_stereo(arr):
    # If multichannel (N, M) with M>=2 -> [first, last] channels
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.stack([arr[:, 0], arr[:, -1]], axis=-1).astype(np.float32)
    # If already mono (N,) or (N,1), return as-is
    return arr.astype(np.float32)

# ---- 9) Save WAVs ----
stamp = time.strftime("%Y%m%d_%H%M%S")
base = os.path.join(RUN_DIR, "ex")


# multichannel -> stereo (first & last); mono stays mono
sf.write(base + "_mix.wav",   to_stereo(y),  FS)
sf.write(base + "_clean.wav", to_stereo(d),  FS)
sf.write(base + "_n1.wav",    to_stereo(n1), FS)
if USE_TWO_NOISES:
    sf.write(base + "_n2.wav", to_stereo(n2), FS)
sf.write(base + "_white.wav", to_stereo(v),  FS)

# model outputs (mono already)
sf.write(base + "_xhat1.wav", xhat1.astype(np.float32), FS)

print("Saved to:", OUT_DIR)
print("Files prefix:", base)

# ---- Stage-1 weights only (outs[3]) -> complex -> pick a single time frame ----
W1 = outs[3]  # shape [B, M, 2F, L]

# convert stacked real/imag (2F) to complex [B, M, F, L]
B, M, F2, L = W1.shape
F = F2 // 2
# real, imag = torch.split(W1, F, dim=2)
W1c =W1 # [B, M, F, L]

# choose a valid frame
frame = min(FRAME_IDX, L - 1)
W1c_frame = W1c[:, :, :, frame:frame+1]  # [B, M, F, 1]

# compute beampatterns
ampl_sq_db, ampl_spec_db = compute_beampattern_amplitudes(
    W_time_fixed_complex=W1c_frame,
    device=device, win_len=WIN_LENGTH, fs=FS, T_sec=T_SEC, hop=HOP, mic_dim=1
)
if not USE_TWO_NOISES:
    angle_n2 = None
# annotate with your randomized angles from above
plot_beampattern(ampl_sq_db, RUN_DIR, angle_x=angle_src, angle_n1=angle_n1, angle_n2=angle_n2, tag=f"frame{frame}")
plot_beam_spectrogram(ampl_spec_db, FS, RUN_DIR, angle_x=angle_src, angle_n1=angle_n1, angle_n2=angle_n2, tag=f"frame{frame}")
