#!/usr/bin/env python3
# Static (time-averaged) beampattern + DOA–freq spectrogram
# for two static speakers (from GeneratedData_Two_Static_Speakers_Babble setup)

import os, ast
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import Preprocesing, Postprocessing, return_as_complex

# ================== USER SETTINGS ==================
MAT_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/29_11_BABBLE_ATTEN_TRUE_RIRS_Two_Speakers/TEST_STFT_domain_results_30_11_2025__02_33_11_0.mat"
# MAT_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/27_11_BABBLE_ATTEN_TRUE_RIRS_Two_Speakers/TEST_STFT_domain_results_27_11_2025__00_17_45_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/21_12_f_x_penalty_first_second/TEST_STFT_domain_results_21_12_2025__11_58_26_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/26_12_Augmented_Langrangian_SISDR_again/TEST_STFT_domain_results_26_12_2025__12_13_40_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/27_12_Augmented_Langrangian_MSE/TEST_STFT_domain_results_27_12_2025__20_36_27_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/30_12_Augmented_Langrangian_Minimum_Variance/TEST_STFT_domain_results_30_12_2025__11_56_33_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/06_01_AL_w_Penalty_SISDR/TEST_STFT_domain_results_06_01_2026__14_59_13_0.mat"
CSV_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
SURROUND_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
OUT_ROOT     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_pngs"

INDEX            = 5       # sample index in batch / CSV
FS               = 16000
WIN_LENGTH       = 512
HOP              = WIN_LENGTH // 4
NOISE_HEAD_SEC   = 0.5        # same as dataset.noise_only_time
BP_DB_MIN        = -15.0
BP_DB_MAX        = 0.0
SPEC_DB_MIN      = -60.0
SPEC_DB_MAX      = 0.0
# ===================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def doa_local_deg(x, y, cx, cy, orientation_deg):
    """
    DOA in array-local frame:
    orientation_deg: array rotation (angleOrientation) from CSV.
    Returns in [0,180], where 0 = front, 90 = side.
    """
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    ang_local  = (ang_global - orientation_deg) % 360.0
    return 360.0 - ang_local if ang_local > 180.0 else ang_local

# ====== Load geometry from CSV ======
df = pd.read_csv(CSV_FILE)
row1 = df.iloc[INDEX]
row2 = df.iloc[(INDEX + 1) % len(df)]  # second speaker uses next row's wav, but we only need room1 geometry

mic_positions = np.array(ast.literal_eval(row1["mic_positions"]), dtype=np.float64)  # (M,3)
mic_center    = mic_positions.mean(axis=0)
cx, cy        = mic_center[0], mic_center[1]
orientation   = float(row1["angleOrientation"])

# speaker 1: its own speaker_start
spk1_pos = np.array([
    float(row1["speaker_start_x"]),
    float(row1["speaker_start_y"]),
    float(row1["speaker_start_z"])
], dtype=np.float64)

# speaker 2: noise1 location (as in your dataset)
spk2_pos = np.array([
    float(row1["noise1_x"]),
    float(row1["noise1_y"]),
    float(row1["noise1_z"])
], dtype=np.float64)

doa_spk1 = doa_local_deg(spk1_pos[0], spk1_pos[1], cx, cy, orientation)
doa_spk2 = doa_local_deg(spk2_pos[0], spk2_pos[1], cx, cy, orientation)

print(f"Speaker 1 DOA (local): {doa_spk1:.2f} deg")
print(f"Speaker 2 DOA (local): {doa_spk2:.2f} deg")

# ====== Load STFT-domain weights ======
mat = sio.loadmat(MAT_FILE)
if "fs" in mat:
    FS = int(np.array(mat["fs"]).squeeze())

W_all = torch.from_numpy(mat["W_Stage1_left"]).to(device)  # [B, M, F, L]
B, M, F, L = W_all.shape

# number of STFT frames during noise-only head
noise_frames = int(NOISE_HEAD_SEC * FS // HOP)
noise_frames = min(noise_frames, L - 1)  # safety

# take sample INDEX
W = W_all[INDEX]  # [M, F, L]

# average over frames from Ln .. L-1 (speech+noise frames)
W_avg = W[:, :, noise_frames:].mean(dim=-1, keepdim=True)  # [M, F, 1]
W_avg = W_avg.unsqueeze(0)  # [1, M, F, 1] for broadcasting

# ====== preload surround inputs ======
surround = []
for doa in range(181):
    m = sio.loadmat(
        os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat")
    )
    feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(device)  # [N,M]
    surround.append(feat.unsqueeze(0))  # [1,N,M]

angles = np.arange(181)
ang_rad = np.deg2rad(angles)
mapped = np.pi/2 - ang_rad  # for polar plot

pow_list = []
spec_acc = torch.zeros((F, 181), dtype=torch.float32, device=device)

for doa in range(181):
    y_in = surround[doa]  # [1, N, M]
    T_sec = y_in.shape[1] / FS

    Y = Preprocesing(y_in, WIN_LENGTH, FS, T_sec, HOP, device)   # [1, M, F, Ls]
    Yc = return_as_complex(Y)                                   # complex

    # apply beamformer: sum over mics
    Z = torch.sum(torch.conj(W_avg) * Yc, dim=1)  # [1, F, Ls]
    z_t = Postprocessing(Z, HOP, WIN_LENGTH, device)  # [1, Ns]

    pow_list.append(torch.sum(torch.abs(z_t)**2))  # scalar tensor
    spec_acc[:, doa] = torch.mean(torch.abs(Z)**2, dim=-1).squeeze(0)

# ====== Beampattern power (scalar vs DOA) ======
a = torch.stack(pow_list)  # [181]
a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a) + 1e-12),
                                      min=1e-12)).detach().cpu().numpy()
a_db = np.clip(a_db, BP_DB_MIN, BP_DB_MAX)
r = a_db - BP_DB_MIN  # shift to [0, BP_DB_MAX-BP_DB_MIN]

# ====== DOA–freq spectrogram ======
spec_db = 10.0 * torch.log10(torch.clamp(spec_acc, min=1e-12)).detach().cpu().numpy()
spec_db = np.clip(spec_db, SPEC_DB_MIN, SPEC_DB_MAX)
freqs = np.linspace(0.0, FS / 2, F)

# ====== Output dir ======
model_dir = os.path.basename(os.path.dirname(MAT_FILE))
out_dir = os.path.join(OUT_ROOT, model_dir)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------
#       POLAR BEAMPATTERN PNG
# ---------------------------------------------------
fig_bp, ax_bp = plt.subplots(subplot_kw={"projection": "polar"},
                             figsize=(6.25, 5.0), dpi=150)

ax_bp.plot(mapped, r, label="Beampower")

# speaker 1 marker + line
phi1 = np.pi/2 - np.deg2rad(doa_spk1)
phi2 = np.pi/2 - np.deg2rad(doa_spk2)
r_max = BP_DB_MAX - BP_DB_MIN

ax_bp.plot([phi1, phi1], [0, r_max], linestyle="--")
ax_bp.plot(phi1, r_max, "ro", label="Speaker 1")

# speaker 2 marker + line
ax_bp.plot([phi2, phi2], [0, r_max], linestyle="--")
ax_bp.plot(phi2, r_max, "go", label="Speaker 2")

ax_bp.set_theta_zero_location("N")
ax_bp.set_theta_direction(-1)
ax_bp.set_thetamin(-90)
ax_bp.set_thetamax(90)
ax_bp.set_rlim(0, r_max)

rticks = np.arange(0, r_max + 0.1, 5)
ax_bp.set_rticks(rticks)
ax_bp.set_yticklabels([f"{BP_DB_MIN + t:.0f} dB" for t in rticks])
ax_bp.set_rlabel_position(135)
ax_bp.grid(True)
ax_bp.set_title(f"Avg Beampower (INDEX={INDEX})", y=0.87)
ax_bp.legend(loc="lower left",
             bbox_to_anchor=(0.0, -0.05 + 0.15),
             fontsize=8, frameon=True)

fig_bp.tight_layout()
png_bp = os.path.join(out_dir, f"static_beampattern_INDEX_{INDEX}.png")
fig_bp.savefig(png_bp, dpi=150)
plt.close(fig_bp)
print(f"[OK] Saved polar beampattern: {png_bp}")

# ---------------------------------------------------
#       DOA–FREQ SPECTROGRAM PNG
# ---------------------------------------------------
fig_sp, ax_sp = plt.subplots(figsize=(6.25, 5.0), dpi=150)
im = ax_sp.imshow(
    spec_db,
    aspect="auto", origin="lower",
    extent=[0, 180, freqs[0], freqs[-1]],
    vmin=SPEC_DB_MIN, vmax=SPEC_DB_MAX,
    cmap="viridis",
)
cbar = fig_sp.colorbar(im, ax=ax_sp)
cbar.set_label("Power (dB)")
ax_sp.set_xlabel("DOA (deg)")
ax_sp.set_ylabel("Frequency (Hz)")
ax_sp.set_title(f"Avg Beam-power Spectrogram (INDEX={INDEX})")

# vertical lines + dots at speakers
for doa, color, label in [(doa_spk1, "r", "Speaker 1"),
                          (doa_spk2, "g", "Speaker 2")]:
    ax_sp.axvline(x=doa, color=color, linestyle="--", linewidth=1.5)
    # dot at mid frequency, just for emphasis
    mid_f = freqs[len(freqs)//2]
    ax_sp.plot(doa, mid_f, marker="o", color=color, label=label)

ax_sp.legend(loc="upper right", fontsize=8)
fig_sp.tight_layout()

png_sp = os.path.join(out_dir, f"static_spectrogram_INDEX_{INDEX}.png")
fig_sp.savefig(png_sp, dpi=150)
plt.close(fig_sp)
print(f"[OK] Saved DOA–freq spectrogram: {png_sp}")
