#!/usr/bin/env python3
# 4-frame polar + spectrogram beampattern collages (paper-ready)

import os, ast, numpy as np, pandas as pd, scipy.io as sio, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import Preprocesing, Postprocessing, return_as_complex

# ===== CONFIG =====
MAT_FILE   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_DUAL_TRUE_RIRS/TEST_STFT_domain_results_28_10_2025__21_56_28_0.mat"
# MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_MIX/TEST_STFT_domain_results_28_10_2025__22_14_30_0.mat"
#
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_DUAL/TEST_STFT_domain_results_28_10_2025__22_04_07_0.mat"
SURROUND_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
CSV_FILE  = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
OUT_DIR   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gifs"

INDEX = 6
FS = 16000
WIN_LENGTH = 512
HOP = WIN_LENGTH // 4
BP_DB_MIN, BP_DB_MAX = -10.0, 0.0
SPEC_DB_MIN, SPEC_DB_MAX = -60.0, 0.0
EFFECTIVE_HEAD = 0.5
TOTAL_T = 4.0
# ==================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def doa_local_deg(x, y, cx, cy, orientation_deg):
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    ang_local  = (ang_global - orientation_deg) % 360.0
    return 360.0 - ang_local if ang_local > 180.0 else ang_local

# ==== Geometry ====
df = pd.read_csv(CSV_FILE)
row = df.iloc[INDEX]
mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
center_xy = mic_positions[:, :2].mean(axis=0)
sp_start = np.array([float(row["speaker_start_x"]), float(row["speaker_start_y"]), float(row["speaker_start_z"])])
sp_stop  = np.array([float(row["speaker_stop_x"]),  float(row["speaker_stop_y"]),  float(row["speaker_stop_z"])])
orientation_deg = float(row["angleOrientation"])

theta0 = np.arctan2(sp_start[1]-center_xy[1], sp_start[0]-center_xy[0])
theta1 = np.arctan2(sp_stop[1]-center_xy[1],  sp_stop[0]-center_xy[0])
dtheta = ((theta1 - theta0 + np.pi) % (2*np.pi)) - np.pi
radius = float(row.get("radius", np.hypot(sp_start[0]-center_xy[0], sp_start[1]-center_xy[1])))

# ==== Load weights ====
mat = sio.loadmat(MAT_FILE)
W = torch.from_numpy(mat["W_Stage1_left"]).to(device)[INDEX, :, :, :].unsqueeze(0)
_, M, F, L = W.shape

# ==== Surround ====
surround = []
for doa in range(181):
    m = sio.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
    feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(device)
    surround.append(feat.unsqueeze(0))

angles = np.arange(181)
mapped = np.pi / 2 - np.deg2rad(angles)
mapped =  np.deg2rad(angles)
# ==== Choose same 4 representative frames ====
active_T = TOTAL_T - EFFECTIVE_HEAD
t_list = [
    EFFECTIVE_HEAD + 0.3 * active_T,
    EFFECTIVE_HEAD + 0.55 * active_T,
    EFFECTIVE_HEAD + 0.68 * active_T,
    EFFECTIVE_HEAD + 0.75 * active_T,
]
def time_to_frame(t_sec):
    f = int(round((t_sec*FS - WIN_LENGTH/2) / HOP))
    return max(0, min(L-1, f))
frame_ids = [time_to_frame(t) for t in t_list]
titles = [f"Frame {fidx} / {L}" for fidx in frame_ids]

def beampattern(Wf):
    pow_list = []
    spec_acc = torch.zeros((F, 181), dtype=torch.float32, device=device)
    for doa in range(181):
        Y  = Preprocesing(surround[doa], WIN_LENGTH, FS, surround[doa].shape[1] / FS, HOP, device)
        Yc = return_as_complex(Y)
        Z  = torch.sum(torch.conj(Wf) * Yc, dim=1)
        zt = Postprocessing(Z, HOP, WIN_LENGTH, device)
        pow_list.append(torch.sum(torch.abs(zt)**2))
        spec_acc[:, doa] = torch.mean(torch.abs(Z)**2, dim=-1).squeeze(0)
    a = torch.stack(pow_list)
    a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a)+1e-12), min=1e-12)).detach().cpu().numpy()
    a_db = np.clip(a_db, BP_DB_MIN, BP_DB_MAX)
    spec_db = 10.0 * torch.log10(torch.clamp(spec_acc, min=1e-12)).detach().cpu().numpy()
    spec_db = np.clip(spec_db, SPEC_DB_MIN, SPEC_DB_MAX)
    return a_db, spec_db

# ==== Create both collages ====
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. POLAR COLLAGE ---
fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "polar"}, figsize=(6.2, 5.4), dpi=300)
axes = axes.flatten()

for ax, fidx, title in zip(axes, frame_ids, titles):
    a_db, spec_db = beampattern(W[:, :, :, fidx:fidx+1])
    r = a_db - BP_DB_MIN
    # true DOA
    t_sec = (fidx*HOP + WIN_LENGTH/2) / FS
    doa_speaker = None
    if t_sec > EFFECTIVE_HEAD:
        alpha = np.clip((t_sec - EFFECTIVE_HEAD) / active_T, 0.0, 1.0)
        theta = theta0 + alpha * dtheta
        cx, cy = center_xy
        x_pos, y_pos = cx + radius*np.cos(theta), cy + radius*np.sin(theta)
        doa_speaker = doa_local_deg(x_pos, y_pos, cx, cy, orientation_deg)

    ax.plot(mapped, r, color="steelblue", linewidth=1.2)
    if doa_speaker is not None:
        ax.plot([np.pi/2 - np.deg2rad(doa_speaker)], [BP_DB_MAX - BP_DB_MIN],
                marker="o", markersize=4, color="crimson")

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_rlim(0, BP_DB_MAX - BP_DB_MIN)
    ax.set_rticks([0,2.5,5,7.5,10])
    ax.set_yticklabels([f"{BP_DB_MIN + t:.0f}" for t in [0,2.5,5,7.5,10]])
    ax.tick_params(labelsize=6)
    ax.set_xticks(np.deg2rad([-90,-45,0,45,90]))
    ax.set_xticklabels(["-90°","-45°","0°","45°","90°"],fontsize=7)
    ax.grid(True,linewidth=0.4)
    ax.set_title(title,fontsize=12,pad=0.0, weight="regular")

handles = [
    plt.Line2D([], [], color="steelblue", lw=1.2, label="Beampattern"),
    plt.Line2D([], [], color="crimson", marker="o", linestyle="", markersize=4, label="Speaker DOA"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.05))
plt.tight_layout(pad=0.9)
plt.subplots_adjust(bottom=0.12)
polar_out = os.path.join(OUT_DIR, "beampattern_collage_paper_DUAL.png")
fig.savefig(polar_out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Polar collage saved: {polar_out}")

# --- 2. SPECTROGRAM COLLAGE ---
freqs = np.linspace(0.0, FS/2, F)
fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.9), dpi=300)
axes = axes.flatten()

for ax, fidx, title in zip(axes, frame_ids, titles):
    a_db, spec_db = beampattern(W[:, :, :, fidx:fidx+1])
    t_sec = (fidx*HOP + WIN_LENGTH/2)/FS
    doa_speaker = None
    if t_sec > EFFECTIVE_HEAD:
        alpha = np.clip((t_sec - EFFECTIVE_HEAD) / active_T, 0.0, 1.0)
        theta = theta0 + alpha * dtheta
        cx, cy = center_xy
        x_pos, y_pos = cx + radius*np.cos(theta), cy + radius*np.sin(theta)
        doa_speaker = doa_local_deg(x_pos, y_pos, cx, cy, orientation_deg)

    # Shift DOA axis from [0,180] → [-90,90]
    doa_shifted = np.linspace(-90, 90, 181)

    im = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        extent=[-90, 90, freqs[0], freqs[-1]],
        vmin=SPEC_DB_MIN,
        vmax=SPEC_DB_MAX,
        cmap="viridis"
    )

    if doa_speaker is not None:
        doa_shifted_val = doa_speaker - 90  # shift reference to [-90, 90]
        ax.axvline(x=doa_shifted_val, color="crimson", linestyle="--", linewidth=1.2)

    ax.set_title(title, fontsize=12, pad=1)
    ax.set_xlabel("DOA (deg)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(False)

plt.tight_layout(pad=0.8, h_pad=1.7, w_pad=0.8)
plt.subplots_adjust(bottom=0.15)  # leave space below the grid

# --- LEF# --- LEFT: horizontal colorbar ---
cbar_ax = fig.add_axes([0.12, 0.10, 0.35, 0.03])  # higher and taller
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Power (dB)", fontsize=11, labelpad=3)  # bigger label
cbar.ax.tick_params(labelsize=9)

# --- RIGHT: Speaker DOA legend ---
handles = [plt.Line2D([], [], color="crimson", linestyle="--", lw=2.0, label="Speaker DOA")]
fig.legend(
    handles=handles,
    loc="lower right",
    bbox_to_anchor=(0.97, 0.10),  # move slightly higher
    fontsize=11,                   # larger font
    frameon=False,
)
plt.subplots_adjust(bottom=0.27)  # increase bottom padding

spectro_out = os.path.join(OUT_DIR, "spectrogram_collage_paper_DUAL.png")
fig.savefig(spectro_out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Spectrogram collage saved: {spectro_out}")
