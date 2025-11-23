#!/usr/bin/env python3
# Compare time-varying beampatterns from two models (polar animation only)

import os, time, ast
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio

from utils import Preprocesing, Postprocessing, return_as_complex

# ===== CONFIG =====
MAT_FILE_A = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_10_2025_PINK_DUAL_TRUE_RIRS/TEST_STFT_domain_results_26_10_2025__10_02_27_0.mat"
MAT_FILE_B = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_10_2025_PINK_MIX/TEST_STFT_domain_results_26_10_2025__10_14_33_0.mat"

SURROUND_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
CSV_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
CLEAN_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen_with_rir"
CLEAN_PREFIX = "clean_example_"
OUT_ROOT = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gifs"

FS = 16000
WIN_LENGTH = 512
HOP = WIN_LENGTH // 4
FRAME_STRIDE = 10
MAX_FRAMES = 60
VIDEO_FPS = 8
INDEX = 4

BP_DB_MIN, BP_DB_MAX = -15.0, 0.0
EFFECTIVE_HEAD = 0.5
# ==================

def doa_local_deg(x, y, cx, cy, orientation_deg):
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    ang_local = (ang_global - orientation_deg) % 360.0
    return 360.0 - ang_local if ang_local > 180.0 else ang_local

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Load geometry ====
df = pd.read_csv(CSV_FILE)
row = df.iloc[INDEX]
mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
mic_center = mic_positions.mean(axis=0)

sp_start = np.array([float(row["speaker_start_x"]), float(row["speaker_start_y"]), float(row["speaker_start_z"])])
sp_stop = np.array([float(row["speaker_stop_x"]), float(row["speaker_stop_y"]), float(row["speaker_stop_z"])])
orientation_deg = float(row["angleOrientation"])

center_xy = mic_positions[:, :2].mean(axis=0)
start_xy, stop_xy = sp_start[:2] - center_xy, sp_stop[:2] - center_xy
theta0, theta1 = np.arctan2(start_xy[1], start_xy[0]), np.arctan2(stop_xy[1], stop_xy[0])
dtheta = ((theta1 - theta0 + np.pi) % (2*np.pi)) - np.pi
radius = float(row.get("radius", np.hypot(start_xy[0], start_xy[1])))
z_fixed = sp_start[2]

# ==== Load both models ====
def load_model_weights(mat_path):
    mat = sio.loadmat(mat_path)
    W = torch.from_numpy(mat["W_Stage1_left"]).to(device)
    return W

W_A = load_model_weights(MAT_FILE_A)[INDEX, :, :, :].unsqueeze(0)
W_B = load_model_weights(MAT_FILE_B)[INDEX, :, :, :].unsqueeze(0)

_, M, F, L = W_A.shape
frames = list(range(0, L, max(1, FRAME_STRIDE)))
if MAX_FRAMES: frames = frames[:MAX_FRAMES]

# Preload surround
surround = []
for doa in range(181):
    m = sio.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
    feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(device)
    surround.append(feat.unsqueeze(0))

angles = np.arange(181)
ang_rad = np.deg2rad(angles)
mapped = np.pi / 2 - ang_rad

frames_beam = []

# ==== Compute frames ====
for fidx in frames:
    # Speaker DOA
    t_center = fidx * HOP + WIN_LENGTH // 2
    t_sec = max(0.0, t_center / FS)
    if t_sec <= EFFECTIVE_HEAD:
        doa_speaker = None
    else:
        alpha = float(np.clip((t_sec - EFFECTIVE_HEAD) / (4 - EFFECTIVE_HEAD), 0.0, 1.0))
        theta = float(theta0 + alpha * dtheta)
        cx, cy = center_xy
        x_pos, y_pos = cx + radius * np.cos(theta), cy + radius * np.sin(theta)
        doa_speaker = doa_local_deg(x_pos, y_pos, cx, cy, orientation_deg)

    # Compute beampatterns
    def beampattern(Wf):
        pow_list = []
        for doa in range(181):
            Y = Preprocesing(surround[doa], WIN_LENGTH, FS, surround[doa].shape[1] / FS, HOP, device)
            Yc = return_as_complex(Y)
            Z = torch.sum(torch.conj(Wf) * Yc, dim=1)
            zt = Postprocessing(Z, HOP, WIN_LENGTH, device)
            pow_list.append(torch.sum(torch.abs(zt)**2))
        a = torch.stack(pow_list)
        a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a) + 1e-12), min=1e-12)).detach().cpu().numpy()
        return np.clip(a_db, BP_DB_MIN, BP_DB_MAX)

    a_db_A = beampattern(W_A[:, :, :, fidx:fidx+1])
    a_db_B = beampattern(W_B[:, :, :, fidx:fidx+1])

    # Normalize relative to BP_DB_MIN
    rA, rB = a_db_A - BP_DB_MIN, a_db_B - BP_DB_MIN

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.25, 5.0), dpi=128)
    ax.plot(mapped, rA, color="blue", linewidth=1.6, label="Model A (BABBLE_MIX)")
    ax.plot(mapped, rB, color="red", linewidth=1.6, label="Model B (BABBLE_DUAL_MODEL)")

    if doa_speaker is not None:
        ax.plot([np.pi/2 - np.deg2rad(doa_speaker)], [BP_DB_MAX - BP_DB_MIN],
                "ko", label="Dynamic Speaker", markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_rlim(0, BP_DB_MAX - BP_DB_MIN)
    rticks = np.arange(0, (BP_DB_MAX - BP_DB_MIN) + 0.1, 5)
    ax.set_rticks(rticks)
    ax.set_yticklabels([f"{BP_DB_MIN + t:.0f} dB" for t in rticks])
    ax.set_rlabel_position(135)
    ax.grid(True)
    ax.set_title(f"Frame {fidx}/{L-1}", y=0.87)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.05 + 0.15), fontsize=8, frameon=True)

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    frames_beam.append(img)
    plt.close(fig)

# ==== Save video ====
out_dir = os.path.join(
    OUT_ROOT,
    "comparison_of_24_10_2025_BABBLE_DUAL_MODEL_and_24_10_2025_BABBLE_MIX"
)
os.makedirs(out_dir, exist_ok=True)

stamp = time.strftime("%Y%m%d_%H%M%S")
mp4_out = os.path.join(out_dir, f"beampattern_comparison_INDEX_{INDEX}_{stamp}.mp4")
iio.imwrite(mp4_out, frames_beam, fps=VIDEO_FPS, codec="libx264")
print(f"[OK] Comparison MP4 saved to:\n{mp4_out}")
