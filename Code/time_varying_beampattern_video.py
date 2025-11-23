#!/usr/bin/env python3
# time-varying beampattern + spectrogram videos from W_Stage1_left
# with true moving-speaker DOA overlay (starts after 0.5s noise-only head)

import os, time, ast
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import imageio.v3 as iio

from utils import Preprocesing, Postprocessing, return_as_complex

# ====== EDIT THESE ======
# MAT_FILE      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/12_10_2025_MIX/TEST_STFT_domain_results_12_10_2025__08_59_42_0.mat"

MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_DUAL_TRUE_RIRS/TEST_STFT_domain_results_28_10_2025__21_56_28_0.mat"
# MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_DUAL/TEST_STFT_domain_results_28_10_2025__22_04_07_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_MIX/TEST_STFT_domain_results_28_10_2025__22_14_30_0.mat"
SURROUND_DIR  = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
OUT_ROOT      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gifs"
CSV_FILE      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
# clean dataset (from your SignalGenerator) — use Test path for this run
CLEAN_DIR     = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen_with_rir"
CLEAN_PREFIX  = "clean_example_"

FS            = 16000
WIN_LENGTH    = 512
HOP           = WIN_LENGTH // 4
FRAME_STRIDE  = 10
MAX_FRAMES    = 60
VIDEO_FPS     = 8
INDEX         =2
# Spectrogram color scale
SPEC_DB_MIN   = -60.0
SPEC_DB_MAX   = 0.0
# Beampattern polar radial labels
BP_DB_MIN     = -15.0
BP_DB_MAX     = 0.0
NOISE_HEAD_SEC = 0.5
# ========================

def doa_local_deg(x, y, cx, cy, orientation_deg):
    """
    Compute the DOA in the array's *local* coordinate frame.
    orientation_deg is the array rotation (angleOrientation) from the CSV.
    Returns 0–180°, where 0° = array front, 90° = side.
    """
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    # Convert to array-local coordinates by subtracting orientation
    ang_local = (ang_global - orientation_deg) % 360.0
    # Fold to 0–180 (since your beampattern bank only covers that range)
    return 360.0 - ang_local if ang_local > 180.0 else ang_local

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Load CSV geometry ====
df = pd.read_csv(CSV_FILE)
row = df.iloc[INDEX]

# mic positions
mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
mic_center = mic_positions.mean(axis=0)

# moving speaker start and stop
sp_start = np.array([float(row["speaker_start_x"]),
                     float(row["speaker_start_y"]),
                     float(row["speaker_start_z"])], dtype=np.float64)
sp_stop  = np.array([float(row["speaker_stop_x"]),
                     float(row["speaker_stop_y"]),
                     float(row["speaker_stop_z"])], dtype=np.float64)

total_T = 4# float(row["T"])  # total signal duration (s)

# ---- helper: compute azimuth angle (fold to 0–180°) ----
def azimuth_deg_front(p, center):
    v = p - center
    ang = (np.degrees(np.arctan2(v[1], v[0])) + 360.0) % 360.0
    if ang > 180.0:
        ang = 360.0 - ang
    return ang
def azimuth_deg_front_2d(x, y, cx, cy):
    ang = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    return 360.0 - ang if ang > 180.0 else ang

def moving_avg(x, win):
    if win <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[win:] - c[:-win]) / float(win)
    # pad to original length (left-pad)
    pad = np.full(win - 1, y[0] if y.size else 0.0, dtype=x.dtype)
    return np.concatenate([pad, y])

# ==== Load STFT data ====
mat = sio.loadmat(MAT_FILE)
if "fs" in mat:
    FS = int(np.array(mat["fs"]).squeeze())

model_dir = os.path.basename(os.path.dirname(MAT_FILE))
out_dir = os.path.join(OUT_ROOT, model_dir)
os.makedirs(out_dir, exist_ok=True)

W = torch.from_numpy(mat["W_Stage1_left"]).to(device)  # [B,M,F,L]
_, M, F, L = W.shape
frames = list(range(0, L, max(1, FRAME_STRIDE)))
if MAX_FRAMES is not None:
    frames = frames[:MAX_FRAMES]

# time mapping: STFT hop → frame center time
N_samples = (L - 1) * HOP + WIN_LENGTH

# polar plot mapping
angles = np.arange(181)
ang_rad = np.deg2rad(angles)
mapped = np.pi / 2 - ang_rad

frames_beam = []
frames_spec = []

# select batch
W = W[INDEX, :, :, :].unsqueeze(0)
clean_mat_path = os.path.join(CLEAN_DIR, f"{CLEAN_PREFIX}{INDEX:07d}.mat")
clean_mat = sio.loadmat(clean_mat_path)
clean = clean_mat["clean"]
# preload surround
surround = []
for doa in range(181):
    m = sio.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
    feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(device)
    surround.append(feat.unsqueeze(0))  # [1, N, M]


 # adjust experimentally (≈60 ms)
EFFECTIVE_HEAD =  0.5
center_xy = mic_positions[:, :2].mean(axis=0)        # (cx, cy)
start_xy  = sp_start[:2] - center_xy
stop_xy   = sp_stop[:2]  - center_xy

theta0 = np.arctan2(start_xy[1], start_xy[0])
theta1 = np.arctan2(stop_xy[1],  stop_xy[0])
# shortest-arc delta (wrap to [-pi, pi))
dtheta = ((theta1 - theta0 + np.pi) % (2*np.pi)) - np.pi

# use CSV radius if provided; otherwise infer from start
radius = float(row.get("radius", np.hypot(start_xy[0], start_xy[1])))

z_fixed = sp_start[2]  # generator keeps Z fixed

for k, fidx in enumerate(frames):
    Wf = W[:, :, :, fidx:fidx + 1]  # [1, M, F, 1]

    # ===== Compute speaker DOA at this frame =====
    t_center = fidx * HOP + WIN_LENGTH // 2 #- HOP // 2
    t_sec = max(0.0, t_center / FS)

    if t_sec <= EFFECTIVE_HEAD:
        doa_speaker = None  # speaker not yet active
    else:
        # normalize along speech duration only
        # alpha = np.clip((t_sec - EFFECTIVE_HEAD) / (total_T - EFFECTIVE_HEAD), 0.0, 1.0)

        # theta = theta0 + alpha * dtheta
        # x_pos = center_xy[0] + r * np.cos(theta)
        # y_pos = center_xy[1] + r * np.sin(theta)
        # sp_t  = np.array([x_pos, y_pos, z_fixed], dtype=np.float64)

        # doa_speaker = azimuth_deg_front(sp_t, mic_center)  # your existing function\
        # === INSIDE the frame loop ===
        alpha = float(np.clip((t_sec - EFFECTIVE_HEAD) / (total_T - EFFECTIVE_HEAD), 0.0, 1.0))

        theta   = float(theta0 + alpha * dtheta)
        # r_val   = float(radius)  # ensure scalar
        cx, cy  = float(center_xy[0]), float(center_xy[1])

        x_pos = float(cx + radius * np.cos(theta))
        y_pos = float(cy + radius * np.sin(theta))
        orientation_deg = float(row["angleOrientation"])  # from your CSV
        doa_speaker = doa_local_deg(x_pos, y_pos, cx, cy, orientation_deg)
        # 2D azimuth: ignore z entirely
        # doa_speaker = azimuth_deg_front_2d(x_pos, y_pos, cx, cy)
        # alpha = np.clip((t_sec - EFFECTIVE_HEAD) / (total_T - EFFECTIVE_HEAD), 0.0, 1.0)
        # sp_t = sp_start + alpha * (sp_stop - sp_start)
        # doa_speaker = azimuth_deg_front(sp_t, mic_center)

    # ===== Beampattern computation =====
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
    a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a) + 1e-12), min=1e-12)).detach().cpu().numpy()
    a_db = np.clip(a_db, BP_DB_MIN, BP_DB_MAX)
    r = a_db - BP_DB_MIN

    spec_db = 10.0 * torch.log10(torch.clamp(spec_acc, min=1e-12)).detach().cpu().numpy()
    spec_db = np.clip(spec_db, SPEC_DB_MIN, SPEC_DB_MAX)

    # ===== Render beampattern frame =====
    fig_bp, ax_bp = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.25, 5.0), dpi=128)
    ax_bp.plot(mapped, r)

    if doa_speaker is not None:
        ax_bp.plot([np.pi/2 - np.deg2rad(doa_speaker)], [BP_DB_MAX - BP_DB_MIN],
                   "ro", label="Dynamic Speaker")

    ax_bp.set_theta_zero_location('N'); ax_bp.set_theta_direction(-1)
    ax_bp.set_thetamin(-90); ax_bp.set_thetamax(90)
    ax_bp.set_rlim(0, BP_DB_MAX - BP_DB_MIN)
    rticks = np.arange(0, (BP_DB_MAX - BP_DB_MIN) + 0.1, 5)
    ax_bp.set_rticks(rticks)
    ax_bp.set_yticklabels([f"{BP_DB_MIN + t:.0f} dB" for t in rticks])
    ax_bp.set_rlabel_position(135)
    ax_bp.grid(True)
    ax_bp.set_title(f"Beampower (frame {fidx}/{L-1})", y=0.87)
    if doa_speaker is not None:
        ax_bp.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, -0.05 + 0.15),  # shift upward by 0.15
            fontsize=8,
            frameon=True
        )
    fig_bp.tight_layout()
    fig_bp.canvas.draw()
    w_bp, h_bp = fig_bp.canvas.get_width_height()
    img_bp = np.frombuffer(fig_bp.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_bp, w_bp, 4))[:, :, :3]
    frames_beam.append(img_bp)
    plt.close(fig_bp)

    # ===== Render spectrogram frame =====
    freqs = np.linspace(0.0, FS / 2, F)
    fig_sp, ax_sp = plt.subplots(figsize=(6.25, 5.0), dpi=128)
    im = ax_sp.imshow(spec_db, aspect="auto", origin="lower",
                      extent=[0, 180, freqs[0], freqs[-1]],
                      vmin=SPEC_DB_MIN, vmax=SPEC_DB_MAX, cmap="viridis")
    cbar = fig_sp.colorbar(im, ax=ax_sp)
    cbar.set_label("Power (dB)")
    ax_sp.set_xlabel("DOA (deg)")
    ax_sp.set_ylabel("Frequency (Hz)")
    ax_sp.set_title(f"Beam-power Spectrogram (frame {fidx}/{L-1})")

    if doa_speaker is not None:
        ax_sp.axvline(x=doa_speaker, color="r", linestyle="--", linewidth=1.5, label="Speaker")
        ax_sp.legend(loc="upper right", fontsize=8)

    fig_sp.tight_layout()
    fig_sp.canvas.draw()
    w_sp, h_sp = fig_sp.canvas.get_width_height()
    img_sp = np.frombuffer(fig_sp.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_sp, w_sp, 4))[:, :, :3]
    frames_spec.append(img_sp)
    plt.close(fig_sp)

# ==== Write videos ====
stamp = time.strftime("%Y%m%d_%H%M%S")
mp4_beam = os.path.join(out_dir, f"beampattern_video_INDEX_{INDEX}_{stamp}.mp4")
mp4_spec = os.path.join(out_dir, f"spectrogram_video_INDEX_{INDEX}_{stamp}.mp4")

iio.imwrite(mp4_beam, frames_beam, fps=VIDEO_FPS, codec="libx264")
print(f"[OK] Beampattern MP4: {mp4_beam}")

iio.imwrite(mp4_spec, frames_spec, fps=VIDEO_FPS, codec="libx264")
print(f"[OK] Spectrogram MP4: {mp4_spec}")
