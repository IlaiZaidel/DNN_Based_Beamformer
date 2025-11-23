#!/usr/bin/env python3
import os
import sys
import math
import ast
import numpy as np
import pandas as pd
import soundfile as sf
import random
import matplotlib.pyplot as plt

# ====== USER SETTINGS ======
CSV_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
INDEX    = 18   # which row to process
OUT_WAV  = "clean_example_test.wav"
OUT_PNG  = "room_plot.png"
NSAMPLES = 1024
ORDER    = 1
HOP      = 32
M_TYPE   = "o"
C        = 343.0
SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_rtf_output"
# ===========================

if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)
from signal_generator import SignalGenerator

# --- Load CSV row ---
df = pd.read_csv(CSV_PATH)
row = df.iloc[INDEX].to_dict()

# --- Room & mic info ---
wav_path = row["speaker_path"]
fs       = int(row.get("fs", 16000))
T_sec    = float(row.get("T", 4.0))
L        = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
beta     = [float(row["beta"])] * 6

# --- Load and prepare audio ---
x, fs_file = sf.read(wav_path)
if x.ndim > 1:
    x = x[:, 0]
if fs_file != fs:
    raise RuntimeError(f"fs mismatch: wav={fs_file}, csv={fs}")

N = int(T_sec * fs)
if len(x) < N:
    reps = math.ceil(N / len(x))
    x = np.tile(x, reps)
x = x[:N].astype(np.float64)

# --- Create mic positions ---
angleOrientation = random.choice(np.arange(-45, 46))
mic_height = random.uniform(1.0, 1.5)
mic_x = random.uniform(1.0, L[0] - 1.0)
mic_y = random.uniform(1.0, L[1] - 1.0)
center = np.array([mic_x, mic_y])

mic_offsets = np.array([
    [-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
    [ 0.04, 0], [ 0.07, 0], [ 0.12, 0], [ 0.17, 0]
])
theta = np.radians(angleOrientation)
rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
mic_rotated = mic_offsets @ rot.T + center
mic_pos = np.column_stack([mic_rotated, np.full(len(mic_rotated), mic_height)])
M = mic_pos.shape[0]

# --- Source motion (arc) ---
radius = random.uniform(1.0, 1.5)
angle_x = random.randint(0, 180)
speaker_start = np.array([
    mic_x + radius * np.cos(np.radians(angle_x + angleOrientation)),
    mic_y + radius * np.sin(np.radians(angle_x + angleOrientation)),
    random.uniform(1.2, 1.8)
])
delta_angle = 120
speaker_stop = np.array([
    mic_x + radius * np.cos(np.radians(angle_x + delta_angle + angleOrientation)),
    mic_y + radius * np.sin(np.radians(angle_x + delta_angle + angleOrientation)),
    speaker_start[2]
])
speaker_stop = speaker_start
# --- Build paths ---
sp_path = np.zeros((N, 3))
rp_path = np.zeros((N, 3, M))
for i in range(0, N, HOP):
    alpha = i / max(1, N - 1)
    sp = speaker_start + alpha * (speaker_stop - speaker_start)
    end = min(i + HOP, N)
    sp_path[i:end] = sp
    for m in range(M):
        rp_path[i:end, :, m] = mic_pos[m]

# --- Run SignalGenerator ---
gen = SignalGenerator()
result = gen.generate(
    list(x), C, fs, rp_path.tolist(), sp_path.tolist(),
    L, beta, NSAMPLES, M_TYPE, ORDER, 3, [], True
)
# ----- pull time-varying RIRs -----
all_rirs = np.array(result.all_rirs, dtype=np.float64)  # [M, T, nsamples]
clean = np.array(result.output, dtype=np.float64).T  # (N, M)

# --- Stereo from outermost mics ---
if clean.shape[1] >= 2:
    stereo = np.stack([clean[:, 0], clean[:, -1]], axis=1)
else:
    stereo = np.repeat(clean, 2, axis=1)

# --- Normalize & save ---
peak = np.max(np.abs(stereo))
if peak > 0:
    stereo /= peak
sf.write(OUT_WAV, stereo.astype(np.float32), fs)
print(f"Saved stereo WAV: {OUT_WAV}")

# --- Plot room top-down ---
plt.figure(figsize=(6, 6))
plt.plot([0, L[0], L[0], 0, 0], [0, 0, L[1], L[1], 0], 'k-', linewidth=1)  # room outline
plt.scatter(mic_pos[:, 0], mic_pos[:, 1], c='b', label="Microphones")
plt.plot(sp_path[:, 0], sp_path[:, 1], 'r--', label="Source path")
plt.scatter(speaker_start[0], speaker_start[1], c='g', marker='*', s=150, label="Start")
plt.scatter(speaker_stop[0], speaker_stop[1], c='r', marker='*', s=150, label="Stop")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.xlim(0, L[0])
plt.ylim(0, L[1])
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("Room Layout and Source Path")
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"Saved room plot: {OUT_PNG}")



# ====== Plot time-varying RIRs (mic 0 by default) ======
import matplotlib.pyplot as plt

STFT_WIN = 512
STFT_HOP = 128

# all_rirs: [M, L_frames, nsamples]
M, L_frames, K = all_rirs.shape
mic_idx = 0  # choose which mic to visualize

# 1) Heatmap: |RIR| vs (frame, time lag)
rir_mag = np.abs(all_rirs[mic_idx])  # [L_frames, K]
t_lag = np.arange(K) / fs * 1e3      # ms
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
im = ax.imshow(rir_mag.T, aspect='auto', origin='lower',
               extent=[0, L_frames-1, t_lag[0], t_lag[-1]])
ax.set_xlabel("STFT frame index")
ax.set_ylabel("Time lag (ms)")
ax.set_title(f"Time-varying direct-path RIR (mic {mic_idx})")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|h|")
fig.tight_layout()
plt.savefig("rir_heatmap.png", dpi=150)
plt.close(fig)
print("[OK] saved rir_heatmap.png")

# 2) A few exemplar RIRs over time (start / mid / end)
pick_frames = [0, max(0, L_frames//2), max(0, L_frames-1)]
fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
for f in pick_frames:
    h = all_rirs[mic_idx, f]
    ax.plot(t_lag, h, label=f"frame {f}")
ax.set_xlim(0, t_lag[-1])
ax.set_xlabel("Time lag (ms)")
ax.set_ylabel("Amplitude")
ax.set_title(f"RIR snapshots (mic {mic_idx}, ORDER=0)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("rir_snapshots.png", dpi=150)
plt.close(fig)
print("[OK] saved rir_snapshots.png")

# (Optional) overlay expected direct-path TOA in the heatmap
# Compute frame-center times and source→mic distance
L_frames_expected = (N - STFT_WIN) // STFT_HOP + 1 if N >= STFT_WIN else 0
assert L_frames_expected == L_frames, "frame count mismatch (sync STFT_WIN/HOP with C++)"

frame_centers = np.arange(L_frames) * STFT_HOP + STFT_WIN//2  # samples
# Use the same source/mic geometry you built to predict TOA for mic_idx
mic_xyz = mic_pos[mic_idx]             # (3,)
src_xyz_per_frame = sp_path[frame_centers]  # (L_frames, 3) -> center sample positions
dist = np.linalg.norm(src_xyz_per_frame - mic_xyz, axis=1)    # meters
toa_ms = (dist / C) * 1e3  # ms

# Plot heatmap again with TOA ridge
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
im = ax.imshow(rir_mag.T, aspect='auto', origin='lower',
               extent=[0, L_frames-1, t_lag[0], t_lag[-1]])
ax.plot(np.arange(L_frames), toa_ms, 'w--', linewidth=1.2, label="pred. TOA")
ax.set_xlabel("STFT frame index")
ax.set_ylabel("Time lag (ms)")
ax.set_title(f"RIR heatmap + predicted direct-path TOA (mic {mic_idx})")
fig.colorbar(im, ax=ax, label="|h|")
ax.legend()
fig.tight_layout()
plt.savefig("rir_heatmap_with_toa.png", dpi=150)
plt.close(fig)
print("[OK] saved rir_heatmap_with_toa.png")