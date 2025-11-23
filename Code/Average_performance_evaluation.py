#!/usr/bin/env python3
# Full test-set performance evaluation (SI-SDR + PESQ + STOI + DNSMOS)
# Averages over all TEST_time_domain_results_*.mat in the given model directory

import os
import numpy as np
import scipy.io as sio
import torch
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    DeepNoiseSuppressionMeanOpinionScore,
)

# ================== CONFIG ==================
MODEL_DIR   = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_DUAL_TRUE_RIRS_SNR_3"
CLEAN_DIR   = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen_with_rir"
CLEAN_PREFIX = "clean_example_"
FS           = 16000
NOISE_HEAD_SEC = 0.5
NOISE_SAMPLES   = int(NOISE_HEAD_SEC * FS)
# ============================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Metrics ====
si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
pesq_metric   = PerceptualEvaluationSpeechQuality(FS, "wb").to(device)
stoi_metric   = ShortTimeObjectiveIntelligibility(FS, False).to(device)


# ==== Collect MAT files ====
mat_files = sorted([
    os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.startswith("TEST_time_domain_results") and f.endswith(".mat")
])

if not mat_files:
    raise FileNotFoundError(f"No TEST_time_domain_results_*.mat found in {MODEL_DIR}")

# ==== Accumulators ====
def init_acc():
    return dict(sisdr_in_L=0, sisdr_in_R=0, sisdr_L=0, sisdr_R=0,
                pesq_in_L=0, pesq_in_R=0, pesq_L=0, pesq_R=0,
                stoi_in_L=0, stoi_in_R=0, stoi_L=0, stoi_R=0       )

acc = init_acc()
count = 0

# ==== Loop over all mat files ====
for f_idx, MAT_FILE in enumerate(mat_files):
    mat = sio.loadmat(MAT_FILE)
    if "fs" in mat:
        FS = int(np.array(mat["fs"]).squeeze())

    for INDEX in range(0, 8):
        global_index = f_idx * 8 + INDEX 
        clean_path = os.path.join(CLEAN_DIR, f"{CLEAN_PREFIX}{global_index:07d}.mat")
        if not os.path.exists(clean_path):
            continue

        clean_mat = sio.loadmat(clean_path)
        clean = clean_mat["clean"]
        clean_L = torch.tensor(clean[:, 0], dtype=torch.float32, device=device)
        clean_R = torch.tensor(clean[:, 7], dtype=torch.float32, device=device)

        y = torch.from_numpy(mat["y"])[INDEX, NOISE_SAMPLES:].to(device).float()
        y_L, y_R = y[:, 0], y[:, 7]
        xL = torch.from_numpy(mat["x_hat_stage1_left"])[INDEX, NOISE_SAMPLES:].to(device).float()
        xR = torch.from_numpy(mat["x_hat_stage1_right"])[INDEX, NOISE_SAMPLES:].to(device).float()

        # === SI-SDR ===
        acc["sisdr_in_L"] += si_sdr_metric(y_L.unsqueeze(0), clean_L.unsqueeze(0))
        acc["sisdr_in_R"] += si_sdr_metric(y_R.unsqueeze(0), clean_R.unsqueeze(0))
        acc["sisdr_L"]    += si_sdr_metric(xL.unsqueeze(0), clean_L.unsqueeze(0))
        acc["sisdr_R"]    += si_sdr_metric(xR.unsqueeze(0), clean_R.unsqueeze(0))

        # === PESQ ===
        acc["pesq_in_L"] += pesq_metric(y_L.unsqueeze(0), clean_L.unsqueeze(0))
        acc["pesq_in_R"] += pesq_metric(y_R.unsqueeze(0), clean_R.unsqueeze(0))
        acc["pesq_L"]    += pesq_metric(xL.unsqueeze(0), clean_L.unsqueeze(0))
        acc["pesq_R"]    += pesq_metric(xR.unsqueeze(0), clean_R.unsqueeze(0))

        # === STOI ===
        acc["stoi_in_L"] += stoi_metric(y_L.unsqueeze(0), clean_L.unsqueeze(0))
        acc["stoi_in_R"] += stoi_metric(y_R.unsqueeze(0), clean_R.unsqueeze(0))
        acc["stoi_L"]    += stoi_metric(xL.unsqueeze(0), clean_L.unsqueeze(0))
        acc["stoi_R"]    += stoi_metric(xR.unsqueeze(0), clean_R.unsqueeze(0))


        count += 1

# ==== Print Averages ====
def avg(x): return float(x / count) if count > 0 else np.nan

print(f"\n[Model: {os.path.basename(MODEL_DIR)}] — Averaged over {count} test samples")
print("--- SI-SDR ---")
print(f"  L: {avg(acc['sisdr_in_L']):+.2f} → {avg(acc['sisdr_L']):+.2f}  (Δ {avg(acc['sisdr_L'] - acc['sisdr_in_L']):+.2f})")
print(f"  R: {avg(acc['sisdr_in_R']):+.2f} → {avg(acc['sisdr_R']):+.2f}  (Δ {avg(acc['sisdr_R'] - acc['sisdr_in_R']):+.2f})")

print("\n--- PESQ ---")
print(f"  L: {avg(acc['pesq_in_L']):+.3f} → {avg(acc['pesq_L']):+.3f}  (Δ {avg(acc['pesq_L'] - acc['pesq_in_L']):+.3f})")
print(f"  R: {avg(acc['pesq_in_R']):+.3f} → {avg(acc['pesq_R']):+.3f}  (Δ {avg(acc['pesq_R'] - acc['pesq_in_R']):+.3f})")

print("\n--- STOI ---")
print(f"  L: {avg(acc['stoi_in_L']):+.3f} → {avg(acc['stoi_L']):+.3f}  (Δ {avg(acc['stoi_L'] - acc['stoi_in_L']):+.3f})")
print(f"  R: {avg(acc['stoi_in_R']):+.3f} → {avg(acc['stoi_R']):+.3f}  (Δ {avg(acc['stoi_R'] - acc['stoi_in_R']):+.3f})")
