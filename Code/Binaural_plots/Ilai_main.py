#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILD–ITD PDF analysis for static sources.
Creates a single joint ILD–ITD comparison:
  Mixture (green) vs Estimated (red) vs Clean (blue)
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cues import calc_pdf_cues, plot_pdf3
import soundfile as sf
# ==========================================================
# === CONFIG ===============================================
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_MIX_6_SNR/TEST_time_domain_results_28_10_2025__23_29_49_0.mat"
MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_MIX_SNR_3/TEST_time_domain_results_29_10_2025__15_09_44_0.mat"
CLEAN_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Signal_Gen_with_rir"
CLEAN_PREFIX = "clean_example_"
INDEX =   6
FS = 16000
NOISE_HEAD_SEC = 0.5
NOISE_SAMPLES = int(NOISE_HEAD_SEC * FS)
TMP_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Binaural_plots/papers_audio"
# ==========================================================

# === Derive model name automatically from MAT_FILE path ===
MODEL_NAME = os.path.basename(os.path.dirname(MAT_FILE))

# === Auto-generate output filename ===
OUTPUT_FULL = os.path.join(
    TMP_DIR,
    f"compare_clean_estimated_itd_ild_{MODEL_NAME}_index_{INDEX}.png"
)
# ==========================================================
def extract_signals_from_mat(mat_path, clean_dir, index, noise_samples):
    """Extract estimated, mixture, and clean stereo signals correctly and save WAVs."""

    os.makedirs(TMP_DIR, exist_ok=True)
    mat = sio.loadmat(mat_path)
    fs = 16000
    if "fs" in mat:
        fs = int(np.array(mat["fs"]).squeeze())

    # === Estimated signals ===
    left  = mat["x_hat_stage1_left"].astype(np.float32)
    right = mat["x_hat_stage1_right"].astype(np.float32)
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left  /= max_val
        right /= max_val
    est_stereo = np.stack([left[index,:], right[index,:]], axis=-1)  # (T, 2)

    # === Mixture (use first and last mic, no truncation) ===
    y = mat["y"].astype(np.float32)
    mix_stereo = np.stack([y[index, :, 0], y[index, :, -1]], axis=-1)

    # === Clean (load from .mat in clean_dir, same index offset) ===
    clean_idx = index
    clean_mat = sio.loadmat(os.path.join(clean_dir, f"{CLEAN_PREFIX}{clean_idx:07d}.mat"))
    clean = clean_mat["clean"].astype(np.float32)
    clean_stereo = np.stack([clean[:, 0], clean[:, -1]], axis=-1)
    zeros = np.zeros((noise_samples, 2), dtype=np.float32)
    clean_stereo = np.concatenate([zeros, clean_stereo], axis=0)
    # === Single consistent normalization for plotting ===
    peak = np.max(np.abs(clean_stereo))
    if peak > 0:
        clean_stereo /= np.max(np.abs(clean_stereo))
        mix_stereo /= np.max(np.abs(mix_stereo))
        est_stereo /= np.max(np.abs(est_stereo))

    # === Save to disk ===
    clean_wav = os.path.join(TMP_DIR, f"clean_{index}.wav")
    est_wav   = os.path.join(TMP_DIR, f"estimated_{index}.wav")
    mix_wav   = os.path.join(TMP_DIR, f"mixture_{index}.wav")

    sf.write(clean_wav, clean_stereo, fs)
    sf.write(est_wav, est_stereo, fs)
    sf.write(mix_wav, mix_stereo, fs)

    print(f"[SAVED] Clean: {clean_wav}")
    print(f"[SAVED] Estimated: {est_wav}")
    print(f"[SAVED] Mixture: {mix_wav}")

    return clean_wav, est_wav, mix_wav, fs

def main():
    print(f"[INFO] Loading from: {MAT_FILE}")
    clean_wav, est_wav, mix_wav, fs = extract_signals_from_mat(MAT_FILE, CLEAN_DIR, INDEX, NOISE_SAMPLES)

    # --- Full scene comparison ---
    print("[INFO] Computing ILD–ITD PDFs (static case)...")
    res_clean = calc_pdf_cues(clean_wav)
    res_est   = calc_pdf_cues(est_wav)
    res_mix   = calc_pdf_cues(mix_wav)

    if res_clean and res_est and res_mix:
        ild_itd_pdf_clean, ild_pdf_clean, itd_pdf_clean, ildaxis, itdaxis = res_clean
        ild_itd_pdf_est, ild_pdf_est, itd_pdf_est, _, _ = res_est
        ild_itd_pdf_mix, ild_pdf_mix, itd_pdf_mix, _, _ = res_mix

        plt.figure(figsize=(8, 6))
        plot_pdf3(
            ild_itd_pdf1=ild_itd_pdf_mix, ild_pdf1=ild_pdf_mix, itd_pdf1=itd_pdf_mix,
            ild_itd_pdf2=ild_itd_pdf_est, ild_pdf2=ild_pdf_est, itd_pdf2=itd_pdf_est,
            ild_itd_pdf3=ild_itd_pdf_clean, ild_pdf3=ild_pdf_clean, itd_pdf3=itd_pdf_clean,
            axis_ild=ildaxis, axis_itd=itdaxis,
            color="blue"
        )
        plt.suptitle("Joint ILD–ITD Distributions — Mixture (green), Estimated (red), Clean (blue)", fontsize=12)
        plt.savefig(OUTPUT_FULL, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Full-scene plot → {OUTPUT_FULL}")
    else:
        print("[WARN] One or more ILD–ITD PDFs could not be computed.")

if __name__ == "__main__":
    main()
