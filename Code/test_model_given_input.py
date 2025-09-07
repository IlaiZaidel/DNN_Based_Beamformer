#!/usr/bin/env python3
# demo: generate moving-speaker scene via SignalGenerator -> add noises -> run model -> save WAVs (no normalization)

import os, sys, math, ast, time, re
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import scipy.io as sio
import torch
from utils import return_as_complex
# ===================== USER SETTINGS =====================
INPUT_MODE = "signalgen_start_stop"  # or "rir_static" if you want the old static path

CSV_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
CSV_INDEX = 0

OUT_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Test_Model_Files"
MODEL_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Trained_Models/13_08_2025/trained_model_dataset_withoutRev.pt"

# SignalGenerator compiled module (.so)
SIGGEN_SO = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator/signal_generator.cpython-310-x86_64-linux-gnu.so"
SIGGEN_DIR = os.path.dirname(SIGGEN_SO)

# Core params
FS              = 16000
T_SEC           = 4.0
CK              = 343.0
WIN_LENGTH      = 512
HOP_STFT        = WIN_LENGTH // 4
MIC_REF_1B      = 4
NOISE_HEAD_SEC  = 0.5
SNR_NOISE_DB    = 3.0
SNR_WHITE_DB    = 30.0

# SignalGenerator params (match your generator)
NSAMPLES   = 1024
ORDER      = 0
HOP_AIR    = 32
M_TYPE     = "o"
HP_FILTER  = True
DIM        = 3
# ========================================================

# Make the compiled module importable
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)
from signal_generator import SignalGenerator  # compiled .so

# ===== Project imports =====
from omegaconf import OmegaConf
cfg = OmegaConf.load("DNN_Based_Beamformer/Code/conf/config.yaml")
from rir_generator import generate as rir_generate
from LoadPreTrainedModel import loadPreTrainedModel
from utils import Preprocesing, Postprocessing, return_as_complex

# ===================== UTIL =====================
def stamp_now():
    return time.strftime("%Y%m%d_%H%M%S")

model_folder = os.path.basename(os.path.dirname(MODEL_PATH))
model_date   = "_".join(model_folder.split("_")[:2])  # e.g. "13_08"
RUN_DIR = os.path.join(OUT_DIR, f"run_{model_date}_{stamp_now()}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def to_stereo(arr):
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.stack([arr[:, 0], arr[:, -1]], axis=-1).astype(np.float32)
    return arr.astype(np.float32)

def randomize_noise_angles(src_angle_deg, min_sep=30):
    pool = [a for a in range(181) if abs(a - src_angle_deg) >= min_sep]
    a1 = int(np.random.choice(pool))
    pool = [a for a in pool if abs(a - a1) >= min_sep]
    a2 = int(np.random.choice(pool))
    return a1, a2

# ===================== PRODUCERS =====================
def _build_paths_from_csv_row(row, fs_expected=FS):
    wav_path = row["speaker_path"]
    fs = int(row.get("fs", FS))
    T = float(row.get("T", T_SEC))
    if fs != fs_expected:
        raise RuntimeError(f"CSV fs={fs} does not match expected FS={fs_expected}")
    L = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
    beta = [float(row["beta"])] * 6
    mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)  # (M,3)
    M = mic_positions.shape[0]

    # read speech, tile to length N
    x, fs_file = sf.read(wav_path)
    if x.ndim > 1: x = x[:, 0]
    if fs_file != fs:
        raise RuntimeError(f"fs mismatch (wav:{fs_file}, csv:{fs})")
    N = int(T * fs)
    if len(x) < N:
        x = np.tile(x, int(math.ceil(N/len(x))))
    x = x[:N].astype(np.float64)

    # moving speaker: start -> stop
    start = np.array([float(row["speaker_start_x"]),
                      float(row["speaker_start_y"]),
                      float(row["speaker_start_z"])], dtype=np.float64)
    stop  = np.array([float(row["speaker_stop_x"]),
                      float(row["speaker_stop_y"]),
                      float(row["speaker_stop_z"])], dtype=np.float64)

    sp_path = np.zeros((N, 3), dtype=np.float64)
    rp_path = np.zeros((N, 3, M), dtype=np.float64)
    for i in range(0, N, HOP_AIR):
        alpha = i / max(1, (N - 1))
        sp = start + alpha * (stop - start)
        end = min(i + HOP_AIR, N)
        sp_path[i:end] = sp
        for m in range(M):
            rp_path[i:end, :, m] = mic_positions[m]

    # optional: stationary noise sources from CSV
    noise1_pos = None; noise2_pos = None
    if all(k in row for k in ["noise1_x","noise1_y","noise1_z"]):
        noise1_pos = [float(row["noise1_x"]), float(row["noise1_y"]), float(row["noise1_z"])]
    if all(k in row for k in ["noise2_x","noise2_y","noise2_z"]):
        noise2_pos = [float(row["noise2_x"]), float(row["noise2_y"]), float(row["noise2_z"])]

    # crude DOA for visualization if needed
    mic_center = mic_positions.mean(axis=0)
    delta = start[:2] - mic_center[:2]
    angle_src = int(np.clip(np.rad2deg(np.arctan2(delta[1], delta[0])), -90, 90) + 90)  # map to 0..180

    return (x, L, beta, sp_path, rp_path, mic_positions, angle_src, noise1_pos, noise2_pos)

def produce_signalgen_start_stop(csv_path, idx):
    """Generate clean multichannel via SignalGenerator (start->stop), then build directional + white noises (no normalization)."""
    df = pd.read_csv(csv_path)
    if idx < 0 or idx >= len(df):
        raise IndexError(f"CSV index {idx} out of range [0,{len(df)-1}]")
    row = df.iloc[idx].to_dict()

    x, L, beta, sp_path, rp_path, mic_positions, angle_src, noise1_pos_csv, noise2_pos_csv = _build_paths_from_csv_row(row)

    # run SignalGenerator -> clean speech at mics
    gen = SignalGenerator()
    result = gen.generate(
        list(x),             # input speech
        CK,                  # speed of sound
        FS,                  # fs
        rp_path.tolist(),    # r_path: [T][3][M]
        sp_path.tolist(),    # s_path: [T][3]
        L,                   # room dims
        beta,                # beta per wall
        NSAMPLES,            # nsamples
        M_TYPE,              # mic type
        ORDER,               # order
        DIM,                 # dimension
        [],                  # orientation (unused)
        HP_FILTER            # hp_filter
    )
    d_full = np.array(result.output, dtype=np.float64).T  # (N, M)
    M = d_full.shape[1]; N = d_full.shape[0]

    # noise-only head for direct speech
    nlead = int(NOISE_HEAD_SEC*FS); keep = N - nlead
    d = np.vstack([np.zeros((nlead, M)), d_full[:keep, :]])

    # stationary directional colored noise positions
    mic_center = mic_positions.mean(axis=0)
    radius = 1.2
    def pos_from_angle(deg):
        rad = np.deg2rad(deg)
        return [float(mic_center[0] + radius*np.cos(rad)),
                float(mic_center[1] + radius*np.sin(rad)),
                float(mic_center[2])]

    if noise1_pos_csv is None or noise2_pos_csv is None:
        angle_n1 = max(0, min(180, angle_src - 40))
        angle_n2 = max(0, min(180, angle_src + 40))
        noise1_pos = pos_from_angle(angle_n1)
        noise2_pos = pos_from_angle(angle_n2)
    else:
        noise1_pos = noise1_pos_csv
        noise2_pos = noise2_pos_csv
        v1 = np.array(noise1_pos) - mic_center
        v2 = np.array(noise2_pos) - mic_center
        angle_n1 = int(np.clip(np.rad2deg(np.arctan2(v1[1], v1[0])), -90, 90) + 90)
        angle_n2 = int(np.clip(np.rad2deg(np.arctan2(v2[1], v2[0])), -90, 90) + 90)

    # Build directional noise via static RIRs + AR drivers
    n_taps = int(row.get("n", max(256, NSAMPLES)))
    h_n1 = np.array(rir_generate(CK, FS, mic_positions, noise1_pos, L, beta, n_taps, order=0)).T
    h_n2 = np.array(rir_generate(CK, FS, mic_positions, noise2_pos, L, beta, n_taps, order=0)).T

    w = np.random.randn(N, 2); AR = [1.0, -0.7]
    drv1 = signal.lfilter([1.0], AR, w[:, 0]); drv2 = signal.lfilter([1.0], AR, w[:, 1])
    n1 = np.stack([signal.lfilter(h_n1[m], [1.0], drv1) for m in range(M)], axis=1)
    n2 = np.stack([signal.lfilter(h_n2[m], [1.0], drv2) for m in range(M)], axis=1)

    # SNR scaling of colored (directional) and white vs. clean @ ref mic
    ref = MIC_REF_1B - 1
    d_pow = np.sum(d[:, ref]**2)
    n_pow = np.sum(n1[:, ref]**2) + 1e-10
    scale_n = np.sqrt(d_pow * 10**(-SNR_NOISE_DB/10.0) / n_pow)
    n1 *= scale_n; n2 *= scale_n
    directional = n1  # choose one directional source; use (n1+n2) if you prefer combined

    white = np.random.randn(N, M)
    v_pow = np.sum(white[:, ref]**2) + 1e-10
    scale_v = np.sqrt(d_pow * 10**(-SNR_WHITE_DB/10.0) / v_pow)
    white *= scale_v

    # mixture (NO normalization)
    y = d + directional + white

    return y, d, directional, white, angle_src, angle_n1, angle_n2

def produce_rir_static():
    """Static scene generator (kept for completeness). No normalization here either."""
    x_lim = np.random.uniform(6.0, 9.0); y_lim = np.random.uniform(6.0, 9.0)
    ROOM_Z = 3.0
    L = [x_lim, y_lim, ROOM_Z]
    BETA_T60 = float(np.random.choice(np.arange(0.30, 0.55, 0.05)))
    beta6 = [BETA_T60]*6
    N_TAPS = int(FS * BETA_T60)
    RIR_ORDER = 0

    angle_orientation = np.random.randint(-45, 46)
    theta = np.deg2rad(angle_orientation)
    mic_center_x = np.random.uniform(2.0, x_lim - 2.0)
    mic_center_y = np.random.uniform(2.0, y_lim - 2.0)
    mic_height   = np.random.uniform(1.0, 1.5)
    mic_offsets = np.array([[-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
                            [ 0.04, 0], [ 0.07, 0], [ 0.12, 0], [ 0.17, 0]])
    Rmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])
    mic_xy = mic_offsets @ Rmat.T + np.array([mic_center_x, mic_center_y])
    mic_positions = np.column_stack([mic_xy, np.full((8,), mic_height)])

    # speech
    SPEECH_WAV = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28241/5105-28241-0011.wav"
    N = int(FS*T_SEC)
    x, fs_file = sf.read(SPEECH_WAV)
    if x.ndim > 1: x = x[:, 0]
    if fs_file != FS: raise RuntimeError(f"Sample-rate mismatch: wav={fs_file}, expected {FS}")
    if len(x) < N: x = np.tile(x, int(math.ceil(N/len(x))))
    x = x[:N].astype(np.float64)

    radius = np.random.uniform(1.0, 1.5)
    angle_src = np.random.randint(0, 181)
    sx = mic_center_x + radius*np.cos(np.deg2rad(angle_src + angle_orientation))
    sy = mic_center_y + radius*np.sin(np.deg2rad(angle_src + angle_orientation))
    sz = np.random.uniform(1.2, 1.8)
    src_pos = [float(sx), float(sy), float(sz)]

    angle_n1, angle_n2 = randomize_noise_angles(angle_src, min_sep=10)
    n1x = mic_center_x + radius*np.cos(np.deg2rad(angle_n1 + angle_orientation))
    n1y = mic_center_y + radius*np.sin(np.deg2rad(angle_n1 + angle_orientation))
    n1z = np.random.uniform(1.2, 1.9)
    noise1_pos = [float(n1x), float(n1y), float(n1z)]
    n2x = mic_center_x + radius*np.cos(np.deg2rad(angle_n2 + angle_orientation))
    n2y = mic_center_y + radius*np.sin(np.deg2rad(angle_n2 + angle_orientation))
    n2z = np.random.uniform(1.2, 1.9)
    noise2_pos = [float(n2x), float(n2y), float(n2z)]

    h_speech = np.array(rir_generate(CK, FS, mic_positions, src_pos,    L, beta6, N_TAPS, order=RIR_ORDER)).T
    h_n1     = np.array(rir_generate(CK, FS, mic_positions, noise1_pos, L, beta6, N_TAPS, order=RIR_ORDER)).T
    xM = np.tile(x[:, None], (1, mic_positions.shape[0]))
    d_full = np.stack([signal.fftconvolve(xM[:, m], h_speech[m], mode="full")[:N] for m in range(mic_positions.shape[0])], axis=1)

    nlead = int(NOISE_HEAD_SEC*FS); keep  = N - nlead
    d = np.vstack([np.zeros((nlead, mic_positions.shape[0]), dtype=np.float64), d_full[:keep, :]])

    # one directional (like above) + white
    w = np.random.randn(N); AR = [1.0, -0.7]
    drv1 = signal.lfilter([1.0], AR, w)
    n1 = np.stack([signal.lfilter(h_n1[m], [1.0], drv1) for m in range(mic_positions.shape[0])], axis=1)

    ref = MIC_REF_1B - 1
    d_pow = np.sum(d[:, ref]**2)
    n_pow = np.sum(n1[:, ref]**2) + 1e-10
    scale_n = np.sqrt(d_pow * 10**(-SNR_NOISE_DB/10.0) / n_pow)
    directional = n1 * scale_n

    white = np.random.randn(N, mic_positions.shape[0])
    v_pow = np.sum(white[:, ref]**2) + 1e-10
    scale_v = np.sqrt(d_pow * 10**(-SNR_WHITE_DB/10.0) / v_pow)
    white *= scale_v

    y = d + directional + white
    return y, d, directional, white, angle_src, angle_n1, angle_n2

# ===================== MAIN =====================
def main():
    model = loadPreTrainedModel(cfg).to(
        torch.device(f"cuda:{cfg.device.device_num}" if torch.cuda.is_available() else "cpu")
    ).eval()
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    if INPUT_MODE == "signalgen_start_stop":
        y, d, directional, white, angle_src, angle_n1, angle_n2 = produce_signalgen_start_stop(CSV_PATH, CSV_INDEX)
    elif INPUT_MODE == "rir_static":
        y, d, directional, white, angle_src, angle_n1, angle_n2 = produce_rir_static()
    else:
        raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE}")

    X_t = torch.from_numpy(y[None, ...]).float().to(device)   # (1,N,M)
    Y   = Preprocesing(X_t, WIN_LENGTH, FS, T_SEC, HOP_STFT, device)
    with torch.no_grad():
            outs = model(Y, device)
            W = return_as_complex(outs[0])


    # helper: run model on a multichannel signal and return beamformed mono
    def run_beamformer(x_multi, W):
        X_t = torch.from_numpy(x_multi[None, ...]).float().to(device)   # (1,N,M)
        Y   = Preprocesing(X_t, WIN_LENGTH, FS, T_SEC, HOP_STFT, device)
        Y = return_as_complex(Y)
        Z = torch.sum(torch.conj(W) * Y, dim=1) 
        xhat = Postprocessing(Z, HOP_STFT, WIN_LENGTH, device).cpu().numpy()[0]
        return xhat

    # ---- Run model on each component (no normalization anywhere) ----
    xhat_mix  = run_beamformer(y,  W)
    xhat_clean = run_beamformer(d,  W)
    xhat_dir   = run_beamformer(directional,  W)
    xhat_white = run_beamformer(white,  W)
    substraction = xhat_mix -xhat_clean -  xhat_dir - xhat_white
    # ---- Save WAVs ----
    base = os.path.join(RUN_DIR, "ex")
    # inputs (so you can hear powers directly)
    sf.write(base + "_mix.wav",   to_stereo(y),  FS)
    sf.write(base + "_clean.wav", to_stereo(d),  FS)
    # NOTE: per your request we DO NOT save the raw noise tracks anymore

    # beamformer outputs
    sf.write(base + "_mix_beam.wav",   xhat_mix.astype(np.float32),  FS)
    sf.write(base + "_clean_beam.wav", xhat_clean.astype(np.float32), FS)
    sf.write(base + "_dir_beam.wav",   xhat_dir.astype(np.float32),   FS)
    sf.write(base + "_white_beam.wav", xhat_white.astype(np.float32), FS)
    sf.write(base + "_substraction_beam.wav", substraction.astype(np.float32), FS)
    print("Saved to:", RUN_DIR)

if __name__ == "__main__":
    main()
