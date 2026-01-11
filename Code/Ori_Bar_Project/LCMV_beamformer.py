#!/usr/bin/env python3
import os
import math
import ast
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import scipy
from rir_generator import generate as rir_generat
import matplotlib.pyplot as plt   ### >>> for plots
from datetime import datetime     ### >>> for filenames

# ==================== USER CONSTANTS ====================

SNR_BABBLE_DB = 15.0           # SNR of babble wrt (d1 + d2) @ ref mic
SNR_WHITE_DB  = 30.0          # SNR of white wrt (d1 + d2) @ ref mic
ORDER         = 4096          # RIR length (number of taps)
FS            = 16000         # Sampling rate [Hz]
T_SEC         = 4.0           # total duration [s]
NOISE_ONLY_T  = 0.5           # leading noise-only duration [s]
REF_MIC       = 4             # 1-based reference mic index
C_K           = 343.0         # speed of sound [m/s]

idx = 0

# Angles of the two speakers, relative to array orientation (degrees)
ANGLE_1_DEG = 30.0    # target (e.g. broadside)
ANGLE_2_DEG = 60.0   # interferer (e.g. 60° to the right)
ANGLE_3_DEG = 90.0   # interferer (e.g. 60° to the right)

G_VEC = np.array([2.0, 0.0, 0.0], dtype=np.complex128)

DF_PATH_TEST  = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
BABBLE_TEST_DIR  = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Babble_Noise/Test"

OUT_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ori_Bar_Project"
os.makedirs(OUT_DIR, exist_ok=True)

### >>> where to save beampattern plots
BEAM_OUT_DIR = os.path.join(OUT_DIR, "LCMV_beampatterns")
os.makedirs(BEAM_OUT_DIR, exist_ok=True)

# how many examples to process (None = all)
MAX_EXAMPLES = 10

# LibriSpeech path fix
OLD_PREFIX = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"
NEW_PREFIX = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"

# --------- STFT / ISTFT settings ---------
WIN_LEN = 512          # n_fft for speech STFT and H1/H2 DFT
HOP     = 128          # hop length (you can set 256 if you prefer)
DEVICE  = torch.device("cpu")   # keep it simple; change to "cuda" if you want

### >>> surround dataset for beampattern computation
SURROUND_BASE = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround/"

def stft_multichannel(y_np, win_len, hop, device):
    """
    y_np : (N, M)  float32
    returns: Y : (M, F, L) complex64
    """
    y_t = torch.from_numpy(y_np).to(device)           # (N, M)
    y_t = y_t.transpose(0, 1)                        # (M, N)
    w = torch.hamming_window(win_len, device=device)
    Y = torch.stft(
        y_t,
        n_fft=win_len,
        hop_length=hop,
        win_length=win_len,
        window=w,
        center=False,
        return_complex=True,
    )                                                # (M, F, L)
    return Y

def istft_single_channel(X_hat, win_len, hop, device, length):
    """
    X_hat : (F, L) complex
    returns: x_hat : (length,) float32
    """
    w = torch.hamming_window(win_len, device=device)
    x_hat = torch.istft(
        X_hat,
        n_fft=win_len,
        hop_length=hop,
        win_length=win_len,
        window=w,
        center=False,
        length=length,
        return_complex=False,
    )
    return x_hat

### >>> ========== BEAMPATTERN HELPERS (MINIMAL) ==========

def compute_beampattern_amplitudes(W_freq):
    """
    Compute beampattern using surround white-noise RIR dataset.

    W_freq : (F, M) complex numpy array (LCMV weights)
    Returns:
        amplitudes_sq_db : (181,)   power vs DOA (dB, max=0)
        amplitudes_spec_db : (181, F) DOA–frequency energy (dB)
    """
    Fbins, M = W_freq.shape
    amplitudes_sq = np.zeros(181, dtype=np.float64)
    amplitudes_spec = np.zeros((181, Fbins), dtype=np.float64)

    for ang in range(181):
        mat_path = os.path.join(
            SURROUND_BASE, f"my_surround_feature_vector_angle_{ang}.mat"
        )
        data = scipy.io.loadmat(mat_path)
        feat = data["feature"].astype(np.float32)  # assume shape (T, M)

        # STFT: (T,M) -> (M,F,L)
        Y_t = stft_multichannel(feat, WIN_LEN, HOP, DEVICE)   # torch complex
        Y = Y_t.detach().cpu().numpy()                        # (M, F, L)

        # beamform: Z(f,l) = sum_m w*(f,m) Y(m,f,l)
        L = Y.shape[2]
        Z = np.zeros((Fbins, L), dtype=np.complex64)

        for k in range(Fbins):
            w_k = W_freq[k, :]               # (M,)
            Y_k = Y[:, k, :]                 # (M, L)
            Z[k, :] = np.conj(w_k) @ Y_k     # (L,)

        # total energy for this DOA
        E_total = np.sum(np.abs(Z) ** 2)
        amplitudes_sq[ang] = E_total

        # frequency-wise energy (sum over time)
        amplitudes_spec[ang, :] = np.sum(np.abs(Z) ** 2, axis=1)

    # dB scaling
    max_val = np.max(amplitudes_sq) + 1e-12
    amplitudes_sq_db = 10.0 * np.log10(amplitudes_sq / max_val + 1e-12)

    amplitudes_spec_db = 10.0 * np.log10(amplitudes_spec + 1e-12)

    return amplitudes_sq_db, amplitudes_spec_db


def plot_beampattern(amplitudes_sq_db, angle1_deg, angle2_deg,angle3_deg, example_idx):
    angles = np.arange(181)
    amps = amplitudes_sq_db  # (181,)

    angles_rad = np.deg2rad(angles)
    mapped_angles = np.pi / 2 - angles_rad  # so 0° = top, +/-90° sides

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))
    ax.plot(mapped_angles, amps, label="Power (dB)")

    # mark the two speakers (angles are relative to array, same as surround data)
    for a, color, label in zip(
        [angle1_deg, angle2_deg, angle3_deg],
        ["red", "blue", "green"],
        ["Speaker 1", "Speaker 2", "Speaker 3"],
    ):
        ang_rad = np.pi / 2 - np.deg2rad(a)
        ax.plot([ang_rad], [0], marker="o", color=color, label=label)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([amps.min() - 3, 1])  # margin
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right")

    plt.title(f"LCMV Beampattern (Example {example_idx})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        BEAM_OUT_DIR, f"LCMV_beampattern_example_{example_idx:07d}_{ts}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Polar beampattern saved to {out_path}")


def plot_spectrogram(amplitudes_spec_db, angle1_deg, angle2_deg, angle3_deg, example_idx):
    angles = np.arange(181)
    Fbins = amplitudes_spec_db.shape[1]
    freqs = np.linspace(0, FS / 2, Fbins)

    amps = amplitudes_spec_db.T  # (F, 181) for imshow

    plt.figure(figsize=(12, 6))
    plt.imshow(
        amps,
        aspect="auto",
        extent=[angles.min(), angles.max(), freqs.min(), freqs.max()],
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("DOA (degrees)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"LCMV Frequency vs DOA (Example {example_idx})")

    for a, color, label in zip(
        [angle1_deg, angle2_deg, angle3_deg],
        ["red", "blue", "green"],
        ["Speaker 1", "Speaker 2", "Speaker 3"],
    ):
        plt.axvline(x=a, color=color, linestyle="--", linewidth=2, label=label)
        plt.text(
            a + 1,
            freqs.max() * 0.9,
            f"{label}\n{a:.0f}°",
            color=color,
            fontsize=9,
            rotation=90,
            va="top",
        )

    plt.legend(loc="upper right")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        BEAM_OUT_DIR, f"LCMV_spectrogram_example_{example_idx:07d}_{ts}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Spectrogram saved to {out_path}")


# --------- MAIN ---------


def main():
    df = pd.read_csv(DF_PATH_TEST)

    N = int(FS * T_SEC)
    noise_only_samples = int(NOISE_ONLY_T * FS)
    speech_samples = N - noise_only_samples

    # mic_ref in code is 1-based
    ref = REF_MIC - 1

    # loop over rows
    for idx in range(1):
        if MAX_EXAMPLES is not None and idx >= MAX_EXAMPLES:
            break

        row1 = df.iloc[idx]
        row2 = df.iloc[(idx + 1) % len(df)]  # second speaker uses next row
        row3 = df.iloc[(idx + 2) % len(df)]  # second speaker uses next row
        # --------------- Room / geometry from row1 ---------------
        L = [
            float(row1["room_x"]),
            float(row1["room_y"]),
            float(row1["room_z"]),
        ]
        beta = [float(row1["beta"])] * 6

        mic_positions = np.array(
            ast.literal_eval(row1["mic_positions"]), dtype=np.float64
        )
        M = mic_positions.shape[0]

        # center and radius used when generating the CSV
        mic_x = float(row1["mic_x"])
        mic_y = float(row1["mic_y"])
        mic_z = float(row1["mic_z"])               # not strictly needed here, but useful
        radius = float(row1["radius"])

        # array orientation (deg) as used in create_room
        angleOrientation = float(row1["angleOrientation"])

        # use speaker_start_z as a reasonable speech height
        speaker_z = float(row1["speaker_start_z"])

        # ----- Choose your own DOAs relative to angleOrientation -----
        # absolute angles in the room (deg)
        abs_angle_1 = angleOrientation + ANGLE_1_DEG
        abs_angle_2 = angleOrientation + ANGLE_2_DEG
        abs_angle_3 = angleOrientation + ANGLE_3_DEG
        # convert to radians
        a1_rad = np.deg2rad(abs_angle_1)
        a2_rad = np.deg2rad(abs_angle_2)
        a3_rad = np.deg2rad(abs_angle_3)
        # positions of the two static sources at distance `radius`
        spk1_pos = [
            mic_x + radius * np.cos(a1_rad),
            mic_y + radius * np.sin(a1_rad),
            speaker_z,
        ]
        spk2_pos = [
            mic_x + radius * np.cos(a2_rad),
            mic_y + radius * np.sin(a2_rad),
            speaker_z,
        ]
        spk3_pos = [
            mic_x + radius * np.cos(a3_rad),
            mic_y + radius * np.sin(a3_rad),
            speaker_z,
        ]
        # --------------- Load WAVs (no onset trimming) ---------------
        wav1 = row1["speaker_path"]
        if isinstance(wav1, str) and wav1.startswith(OLD_PREFIX):
            wav1 = wav1.replace(OLD_PREFIX, NEW_PREFIX)

        x1, fs1 = sf.read(wav1)
        if x1.ndim > 1:
            x1 = x1[:, 0]

        wav2 = row2["speaker_path"]
        if isinstance(wav2, str) and wav2.startswith(OLD_PREFIX):
            wav2 = wav2.replace(OLD_PREFIX, NEW_PREFIX)

        x2, fs2 = sf.read(wav2)
        if x2.ndim > 1:
            x2 = x2[:, 0]
        
        wav3 = row3["speaker_path"]
        if isinstance(wav3, str) and wav3.startswith(OLD_PREFIX):
            wav3 = wav3.replace(OLD_PREFIX, NEW_PREFIX)

        x3, fs2 = sf.read(wav3)
        if x3.ndim > 1:
            x3 = x3[:, 0]

        # ensure long enough: tile, then cut straight
        if len(x1) < speech_samples:
            reps = math.ceil(speech_samples / len(x1))
            x1 = np.tile(x1, reps)
        if len(x2) < speech_samples:
            reps = math.ceil(speech_samples / len(x2))
            x2 = np.tile(x2, reps)
        if len(x3) < speech_samples:
            reps = math.ceil(speech_samples / len(x3))
            x3 = np.tile(x3, reps)

        x1_speech = x1[:speech_samples].astype(np.float32)
        x2_speech = x2[:speech_samples].astype(np.float32)
        x3_speech = x3[:speech_samples].astype(np.float32)

        x1_full = np.concatenate(
            [np.zeros(noise_only_samples, dtype=np.float32), x1_speech],
            axis=0,
        )
        x2_full = np.concatenate(
            [np.zeros(noise_only_samples, dtype=np.float32), x2_speech],
            axis=0,
        )
        x3_full = np.concatenate(
            [np.zeros(noise_only_samples, dtype=np.float32), x3_speech],
            axis=0,
        )

        # --------------- RIRs for two static speakers ---------------
        h1 = np.array(
            rir_generat(C_K, FS, mic_positions, spk1_pos, L, beta, ORDER)
        ).T.astype(np.float32)  # shape (M, ORDER)

        h2 = np.array(
            rir_generat(C_K, FS, mic_positions, spk2_pos, L, beta, ORDER)
        ).T.astype(np.float32)  # shape (M, ORDER)

        h3 = np.array(
            rir_generat(C_K, FS, mic_positions, spk3_pos, L, beta, ORDER)
        ).T.astype(np.float32)  # shape (M, ORDER)


        # --------------- Convolve to multichannel signals ---------------
        d1 = np.zeros((N, M), dtype=np.float32)
        d2 = np.zeros((N, M), dtype=np.float32)
        d3 = np.zeros((N, M), dtype=np.float32)
        for m in range(M):
            d1[:, m] = signal.lfilter(h1[m], [1.0], x1_full)[:N]
            d2[:, m] = signal.lfilter(h2[m], [1.0], x2_full)[:N]
            d3[:, m] = signal.lfilter(h3[m], [1.0], x3_full)[:N]

        d = d1 + d2 + d3

        # --------------- Babble noise (precomputed multichannel) ---------------
        babble_path = os.path.join(BABBLE_TEST_DIR, f"babble_{idx:07d}.wav")
        babble, sr_b = sf.read(babble_path, always_2d=True)
        babble = babble.astype(np.float32)

        if babble.shape[0] < N:
            reps = math.ceil(N / babble.shape[0])
            babble = np.tile(babble, (reps, 1))
        babble = babble[:N, :M]

        # --------------- Scale babble to desired SNR ---------------
        d_power = float(np.sum(d[:, ref] ** 2) + 1e-12)
        b_power = float(np.sum(babble[:, ref] ** 2) + 1e-12)
        scale_b = np.sqrt(d_power * 10.0 ** (-SNR_BABBLE_DB / 10.0) / b_power).astype(
            np.float32
        )
        babble *= scale_b

        # --------------- White noise ---------------
        v = np.random.randn(N, M).astype(np.float32)
        v_power = float(np.sum(v[:, ref] ** 2) + 1e-12)
        scale_v = np.sqrt(d_power * 10.0 ** (-SNR_WHITE_DB / 10.0) / v_power).astype(
            np.float32
        )
        v *= scale_v

        # --------------- Final mixture ---------------
        y = d + babble + v

        # --------------- LCMV BEAMFORMER (STFT-BASED) ---------------
        # DFT of RIRs: steering vectors a1(f), a2(f) with nfft = WIN_LEN
        H1 = np.fft.rfft(h1, n=WIN_LEN, axis=1)   # (M, F)
        H2 = np.fft.rfft(h2, n=WIN_LEN, axis=1)   # (M, F)
        H3 = np.fft.rfft(h3, n=WIN_LEN, axis=1)   # (M, F)
        Fbins = H1.shape[1]

        # STFT of mixture y and noise-only part (babble + white)
        noise = (babble + v).astype(np.float32)

        Y_stft = stft_multichannel(y.astype(np.float32), WIN_LEN, HOP, DEVICE)      # (M, F, L)
        Noise_stft = stft_multichannel(noise, WIN_LEN, HOP, DEVICE)  # (M, F, L_noise)

        Y_stft_np = Y_stft.detach().cpu().numpy()          # (M, F, L)
        Noise_stft_np = Noise_stft.detach().cpu().numpy()  # (M, F, L_noise)

        M, F, L  = Y_stft_np.shape
        _, _, Ln = Noise_stft_np.shape

        W = np.zeros((Fbins, M), dtype=np.complex128)
        
        eps_reg = 1e-6

        for k in range(Fbins):
            a1_k = H1[:, k].astype(np.complex128)      # (M,)
            a2_k = H2[:, k].astype(np.complex128)      # (M,)
            a3_k = H3[:, k].astype(np.complex128)
            C = np.stack([a1_k, a2_k, a3_k], axis=1)         # (M, 2)

            # Noise covariance from noise-only STFT: Rv(f) = 1/Ln * sum_l v_l v_l^H
            V = Noise_stft_np[:, k, :]                 # (M, Ln)
            Rv = (V @ V.conj().T) / float(L)
            Rv = Rv + eps_reg * np.eye(M, dtype=np.complex128)

            # LCMV: w = Rv^{-1} C (C^H Rv^{-1} C)^{-1} g
            Rv_inv_C = np.linalg.solve(Rv, C)                 # (M, 2)
            CH_Rv_inv_C = C.conj().T @ Rv_inv_C               # (2, 2)
            middle = np.linalg.solve(CH_Rv_inv_C, G_VEC)      # (2,)
            w_k = Rv_inv_C @ middle                           # (M,)

            W[k, :] = w_k

        ### >>> ====== BEAMPATTERN COMPUTATION (using W) ======
        amplitudes_sq_db, amplitudes_spec_db = compute_beampattern_amplitudes(W)
        # ANGLE_1_DEG, ANGLE_2_DEG are the DOAs relative to the array,
        # which match the surround dataset angles.
        plot_beampattern(amplitudes_sq_db, ANGLE_1_DEG, ANGLE_2_DEG,ANGLE_3_DEG , idx)
        plot_spectrogram(amplitudes_spec_db, ANGLE_1_DEG, ANGLE_2_DEG,ANGLE_3_DEG, idx)
        ### >>> ==============================================

        # apply beamformer per-bin to STFT of mixture
        X_hat_spec = np.zeros((Fbins, L), dtype=np.complex128)
        for k in range(Fbins):
            w_k = W[k, :]                          # (M,)
            Y_k = Y_stft_np[:, k, :]               # (M, L)
            X_hat_spec[k, :] = np.conj(w_k) @ Y_k  # (L,)

        # ISTFT back to time domain
        X_hat_torch = torch.from_numpy(X_hat_spec).to(DEVICE)   # (F, L), complex
        x_hat = istft_single_channel(X_hat_torch, WIN_LEN, HOP, DEVICE, length=N)
        x_hat = x_hat.cpu().numpy().astype(np.float32)

        # --------------- Save WAVs ---------------

        # multichannel mixture Y (N, M)
        y_float = y.astype(np.float32)

        # single-channel LCMV output (N,)
        x_float = x_hat.astype(np.float32)

        # normalization (independent per file)
        y_out =  y_float / np.max(np.abs(y_float))
        x_out =  x_float / np.max(np.abs(x_float))

        # (1) stereo mixture: first and last microphone
        out_path_Y = os.path.join(
            OUT_DIR, f"example_{idx:07d}_Y_multich.wav"
        )
        y_stereo = np.stack([y_out[:, 0], y_out[:, -1]], axis=1).astype(np.float32)
        sf.write(out_path_Y, y_stereo, FS)

        # (2) single-channel LCMV output
        out_path_LCMV = os.path.join(
            OUT_DIR, f"example_{idx:07d}_LCMV.wav"
        )
        sf.write(out_path_LCMV, x_out, FS)

        print(f"Saved: {out_path_Y}")
        print(f"Saved: {out_path_LCMV}")


if __name__ == "__main__":
    main()
