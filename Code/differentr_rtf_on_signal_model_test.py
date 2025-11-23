#!/usr/bin/env python3
# Apply weights from INDEX_SRC to signals of INDEX_TGT, save original & beamformed stereo
import os
import numpy as np
import scipy.io as sio
import soundfile as sf
import torch

# ==== Your project utils (ISTFT/STFT wrappers) ====
from utils import Preprocesing, Postprocessing, return_as_complex

# ===================== USER CONFIG =====================
MAT_FILE     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/05_10_2025/TEST_STFT_domain_results_05_10_2025__09_15_43_0.mat"
OUT_DIR      = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/xfer_weights_apply"
INDEX_SRC    = 3   # take weights from this index
INDEX_TGT    = 5   # apply them to this signal index
REF_LEFT     = 3   # reference mic index for "original left"
REF_RIGHT    = 4   # reference mic index for "original right"
WIN_LENGTH   = 512
HOP          = WIN_LENGTH // 4
FS_FALLBACK  = 16000
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ========================================================

os.makedirs(OUT_DIR, exist_ok=True)

def _get_first_existing(d, names):
    for n in names:
        if n in d:
            return d[n]
    return None

def _to_torch_complex(x):
    # scipy.io.loadmat returns complex dtype if saved as such in MATLAB.
    # If someone stored as struct with .real/.imag, handle here as well.
    if isinstance(x, np.ndarray) and np.iscomplexobj(x):
        return torch.from_numpy(x)
    if isinstance(x, dict) and "real" in x and "imag" in x:
        return torch.from_numpy(x["real"] + 1j * x["imag"])
    # as a last resort—return as torch (could be real)
    return torch.from_numpy(np.asarray(x))

def _ensure_BLFL(x):
    """
    Ensure shape is [B, M, F, L] or [B, F, L] as needed.
    We expect [B, M, F, L] for multichannel STFT tensors, and [B, M, F, L] for W.
    """
    x = x.squeeze()
    return x

def _match_time_len(W, Y):
    # match the time dimension L
    Lw = W.shape[-1]
    Ly = Y.shape[-1]
    L  = min(Lw, Ly)
    return W[..., :L], Y[..., :L]

def _istft_ref_channel(Y_bmfl, ref_idx, hop, win_len, device):
    # Y_bmfl: [B, M, F, L] complex
    # pick [B,F,L] of a ref mic and ISTFT
    y_ref = Y_bmfl[:, ref_idx, :, :]    # [B, F, L]
    y_ref_t = Postprocessing(y_ref.to(device), hop, win_len, device)  # [B, N]
    return y_ref_t

def _save_stereo(wav_path, left, right, fs):
    # left/right: torch tensors [B, N] or [N]
    if left.dim() == 2:
        left = left[0]
    if right.dim() == 2:
        right = right[0]
    x = torch.stack([left, right], dim=0).T.detach().cpu().numpy()  # [N, 2]
    sf.write(wav_path, x, fs)

def main():
    mat = sio.loadmat(MAT_FILE)

    # ---- sample rate ----
    fs = FS_FALLBACK
    if "fs" in mat:
        try:
            fs = int(np.array(mat["fs"]).squeeze())
        except Exception:
            pass

    # ---- weights ----
    Wl_np = _get_first_existing(mat, ["W_Stage1_left", "W_left", "Wl", "W_stage1_left"])
    Wr_np = _get_first_existing(mat, ["W_Stage1_right", "W_right", "Wr", "W_stage1_right"])
    if Wl_np is None or Wr_np is None:
        raise RuntimeError("Could not find left/right weights in MAT file.")

    Wl = _to_torch_complex(Wl_np).to(DEVICE)  # [B, M, F, L]
    Wr = _to_torch_complex(Wr_np).to(DEVICE)

    if Wl.dim() != 4 or Wr.dim() != 4:
        raise RuntimeError(f"Expected W dims [B,M,F,L], got {Wl.shape=} and {Wr.shape=}")

    B_w, M, F_w, L_w = Wl.shape
    if INDEX_SRC < 0 or INDEX_SRC >= B_w:
        raise IndexError(f"INDEX_SRC={INDEX_SRC} out of range (0..{B_w-1})")

    # slice source weights, keep batch dim = 1
    Wl_src = Wl[INDEX_SRC:INDEX_SRC+1, :, :, :]  # [1, M, F, L]
    Wr_src = Wr[INDEX_SRC:INDEX_SRC+1, :, :, :]

    # ---- signals (STFT) for target index ----
    # Try common key names for clean & babble STFT
    X_np = _get_first_existing(mat, ["X_stft", "X_STFT", "S_stft", "clean_stft", "speech_stft"])
    N_np = _get_first_existing(mat, ["N_babble_stft", "babble_stft", "B_stft", "N_stft", "noise_stft"])
    Y_np = _get_first_existing(mat, ["Y", "Y_stft", "mixture_stft"])  # optional prebuilt mixture

    if Y_np is None:
        # We’ll form Y = X + N (preferred)
        if X_np is None or N_np is None:
            raise RuntimeError("Missing STFTs. Need either Y, or both X_stft and N_babble_stft.")
        X = _to_torch_complex(X_np).to(DEVICE)  # expected [B, M, F, L]
        N = _to_torch_complex(N_np).to(DEVICE)
        if X.dim() != 4 or N.dim() != 4:
            raise RuntimeError(f"Expected X,N dims [B,M,F,L], got {X.shape=} and {N.shape=}")
        B_x, M_x, F_x, L_x = X.shape
        B_n, M_n, F_n, L_n = N.shape
        if INDEX_TGT < 0 or INDEX_TGT >= B_x:
            raise IndexError(f"INDEX_TGT={INDEX_TGT} out of range (0..{B_x-1})")
        if (M_x != M_n) or (F_x != F_n):
            raise RuntimeError("M/F mismatch between X and N.")
        # Slice target
        X_tgt = X[INDEX_TGT:INDEX_TGT+1, :, :, :]  # [1,M,F,L]
        N_tgt = N[INDEX_TGT:INDEX_TGT+1, :, :, :]
        # Time alignment with W (on L-dim). We’ll later also align W.
        Y_tgt = X_tgt + N_tgt  # [1,M,F,L]
    else:
        Y_all = _to_torch_complex(Y_np).to(DEVICE)  # [B,M,F,L]
        if Y_all.dim() != 4:
            raise RuntimeError(f"Expected Y dims [B,M,F,L], got {Y_all.shape=}")
        B_y, M_y, F_y, L_y = Y_all.shape
        if INDEX_TGT < 0 or INDEX_TGT >= B_y:
            raise IndexError(f"INDEX_TGT={INDEX_TGT} out of range (0..{B_y-1})")
        Y_tgt = Y_all[INDEX_TGT:INDEX_TGT+1, :, :, :]  # [1,M,F,L]
        M = M_y  # override M with signals M if mismatch

    # ---- frequency/time alignment between W and Y ----
    _, _, F_y, L_y = Y_tgt.shape
    if F_w != F_y:
        raise RuntimeError(f"Frequency mismatch: weights F={F_w}, signals F={F_y}.")
    Wl_src, Y_tgt = _match_time_len(Wl_src, Y_tgt)  # match L
    Wr_src, Y_tgt = _match_time_len(Wr_src, Y_tgt)
    # Conjugate weights for standard w^H y
    WlH = torch.conj(Wl_src)  # [1,M,F,L]
    WrH = torch.conj(Wr_src)

    # ---- ORIGINAL mixture (stereo) from reference mics ----
    # Inverse STFT of the target mixture at the chosen ref mics:
    orig_left_t  = _istft_ref_channel(Y_tgt, REF_LEFT,  HOP, WIN_LENGTH, DEVICE)   # [1,N]
    orig_right_t = _istft_ref_channel(Y_tgt, REF_RIGHT, HOP, WIN_LENGTH, DEVICE)   # [1,N]

    # ---- BEAMFORMED (stereo) using weights from INDEX_SRC ----
    # Z = sum_m W^*[:,m,:,:] * Y[:,m,:,:] over m
    Z_left  = torch.sum(WlH * Y_tgt, dim=1)  # [1, F, L]
    Z_right = torch.sum(WrH * Y_tgt, dim=1)  # [1, F, L]

    bf_left_t  = Postprocessing(Z_left.to(DEVICE),  HOP, WIN_LENGTH, DEVICE)   # [1,N]
    bf_right_t = Postprocessing(Z_right.to(DEVICE), HOP, WIN_LENGTH, DEVICE)   # [1,N]

    # ---- Length alignment for saving (guard small off-by-one) ----
    Nmin = min(orig_left_t.shape[-1], orig_right_t.shape[-1], bf_left_t.shape[-1], bf_right_t.shape[-1])
    orig_left_t  = orig_left_t[..., :Nmin]
    orig_right_t = orig_right_t[..., :Nmin]
    bf_left_t    = bf_left_t[..., :Nmin]
    bf_right_t   = bf_right_t[..., :Nmin]

    # ---- Save WAVs ----
    base = f"srcW_{INDEX_SRC}_tgtSig_{INDEX_TGT}_fs{fs}"
    wav_orig = os.path.join(OUT_DIR, f"orig_stereo_{base}.wav")
    wav_bf   = os.path.join(OUT_DIR, f"bf_stereo_{base}.wav")

    _save_stereo(wav_orig, orig_left_t, orig_right_t, fs)
    _save_stereo(wav_bf,   bf_left_t,   bf_right_t,   fs)

    print(f"[OK] Wrote original stereo: {wav_orig}")
    print(f"[OK] Wrote beamformed stereo: {wav_bf}")
    print("Done.")

if __name__ == "__main__":
    with torch.no_grad():
        main()
