import os, numpy as np, scipy.io as sio, csv

# ======== CONFIG ========
MAT_STFT = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/15_10_2025_CORRELATION/TEST_STFT_domain_results_16_10_2025__02_48_36_0.mat"
OUT_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Performance_evaluation"
BATCH_IDX = 0          # which utterance in the batch to analyze
L_MIC, R_MIC = 0, 7    # which mics represent "left" and "right" in X_STFT
D_BASELINE_M = 0.34    # ear/mic spacing in meters (SET THIS)
HOP_SAMPLES  = 128     # your STFT hop (samples). You often use 128 with fs=16k.
OUT_CSV      = os.path.join(os.path.dirname(OUT_DIR), f"gccphat_doa_from_stft_{BATCH_IDX:02d}.csv")

C_SOUND = 343.0

def to_complex_from_514(stacked):  # [*, 514, T] -> [*, 257, T] complex
    """Split along F: first 257 real, last 257 imag."""
    *prefix, F, T = stacked.shape
    assert F == 514, f"Expected 514 stacked bins, got {F}"
    real = stacked[(..., slice(0,257), slice(0,T))]
    imag = stacked[(..., slice(257,514), slice(0,T))]
    return real + 1j*imag

def ensure_complex(X):
    """Return complex array shaped [..., Fpos(=257), T]. Supports complex or 514-stacked real+imag."""
    X = np.array(X)
    if np.iscomplexobj(X):
        # assume already [..., 257, T]
        return X
    # try 514-stacked last two dims like [B, M, 514, T] or [B, 514, T]
    if X.ndim >= 3 and X.shape[-2] == 514:
        return to_complex_from_514(X)
    raise ValueError(f"Don't know how to interpret STFT shape {X.shape} (need complex or 514-stacked).")

def gcc_phat_tau_from_stft_pair(XL, XR, fs, d_baseline):
    """
    XL, XR: complex STFT [Fpos, T] (one utterance)
    Returns: tau[t] in seconds, and time axis (sec) based on hop
    """
    Fpos, T = XL.shape
    # cross-spectrum per frame
    R12 = XL * np.conj(XR)                    # [Fpos, T]
    Rmag = np.abs(R12) + 1e-12
    R12_phat = R12 / Rmag

    # Use symmetric IRFFT length for GCC (2*(Fpos-1))
    irfft_len = 2 * (Fpos - 1)
    r = np.fft.irfft(R12_phat, n=irfft_len, axis=0)   # [irfft_len, T], zero-lag at center
    center = irfft_len // 2

    # physics-based lag bound
    max_tau = d_baseline / C_SOUND
    max_lag = int(np.floor(max_tau * fs))
    lo, hi = center - max_lag, center + max_lag + 1
    lo = max(lo, 0); hi = min(hi, irfft_len)

    # argmax within physically allowed window
    window = r[lo:hi, :]                              # [2*max_lag+1, T]
    peak = np.argmax(np.abs(window), axis=0)          # [T]
    lag_samples = (peak + lo) - center                # signed lag in samples
    tau = lag_samples / float(fs)                     # seconds

    # time axis (approx from hop)
    t = np.arange(T) * (HOP_SAMPLES / float(fs))
    return t, tau

def tau_to_azimuth_deg(tau, d):
    x = np.clip((C_SOUND * tau) / d, -1.0, 1.0)
    return np.degrees(np.arcsin(x))

def angular_error_deg(theta_hat, theta_true):
    e = (theta_hat - theta_true + 180.0) % 360.0 - 180.0
    return np.abs(e)

def summarize_errors(err):
    err = err[~np.isnan(err)]
    if err.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "P90": np.nan, "Within5": np.nan, "Within10": np.nan}
    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "P90": float(np.percentile(err, 90)),
        "Within5": float(np.mean(err <= 5.0)),
        "Within10": float(np.mean(err <= 10.0)),
    }

# ----- LOAD MAT -----
mat = sio.loadmat(MAT_STFT)
if "fs" in mat: FS = int(np.array(mat["fs"]).squeeze())
else: raise ValueError("fs not found in MAT; please include sampling rate.")

# Pull STFT tensors
X_STFT  = mat["X_STFT"]                  # expect [B, M, 514, T] real+imag stacked OR complex [B,M,257,T]
XhL     = mat.get("X_hat_Stage1_C_left")     # [B, 257, T] complex (likely)
XhR     = mat.get("X_hat_Stage1_C_right")    # [B, 257, T] complex (likely)

if XhL is None or XhR is None:
    raise ValueError("X_hat_Stage1_C_left/right not found in MAT.")

# Ensure complex forms
X_STFT_c = ensure_complex(X_STFT)               # [B, M, 257, T] complex
XhL_c    = ensure_complex(XhL)                  # [B, 257, T] complex
XhR_c    = ensure_complex(XhR)                  # [B, 257, T] complex

# Select utterance
XL_ref = X_STFT_c[BATCH_IDX, L_MIC, :, :]       # [257, T]
XR_ref = X_STFT_c[BATCH_IDX, R_MIC, :, :]       # [257, T]
XL_est = XhL_c[BATCH_IDX, :, :]                 # [257, T]
XR_est = XhR_c[BATCH_IDX, :, :]                 # [257, T]

# ----- GCC-PHAT per frame -----
t_ref, tau_ref = gcc_phat_tau_from_stft_pair(XL_ref, XR_ref, FS, D_BASELINE_M)
t_est, tau_est = gcc_phat_tau_from_stft_pair(XL_est, XR_est, FS, D_BASELINE_M)

# Align lengths just in case
Tmin = min(len(t_ref), len(t_est))
t_ref, tau_ref = t_ref[:Tmin], tau_ref[:Tmin]
t_est, tau_est = t_est[:Tmin], tau_est[:Tmin]

theta_ref = tau_to_azimuth_deg(tau_ref, D_BASELINE_M)
theta_est = tau_to_azimuth_deg(tau_est, D_BASELINE_M)

# ----- Metrics -----
err = angular_error_deg(theta_est, theta_ref)
stats = summarize_errors(err)
print("\n[DOA Trajectory via GCC-PHAT on STFT]")
print(f"  MAE:   {stats['MAE']:.2f} deg")
print(f"  RMSE:  {stats['RMSE']:.2f} deg")
print(f"  P90:   {stats['P90']:.2f} deg")
print(f"  ≤5°:   {100*stats['Within5']:.1f}%")
print(f"  ≤10°:  {100*stats['Within10']:.1f}%")

# ----- Save CSV -----
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_sec", "theta_true_deg", "theta_est_deg", "abs_err_deg"])
    for i in range(Tmin):
        w.writerow([float(t_ref[i]), float(theta_ref[i]), float(theta_est[i]), float(err[i])])

print(f"[OK] DOA CSV saved: {OUT_CSV}")
