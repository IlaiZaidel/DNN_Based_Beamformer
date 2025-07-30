import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import rir_generator

# --- Custom modules ---
from UnetModel import UNET
from Beamformer import beamformingOperationStage1, MaskOperationStage2
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening, noise_estimation
from utils import beamformingOpreation, Postprocessing, Preprocesing, return_as_complex, snr, save_vector_as_wav, save_feature_as_wav, place_source_2d
from RTF_from_clean_speech import RTF_from_clean_speech
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking
# In Python, add this early in the script

torch.cuda.empty_cache()
# -------------------------
# Parameters
# -------------------------
M = 8
fs = 16000
c = 343
T = 8.0
noise_only_time = 0.5
win_length = 512
R = win_length // 4

# -------------------------
# Room and Signal Setup
# -------------------------
room_dim = [7.168, 7.451, 3]
beta = np.array([0.5] * 6) #0.5
rir_len = 4800

mic_positions = np.array([
    [3.7556, 3.2449, 1.0638],
    [3.8047, 3.2353, 1.0638],
    [3.8538, 3.2258, 1.0638],
    [3.8832, 3.2201, 1.0638],
    [3.9618, 3.2048, 1.0638],
    [3.9912, 3.1991, 1.0638],
    [4.0403, 3.1895, 1.0638],
    [4.0894, 3.1800, 1.0638],
])
mic_center = np.mean(mic_positions, axis=0)

# Define angles and distance
az1 = 0    # degrees
az2 = 90    # degrees
az3 = 180    # degrees
az_noise = 45   # degrees
distance = 2.0  # meters

source_position_1 = place_source_2d(az1, distance, mic_center)
source_position_2 = place_source_2d(az2, distance, mic_center)
source_position_3 = place_source_2d(az3, distance, mic_center)
noise_position  = place_source_2d(az_noise, distance, mic_center)
# source_position = [2.3491, 5.1222, 1.4190]
# noise_position = [5.1214, 3.6328, 1.6707]

# -------------------------
# Load and Prepare Source Signal
# -------------------------
x1, _ = sf.read("/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28241/5105-28241-0011.wav")
x2, _ = sf.read("/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/3729/6852/3729-6852-0034.wav")
x3, _ = sf.read("/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/8224/274381/8224-274381-0005.wav")

half_len = int((T - noise_only_time) * fs / 3) +1
x1, x2,x3 = x1[:half_len], x2[:half_len], x3[:half_len]

x1_multi = np.tile(x1[:, None], (1, M))
x2_multi = np.tile(x2[:, None], (1, M))
x3_multi = np.tile(x3[:, None], (1, M))

ORDER = 0
# -------------------------
# RIR Generation and Signal Propagation
# -------------------------
h_src1 = np.array(rir_generator.generate(c, fs, mic_positions, source_position_1, room_dim, beta, rir_len, order=ORDER)).T
h_src2 = np.array(rir_generator.generate(c, fs, mic_positions, source_position_2, room_dim, beta, rir_len, order=ORDER)).T
h_src3 = np.array(rir_generator.generate(c, fs, mic_positions, source_position_3, room_dim, beta, rir_len, order=ORDER)).T
h_noise = np.array(rir_generator.generate(c, fs, mic_positions, noise_position, room_dim, beta, rir_len, order=0)).T



# # Direct path speech convolution
# d = np.stack([
#     signal.fftconvolve(x_multi[:, m], h_src[m, :], mode='full')[:x.shape[0]]
#     for m in range(M)
# ], axis=1)

# === Convolve each speaker ===
d1 = np.stack([signal.fftconvolve(x1_multi[:, m], h_src1[m, :], mode='full')[:half_len] for m in range(M)], axis=1)
d2 = np.stack([signal.fftconvolve(x2_multi[:, m], h_src2[m, :], mode='full')[:half_len] for m in range(M)], axis=1)
d3 = np.stack([signal.fftconvolve(x3_multi[:, m], h_src3[m, :], mode='full')[:half_len] for m in range(M)], axis=1)
d_cat = np.concatenate((d1, d2,d3), axis=0)



# Add zero-padding for noise-only segment
zero_pad = np.zeros((int(fs * noise_only_time), M))
d_padded = np.vstack((zero_pad, d_cat[:int(fs * (T - noise_only_time)), :]))  # (T*fs, M)

# -------------------------
# Simulate Noise
# -------------------------
w = np.random.randn(int(T * fs), 2)
AR = [1, -0.7]  # AR filter coefficients
noise_sources = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

n1 = np.stack([
    signal.lfilter(h_noise[m, :], [1], noise_sources[:, 0])
    for m in range(M)
], axis=1)  # Shape: (T*fs, M)



SNR_noise = 3  # Desired SNR in dB
mic_ref = 3   # zero-based index? If MATLAB style, 1-based.

# Calculate power of speech and noise at the reference mic
power_speech = np.sum(d_cat[:, mic_ref] ** 2)
power_noise = np.sum(n1[:, mic_ref] ** 2) + 1e-10  # avoid div by zero

# Calculate scaling factor for noise to get desired SNR
dVSn = np.sqrt(power_speech / (power_noise * 10 ** (SNR_noise / 10)))

# Scale noise
n1 = n1 * dVSn



# -------------------------
# Mixture Creation
# -------------------------
y = d_padded + n1
max_y = np.abs(y).max()
y /= max_y


y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, T*fs, M)

# -------------------------
# STFT and Preprocessing
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_tensor = y_tensor.to(device)
Y = Preprocesing(y_tensor, win_length, fs, T, R, device)  # (1, 8, 514, 497)
Y_stft = return_as_complex(Y)  # (1, 8, 257, 497)

# -------------------------
# Subspace Tracking (PASTd)
# -------------------------
Ln = int(noise_only_time * fs // R)
W, eigvals, eigvecs = pastd_rank1_whitened(Y_stft, noise_only_time, beta=0.995)
a_hat = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time) #torch.Size([1, 257, 8, 435])

# -------------------------
# RTF Estimation via Covariance Whitening
# -------------------------
rtf_cw = covariance_whitening(Y_stft, noise_only_time)  # (1, 8, 257, 1)
rtf_cw = fix_covariance_whitening(rtf_cw)
# -------------------------
# Noise Covariance and Inversion
# -------------------------
Rnn = noise_estimation(Y_stft, Ln)  # (1, 257, 8, 8)
epsilon = 1e-6
eye = torch.eye(Rnn.size(-1), device=Rnn.device).unsqueeze(0)  # (1, 8, 8)
Rnn_reg = Rnn + epsilon * eye
Rnn_inv = torch.linalg.inv(Rnn_reg)  # (1, 257, 8, 8)

# -------------------------
# Compute MVDR Weights
# -------------------------
# Align rtf_cw to (1, 257, 8, 1)
a = rtf_cw.permute(0, 2, 1, 3)  # (1, 257, 8, 1)

# Ra = Rnn_inv @ a → (1, 257, 8, 1)
Ra = torch.matmul(Rnn_inv, a)  # (1, 257, 8, 1)
Ra = Ra.permute(0, 2, 1, 3).squeeze(-1)  # (1, 8, 257)

# Compute denominator: a^H @ Rnn_inv @ a → (1, 257, 1, 1)
a_H = a.conj().transpose(-2, -1)  # (1, 257, 1, 8)
s = torch.matmul(a_H, torch.matmul(Rnn_inv, a))  # (1, 257, 1, 1)
s = s.squeeze(-1).squeeze(-1)  # (1, 257)

# Final MVDR weights
W_mvdr = Ra / s  # (1, 8, 257)
# Ensure W_mvdr is shape: (1, 8, 257, 1)
W_mvdr = W_mvdr.unsqueeze(-1)  # already done earlier

# Take Hermitian transpose: (1, 257, 1, 8)
W_H = W_mvdr.permute(0, 2, 3, 1).conj()

# Y_stft: (1, 8, 257, T) → permute to (1, 257, 8, T)
Y_aligned = Y_stft.permute(0, 2, 1, 3)

# Matrix multiplication: (1, 257, 1, 8) @ (1, 257, 8, T) → (1, 257, 1, T)
Y_bf = torch.matmul(W_H, Y_aligned)

# Squeeze to (1, 257, T)
Y_bf = Y_bf.squeeze(2)


speech_hat =  Postprocessing(Y_bf,R,win_length,device) #torch.Size([1, 64000])


save_vector_as_wav(y[:,mic_ref], output_wav_path='mvdr_noisy_signal.wav', sample_rate=16000)
save_vector_as_wav(speech_hat, output_wav_path='mvdr_output.wav', sample_rate=16000)




# -------------------------
# MVDR For Subspace Tracking
# -------------------------

B, F, M, T = a_hat.shape

# Ensure Y_stft shape is (B, F, M, T)
Y_stft = Y_stft.permute(0, 2, 1, 3).contiguous()

# Result holder for MVDR weights
W_mvdr_time = torch.zeros((B, F, M, T), dtype=torch.cfloat, device=Y_stft.device)

for t in range(T):
    # RTF estimate at time t: (B, F, M)
    a_t = a_hat[:, :, :, t].contiguous()

    # Hermitian transpose: (B, F, 1, M)
    a_t_H = a_t.conj().unsqueeze(2).transpose(-1, -2)

    # MVDR numerator: Rnn_inv @ a → (B, F, M, 1)
    Ra = torch.matmul(Rnn_inv, a_t.unsqueeze(-1))

    # MVDR denominator: a^H @ Rnn_inv @ a → (B, F, 1, 1) → squeeze → (B, F)
    denom = torch.matmul(a_t.conj().unsqueeze(2), Ra).squeeze(-1).squeeze(-1)

    # denom shape: (1, F) → (1, F, 1, 1) for broadcasting
    denom_exp = denom.unsqueeze(-1).unsqueeze(-1)  # (1, 257, 1, 1)

    W_t = (Ra / (denom_exp + 1e-8)).squeeze(-1)  # Result shape: (1, 257, 8)

    # Store in time-varying weight tensor
    W_mvdr_time[:, :, :, t] = W_t

# -------------------------
# Apply MVDR Beamforming
# -------------------------
# Create zero padding with the same shape except time = Ln
zeros_pad = torch.zeros((B, F, M, Ln), dtype=torch.cfloat, device=W_mvdr_time.device)

# Concatenate zeros at the start on the time dimension
W_mvdr_padded = torch.cat((zeros_pad, W_mvdr_time), dim=3)  # shape now (B, F, M, 497)

# Now apply beamforming
Y_bf_tracking = torch.sum(W_mvdr_padded.conj() * Y_stft, dim=2)  # (B, F, T)
# -------------------------
# ISTFT (Postprocessing)
# -------------------------
speech_hat_tracking = Postprocessing(Y_bf_tracking, R, win_length, device)  # (B, time)

# Example save
save_vector_as_wav(speech_hat_tracking, output_wav_path='mvdr_tracking_output.wav', sample_rate=fs)
