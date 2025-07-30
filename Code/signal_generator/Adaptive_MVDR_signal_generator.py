import os
import sys

# Add parent directory to the Python path so we can import from Code/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Standard Library ===
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import scipy.signal as signal
import rir_generator
# === Project Imports from ../ ===
from signal_generator import SignalGenerator
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening, noise_estimation
from utils import beamformingOpreation, Postprocessing, Preprocesing, return_as_complex, save_vector_as_wav, place_source_2d
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking

# --- Parameters ---
c = 340.0
L = [4.0, 5.0, 6.0]
beta = [0.2]
nsamples = 1024
order = 2
hop = 32
fs = 16000
win_length = 512
R = win_length // 4
noise_only_time = 0.5
T_sec = 8.0
T = int(fs * T_sec)
x1, _ = sf.read("/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28241/5105-28241-0011.wav")
# --- Load source signal ---
in_signal, fs = sf.read('/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator/female_speech.wav')
in_signal=x1
if in_signal.ndim > 1:
    in_signal = in_signal[:, 0]
in_signal = in_signal[:T]

# --- Receiver setup ---
rp = np.array([
    [1.5, 2.4, 3.0],
    [1.5, 2.6, 3.0],
    [1.5, 2.8, 3.0],
    [1.5, 3.0, 3.0],
    [1.5, 3.2, 3.0],
    [1.5, 3.4, 3.0],
    [1.5, 3.6, 3.0],
    [1.5, 3.8, 3.0]
])
M = rp.shape[0]
cp = np.mean(rp, axis=0)

# --- Source motion ---
start = np.array([2.0, 1.0, 3.0])
stop = np.array([2.0, 3.5, 3.0])
sp_path = np.zeros((T, 3))
rp_path = np.zeros((T, 3, M))

for i in range(0, T, hop):
    alpha = i / T
    sp = start + alpha * (stop - start)
    sp_path[i:i + hop] = sp
    for m in range(M):
        rp_path[i:i + hop, :, m] = rp[m]

# --- Generate reverberant signal ---
gen = SignalGenerator()
result = gen.generate(
    input_signal=list(in_signal),
    c=c,
    fs=fs,
    r_path=rp_path.tolist(),
    s_path=sp_path.tolist(),
    L=L,
    beta_or_tr=beta,
    nsamples=nsamples,
    mtype="o",
    order=order,
    hp_filter=True
)

output = np.array(result.output).T  # (T, M)
output = output[:T, :]



# --- Simulate noise ---
az_noise = -90   # degrees
distance = 1.0  # meters
mic_center = np.mean(rp, axis=0)
noise_position  = place_source_2d(az_noise, distance, mic_center)
beta = np.array([0.5] * 6) #0.5
rir_len = 4800
h_noise = np.array(rir_generator.generate(c, fs, rp, noise_position, L, beta, rir_len, order=0)).T #(8, 2048)

print("flag before w")
w = np.random.randn(int(T_sec * fs), 2)
print("flag after w")
AR = [1, -0.7]  # AR filter coefficients
noise_sources = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

n = np.stack([
    signal.lfilter(h_noise[m, :], [1], noise_sources[:, 0])
    for m in range(M)
], axis=1)  # Shape: (T*fs, M)



# --- Scale noise to desired SNR ---
SNR_noise = -1
mic_ref = 3
power_speech = np.sum(output[:, mic_ref] ** 2)
power_noise = np.sum(n[:, mic_ref] ** 2) + 1e-10
dVSn = np.sqrt(power_speech / (power_noise * 10 ** (SNR_noise / 10)))
n *= dVSn

# --- Create mixture ---
y = output + n 
y = y / (np.max(np.abs(y)) + 1e-9)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# --- STFT ---
Y = Preprocesing(y_tensor, win_length, fs, T_sec, R, y_tensor.device)
Y_stft = return_as_complex(Y)

# --- Subspace Tracking ---
Ln = int(noise_only_time * fs // R)
W, eigvals, eigvecs = pastd_rank1_whitened(Y_stft, noise_only_time, beta=0.995)
a_hat = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time)

# --- Covariance Whitening ---
rtf_cw = covariance_whitening(Y_stft, noise_only_time)
rtf_cw = fix_covariance_whitening(rtf_cw)

# --- Noise Covariance ---
Rnn = noise_estimation(Y_stft, Ln)
epsilon = 1e-6
eye = torch.eye(Rnn.size(-1), device=Rnn.device).unsqueeze(0)
Rnn_reg = Rnn + epsilon * eye
Rnn_inv = torch.linalg.inv(Rnn_reg)




# --- MVDR Weights (time-independent RTF version) ---
a = rtf_cw.squeeze(-1).permute(0, 2, 1)  # from [1, 8, 257, 1] → [1, 257, 8]
a_H = a.conj().unsqueeze(-2)         # [1, 257, 1, 8]
a_unsq = a.unsqueeze(-1)             # [1, 257, 8, 1]

Ra = torch.matmul(Rnn_inv, a_unsq)   # [1, 257, 8, 1]
aRa = torch.matmul(a_H, Ra)          # [1, 257, 1, 1]
aRa = aRa.squeeze(-1).squeeze(-1)    # [1, 257]

W_mvdr = Ra.squeeze(-1) / (aRa.unsqueeze(-1) + 1e-8)  # [1, 257, 8]

# --- Beamforming ---
W_H = W_mvdr.conj().unsqueeze(-1)    # [1, 257, 8, 1]
Y_stft_perm = Y_stft.permute(0, 2, 1, 3)  # [1, 257, 8, T]
Y_bf = torch.sum(W_H * Y_stft_perm, dim=2)  # [1, 257, T]






# --- ISTFT ---
speech_hat = Postprocessing(Y_bf, R, win_length, y_tensor.device)
save_vector_as_wav(speech_hat, output_wav_path='mvdr_output.wav', sample_rate=fs)

# --- MVDR with time-varying RTF (PASTd) ---
B, F, M, T_spec = a_hat.shape
Y_stft = Y_stft.permute(0, 2, 1, 3).contiguous()
W_mvdr_time = torch.zeros((B, F, M, T_spec), dtype=torch.cfloat, device=Y_stft.device)

for t in range(T_spec):
    a_t = a_hat[:, :, :, t].contiguous()                      # [1, 257, 8]
    a_t_unsq = a_t.unsqueeze(-1)                              # [1, 257, 8, 1]
    a_t_H = a_t_unsq.conj().transpose(-1, -2)                 # [1, 257, 1, 8]
    
    Ra = torch.matmul(Rnn_inv, a_t_unsq)                      # [1, 257, 8, 1]
    denom = torch.matmul(a_t_H, Ra).squeeze(-1).squeeze(-1)   # [1, 257]
    
    W_t = (Ra / (denom.unsqueeze(-1).unsqueeze(-1) + 1e-8)).squeeze(-1)  # [1, 257, 8]
    W_mvdr_time[:, :, :, t] = W_t

zeros_pad = torch.zeros((B, F, M, Ln), dtype=torch.cfloat, device=W_mvdr_time.device)
W_mvdr_padded = torch.cat((zeros_pad, W_mvdr_time), dim=3)
Y_bf_tracking = torch.sum(W_mvdr_padded.conj() * Y_stft, dim=2)
speech_hat_tracking = Postprocessing(Y_bf_tracking, R, win_length, y_tensor.device)

# Save MVDR beamformed signal (single-channel)
sf.write('mvdr_output.wav', speech_hat.squeeze().cpu().numpy(), fs, subtype='FLOAT')

# Save MVDR tracking-based beamformed signal (single-channel)
sf.write('mvdr_tracking_output.wav', speech_hat_tracking.squeeze().cpu().numpy(), fs, subtype='FLOAT')

# Save original noisy signal (multi-channel → stereo if M >= 2)
y_np = y[:, :2] if y.shape[1] >= 2 else y  # stereo if possible
sf.write('original_noisy_moving_output.wav', y_np, fs, subtype='FLOAT')

# Save original clean signal (multi-channel → stereo if M >= 2)
output_np = output[:, :2] if output.shape[1] >= 2 else output
sf.write('original_clean_moving_output.wav', output_np, fs, subtype='FLOAT')