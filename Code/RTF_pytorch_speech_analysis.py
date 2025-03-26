import torch
import numpy as np
from RTF_covariance_whitening import RTF_Cov_W, noise_estimation, mix_estimation, covariance_whitening, fix_covariance_whitening
import scipy.io
import matplotlib.pyplot as plt

import soundfile as sf
from utils import Preprocesing, Postprocessing, beamformingOpreation
#from covariance_whitening_backup import covariance_whitening, fix_covariance_whitening
device_ids = 2
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
device = "cpu"
file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Dataset/Non-Reverberant Environment/Standard/feature_vector_1.mat"
#'/home/dsi/ilaiz/DNN_Based_Beamformer/Code/my_feature_vector_0.mat'
# Load the .mat file
data = scipy.io.loadmat(file_path)

feature_vector = data['feature']
noise = data['fullnoise']
only_noise = data['feature'][:,3] - data['fulloriginal'][:,3]
white_noise_only = data['feature'][:,3] - data['fulloriginal'][:,3] - data['fullnoise'][:,3]
# Normalize to [-1, 1] (optional)
only_noise = only_noise / np.max(np.abs(only_noise))
white_noise_only = white_noise_only#/ np.max(np.abs(white_noise_only))
# Define sample rate
sample_rate = 16000  # Set the sample rate (in Hz)

# Save as .wav file
output_file = "only_noise.wav"
sf.write(output_file, only_noise, sample_rate)
print(f"Saved 'only_noise' audio to {output_file}.")
# Save as .wav file
output_file = "only_white_noise.wav"
sf.write(output_file, white_noise_only, sample_rate)
print(f"Saved 'only_white_noise' audio to {output_file}.")
# Convert the NumPy array to a PyTorch tensor and move it to the specified device
feature_vector_tensor = torch.from_numpy(feature_vector).float().to(device)
feature_vector_tensor = feature_vector_tensor.unsqueeze(0)  # Add batch dimension
print(feature_vector_tensor.shape)

win_len = 512            # Window length for STFT
R = win_len // 4         # Hop length
T = 4                    # Length of the signal in time domain
M = 8                    # Number of microphones
fs = 16000               # Sample rate
mic_ref = 4

# Preprocessing: Ensure Y is on the same device as the input
Y = Preprocesing(feature_vector_tensor, win_len, fs, T, R, device)
print(Y.shape)

B, M, F, L = Y.size()
Y_stft = Y.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()  # Reshape to separate real and imaginary parts
print(Y_stft.shape)
Y_stft = torch.view_as_complex(Y_stft)  # Convert to complex tensor
print(Y.shape)

y = Y_stft[0]
print(y.shape)

F = F // 2

# Covariance whitening: Ensure the input tensor is on the correct device
a_cw = covariance_whitening(Y_stft)  # Output: B, M, F, 1 (F = F//2)

G = fix_covariance_whitening(a_cw)



print("a shape is " , a_cw.shape)
# Create weights on the correct device
W = torch.rand((1, M, F, 1), dtype=torch.cfloat, device=device)
print("W shape is " , W.shape)
# Ensure Y is on the correct device#WX, _, _ = beamformingOpreation(Y.to(device), mic_ref, W)
wa = torch.mul(torch.conj(W), a_cw)
ones = torch.ones(F, L, device=device)  # Ensure this is also on the same device
cost_wa = torch.mean(torch.abs(wa - ones)) * 10000
print(cost_wa)


import os
import matplotlib.pyplot as plt
import torch

# Assuming `a_cw` is already computed and has shape [B, M, F, L]
B, M, F, L = a_cw.shape

# Directory to save plots
output_dir = "/home/dsi/ilaiz/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Ensure `a_cw` is detached from the computation graph and moved to CPU for processing
a_cw_cpu = a_cw.squeeze(-1).permute(1, 0, 2).contiguous().detach().cpu()  # [M, B, F]

# Compute the IFFT for each microphone (across the frequency axis)
ifft_results = torch.fft.ifftshift(torch.fft.ifft(a_cw_cpu, dim=-1), dim=-1)  # Shape: [M, B, F]

# Plot and save the IFFT for each microphone
for mic_idx in range(M):
    plt.figure(figsize=(8, 4))
    plt.plot(ifft_results[mic_idx, 0, :].real.numpy())  # Plot the real part of the IFFT
    plt.title(f"RTF - Microphone {mic_idx + 1}")
    plt.xlabel("Frequency Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plot_path = os.path.join(output_dir, f"RTF_Microphone_{mic_idx + 1}.png")
    plt.savefig(plot_path)  # Save the plot as a PNG file
    plt.close()  # Close the figure to save memory

print(f"Plots saved in {output_dir}")


G = fix_covariance_whitening(Y_stft)