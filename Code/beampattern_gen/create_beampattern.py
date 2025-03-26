import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from RTF_covariance_whitening import RTF_Cov_W, noise_estimation, mix_estimation, covariance_whitening, fix_covariance_whitening
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from generate_dataset import GeneretedInputOutput
import hydra
from datetime import datetime
from config import CUNETConfig 
from hydra.core.config_store import ConfigStore
from ExNetBFPFModel import ExNetBFPF
import scipy.io as sio
import os

import numpy as np
from scipy.io.wavfile import write
# path_folder = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/my_second_reference_feature_vector.mat'
# data = scipy.io.loadmat(path_folder)
# mic_4_data = data['feature'][:, 3]  
# # Normalize the mic_4_data to the range [-1, 1]
# mic_4_data = mic_4_data / np.max(np.abs(mic_4_data))

# # Convert to 16-bit PCM format for WAV file
# mic_4_data_int16 = (mic_4_data * 32767).astype(np.int16)

# # Set the file path where you want to save the WAV file
# output_path = 'second_reference_feature.wav'

# #Save the data to a WAV file with the chosen sample rate (16000 Hz)
# write(output_path, 16000, mic_4_data_int16)

# print(f"WAV file saved as {output_path}")



# Now you can import the functions from utils.py
from utils import Preprocesing, Postprocessing, beamformingOpreation

device_ids = 2
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
device = "cpu"

#path_folder = '/dsi/gannot-lab1/datasets/Ilai_data/Beampattern_Surround/my_surround_feature_vector_angle_148.mat'

path_folder = 'DNN_Based_Beamformer/Code/beampattern_gen/reference_feature_vector_0.mat'
# Load the .mat file
data = scipy.io.loadmat(path_folder)
# Display the keys (top-level variables) in the .mat file
print("Keys in the .mat file:")
print(data.keys())
#dict_keys(['__header__', '__version__', '__globals__', 'beta', 'x_position', 'n_position', 'mic_position',
#  'room_dim', 'angleOrientation', 'angle_x', 'angle_n', 'radius', 'feature', 'fulloriginal', 'fullnoise', 'target_s'])
feature_vector = data['feature']
print("Shape of feature_vector:", feature_vector.shape)
print("Shape of feature_angle:", data['angle_x'])

#path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/TEST_STFT_domain_results_30_12_2024__12_26_44_0.mat' # 27_12_2024 Model
#path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/TEST_STFT_domain_results_30_12_2024__14_00_17_1.mat' # Adi Model
#path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/30_12_2024/TEST_STFT_domain_results_31_12_2024__10_28_10_0.mat' #30_12_2024 Model
#path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/06_01_2025/TEST_STFT_domain_results_06_01_2025__15_47_16_2.mat' #reference
#path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/06_01_2025/TEST_STFT_domain_results_06_01_2025__17_12_32_1.mat' # second reference
path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/06_01_2025/TEST_STFT_domain_results_06_01_2025__15_47_16_2.mat'
data = scipy.io.loadmat(path)
# Display the keys (top-level variables) in the .mat file
print("Keys in the .mat file:")
print(data.keys())
#dict_keys(['__header__', '__version__', '__globals__', 'Y_STFT', 'X_STFT', 'W_STFT_timeChange', 'W_STFT_timeFixed', 
# 'W_Stage2', 'X_hat_Stage1', 'X_hat_Stage2', 'skip_Stage1', 'skip_Stage2'])
W_Stage1 = data['W_STFT_timeFixed'][0]
W_Stage2 = data['W_Stage2'][0]

W_Stage1_tensor = torch.tensor(W_Stage1)
W_Stage2_tensor = torch.tensor(W_Stage2)

# Add a batch dimention
W_Stage1 = W_Stage1_tensor.unsqueeze(0) # shape of 1,8,257,1
W_Stage2 = W_Stage2_tensor.unsqueeze(0) # shape of 1,1,514,497

X_hat_Stage2 = data['X_hat_Stage2'][0]


Y =torch.tensor(data['Y_STFT'][0]).unsqueeze(0)  # Shape of (1,8,257,497)
a_cw = covariance_whitening(Y)  # Output: B, M, F, 1 ([1, 8, 257, 1])

G = fix_covariance_whitening(a_cw) # ([1, 8, 257, 1])

#wa = beamformingOpreation(a_cw,mic_ref,W_Stage1)
wa = torch.mul(torch.conj(W_Stage1), G)
wa = torch.sum(wa, dim=1).squeeze(-1) # shape 1,257

reference_wa = torch.sum(torch.abs(wa)**2)

amplitudes = torch.zeros(1,181)

for i in range(181):

    path = '/dsi/gannot-lab1/datasets/Ilai_data/Noisy_Beampattern_Surround/'
    new_path = path + 'my_surround_feature_vector_angle_' + str(i) + '.mat'   # The name of the file
    data = scipy.io.loadmat(new_path)

    feature_vector = data['feature']

    # Convert the NumPy array to a PyTorch tensor and move it to the specified device
    feature_vector_tensor = torch.from_numpy(feature_vector).float().to(device)
    feature_vector_tensor = feature_vector_tensor.unsqueeze(0)  # Add batch dimension
    #print(feature_vector_tensor.shape)

    win_len = 512            # Window length for STFT
    R = win_len // 4         # Hop length
    T = 4                    # Length of the signal in time domain
    M = 8                    # Number of microphones
    fs = 16000               # Sample rate
    mic_ref = 4

    # Preprocessing: Ensure Y is on the same device as the input
    Y = Preprocesing(feature_vector_tensor, win_len, fs, T, R, device) # torch.Size([1, 8, 514, 497])
    

    B, M, F, L = Y.size()
    Y_stft = Y.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()  # Reshape to separate real and imaginary parts
    Y= torch.view_as_complex(Y_stft)  # Convert to complex tensor, torch.Size([1, 8, 257, 497])
    a_cw = covariance_whitening(Y)  # Output: B, M, F, 1 ([1, 8, 257, 1])
    G = fix_covariance_whitening(a_cw) # ([1, 8, 257, 1])

    wa = torch.mul(torch.conj(W_Stage1), G)
    wa = torch.sum(wa, dim=1).squeeze(-1) # shape 1,257

    surround_wa = torch.sum(torch.abs(wa)**2)/reference_wa
    amplitudes[0,i] = 10*torch.log(surround_wa)

print(amplitudes)

import matplotlib.pyplot as plt
import torch
import numpy as np


# Assuming `amplitudes` tensor is already computed
angles = torch.arange(181)  # Create angles from 0 to 180

# Convert amplitudes to NumPy for plotting
amplitudes_np = amplitudes.squeeze(0).numpy()  # Remove batch dimension and convert to NumPy

# Convert angles to radians and map them to the top half of the sphere
angles_rad = np.deg2rad(angles.numpy())
mapped_angles = np.pi / 2 - angles_rad  # Map 0° to 90° and 180° to -90°

# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
ax.plot(mapped_angles, amplitudes_np, label="Amplitude (dB)")

# Highlight the maximum value
max_index = np.argmax(amplitudes_np)
max_angle_mapped = mapped_angles[max_index]
max_amplitude = amplitudes_np[max_index]

# Add Source at 100 degrees
source_angle = 100  # degrees
source_amplitude = amplitudes_np[source_angle]  # Amplitude at 100 degrees
source_angle_mapped = np.pi / 2 - np.deg2rad(source_angle)
ax.plot([source_angle_mapped], [0], 'ro', label='Source')  # Red dot for source

# Add Noise at 120 degrees
noise_angle = 120  # degrees
noise_amplitude = amplitudes_np[noise_angle]  # Amplitude at 120 degrees
noise_angle_mapped = np.pi / 2 - np.deg2rad(noise_angle)
ax.plot([noise_angle_mapped], [0], 'yo', label='Noise')  # Yellow dot for noise

# Set up labels and appearance
ax.set_theta_zero_location('N')  # 0 degrees points up
ax.set_theta_direction(-1)  # Clockwise direction
ax.set_thetamin(-90)  # Show only the top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1])  # Set radial axis limits

# Reduce the font size of the radial ticks (dB axis)
ax.tick_params(axis='y', labelsize=5)

# Show legend for Source and Noise
ax.legend(loc='upper right')

# Save and show the plot
output_plot_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/amplitude_vs_angle_polar.png"
plt.savefig(output_plot_path, dpi=300)
plt.show()
