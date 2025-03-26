import os
import sys
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from RTF_covariance_whitening import (
    covariance_whitening, fix_covariance_whitening
)
from utils import Preprocesing, return_as_complex, Postprocessing


# Constants
DEVICE_ID = 2
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
WIN_LEN = 512
HOP_LEN = WIN_LEN // 4
SAMPLE_RATE = 16000
MIC_REF = 4

# Compute beampattern amplitudes_sq
def compute_beampattern_amplitudes(W_Stage1):
    amplitudes_sq = torch.zeros(1, 181)
    base_path = '/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround/'
    amplitudes_spec = torch.zeros(181, 257)
    for i in range(181):
        data = scipy.io.loadmat(os.path.join(base_path, f"my_surround_feature_vector_angle_{i}.mat"))
        feature_vector = torch.from_numpy(data['feature']).float().to(DEVICE).unsqueeze(0)

        Y = Preprocesing(feature_vector, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)
        Y = return_as_complex(Y)

        wy = torch.mul(torch.conj(W_Stage1),Y) #torch.Size([1, 8, 257, 497])

        Z_STFT = torch.sum(wy, dim=1).squeeze(-1) # torch.Size([1, 257, 497])
        
        z = Postprocessing(Z_STFT, HOP_LEN, WIN_LEN, DEVICE) # torch.Size([1, 64000])       
        norm_z = torch.sum(torch.abs(z) ** 2)
        amplitudes_sq[0,i] = norm_z

        Z_STFT = Z_STFT.squeeze(0)
        Z_energy = torch.sum(torch.abs(Z_STFT)**2, dim=1).squeeze(0) #torch.Size([257])
        amplitudes_spec[i,:] = Z_energy



    #maximum = amplitudes_sq[0,100]
    maximum = torch.max(amplitudes_sq)
    amplitudes_sq = 10*torch.log10((amplitudes_sq/maximum))
    amplitudes_spec = 10*torch.log10(amplitudes_spec)
    return amplitudes_sq, amplitudes_spec

# Plot polar beampattern
def plot_beampattern(amplitudes_sq):
    angles = torch.arange(181)
    amplitudes_np = amplitudes_sq.squeeze(0).numpy()
    angles_rad = np.deg2rad(angles.numpy())
    mapped_angles = np.pi / 2 - angles_rad

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    ax.plot(mapped_angles, amplitudes_np, label="Power (dB)")

    source_angle, noise_angle = 100, 120
    ax.plot([np.pi / 2 - np.deg2rad(source_angle)], [0], 'ro', label='Source')
    soft_yellow = '#FFD700'  # Softer yellow
    ax.plot([np.pi / 2 - np.deg2rad(noise_angle)], [0], color=soft_yellow, marker='o', linestyle='', label='Noise')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    ax.tick_params(axis='y', labelsize=5)
    ax.legend(loc='upper right')

    output_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/Beampatterns_and_Frequency_vs_DOA/white_amplitude_vs_angle_polar.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

# Plot spectrogram
def plot_spectrogram(amplitudes_spec):
    angles = torch.arange(181)  # Angles from 0 to 180 degrees
    frequencies = np.linspace(0, SAMPLE_RATE / 2, amplitudes_spec.shape[1])  # Frequency range up to Nyquist frequency

    # Compute the magnitude of the complex amplitudes_sq
    amplitudes_magnitude = amplitudes_spec.float().numpy() 
    

    # Create the spectrogram plot
    plt.figure(figsize=(15, 10))
    plt.imshow(
        amplitudes_magnitude.T,  # Transpose to align frequency on the y-axis
        aspect='auto',
        extent=[angles.min(), angles.max(), frequencies.min(), frequencies.max()],
        origin='lower',
        cmap='viridis',  # Updated colormap to highlight high amplitudes_sq in yellow

    )
    plt.colorbar(label='Power (dB)')
    plt.xlabel('DOA (angles in degrees)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram: Frequency vs DOA (angles)')
    output_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/Beampatterns_and_Frequency_vs_DOA/spectogram_amplitude_vs_angle_polar.png"
    plt.savefig(output_path, dpi=300)
    plt.show()


# Main execution
if __name__ == "__main__":
    #mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/14_01_2025/TEST_STFT_domain_results_14_01_2025__11_00_56_0.mat' # First
    #mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/Adi_Original_First/TEST_STFT_domain_results_09_01_2025__09_51_44_0.mat'
    #mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/04_01_2025/TEST_STFT_domain_results_07_01_2025__21_34_07_0.mat'
    #mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/Adi_Original/TEST_STFT_domain_results_28_12_2024__10_21_13_0.mat'
    mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/12_03_2025/TEST_STFT_domain_results_12_03_2025__07_38_19_1.mat'# Second
    data = scipy.io.loadmat(mat_file_path)

    W_Stage1 = torch.tensor(data['W_STFT_timeFixed'][1]).unsqueeze(0).to(DEVICE)
    Y = torch.tensor(data['Y_STFT'][1]).unsqueeze(0).to(DEVICE)

    amplitudes_sq, amplitudes_spec = compute_beampattern_amplitudes(W_Stage1)

    print(amplitudes_sq)
    plot_beampattern(amplitudes_sq)
    plot_spectrogram(amplitudes_spec)



    print('done')

