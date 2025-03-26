import os
import sys
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary functions
from utils import Preprocesing, return_as_complex, Postprocessing

# Constants
DEVICE_ID = 2  # Set based on your system
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
WIN_LEN = 512
HOP_LEN = WIN_LEN // 4
SAMPLE_RATE = 16000
MIC_REF = 4  # Reference microphone

# Set the feature index (change this to process a different sample)
INDEX = 9  # Modify this value to select a different test sample

# Paths to required files
STFT_MAT_FILE = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_03_2025/TEST_STFT_domain_results_25_03_2025__12_30_27_0.mat"
FEATURE_MAT_FILE = f"/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Test/my_feature_vector_{INDEX}.mat"
OUTPUT_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/plots"
NAME = f"ILAI"
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_beampattern_amplitudes(W_STFT_timeFixed):
    """
    Compute the beampattern using the beamforming filter W_STFT_timeFixed.
    """
    amplitudes_sq = torch.zeros(1, 181, device=DEVICE)
    base_path = "/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround/"
    amplitudes_spec = torch.zeros(181, 257, device=DEVICE)

    for i in range(181):
        # Load surround feature for the current angle
        data = scipy.io.loadmat(os.path.join(base_path, f"my_surround_feature_vector_angle_{i}.mat"))
        feature_vector = torch.from_numpy(data["feature"]).float().to(DEVICE).unsqueeze(0)

        # Perform STFT
        Y = Preprocesing(feature_vector, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)
        Y = return_as_complex(Y)

        # Apply beamforming
        wy = torch.mul(torch.conj(W_STFT_timeFixed), Y)
        Z_STFT = torch.sum(wy, dim=1).squeeze(-1)

        # Convert back to the time domain
        z = Postprocessing(Z_STFT, HOP_LEN, WIN_LEN, DEVICE)
        norm_z = torch.sum(torch.abs(z) ** 2)
        amplitudes_sq[0, i] = norm_z

        # Compute frequency-wise energy
        Z_STFT = Z_STFT.squeeze(0)
        Z_energy = torch.sum(torch.abs(Z_STFT) ** 2, dim=1).squeeze(0)
        amplitudes_spec[i, :] = Z_energy

    # Convert to dB scale
    maximum = torch.max(amplitudes_sq)
    amplitudes_sq = 10 * torch.log10(amplitudes_sq / maximum)
    amplitudes_spec = 10 * torch.log10(amplitudes_spec)
    return amplitudes_sq, amplitudes_spec

def plot_beampattern(amplitudes_sq, angle_x, angle_n1, angle_n2, index):
    """
    Plots the polar beampattern with the speaker and two noise angles and saves the figure.
    """
    angles = torch.arange(181)
    amplitudes_np = amplitudes_sq.squeeze(0).cpu().numpy()
    angles_rad = np.deg2rad(angles.numpy())
    mapped_angles = np.pi / 2 - angles_rad

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 6))
    ax.plot(mapped_angles, amplitudes_np, label="Power (dB)")

    # Plot source and noise angles
    ax.plot([np.pi / 2 - np.deg2rad(angle_x)], [0], 'ro', label='Speaker')
    ax.plot([np.pi / 2 - np.deg2rad(angle_n1)], [0], 'bo', label='Noise 1')
    ax.plot([np.pi / 2 - np.deg2rad(angle_n2)], [0], 'go', label='Noise 2')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc='upper right')

    plt.title(f"Beamforming Beampattern (Feature Index {index})")

    # Save the figure instead of showing it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"beampattern_index_{index}_{timestamp}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Beampattern plot saved to {output_path}")

def plot_spectrogram(amplitudes_spec, index):
    """
    Plots the spectrogram (frequency vs DOA) and saves the figure.
    """
    angles = np.arange(181)
    frequencies = np.linspace(0, SAMPLE_RATE / 2, amplitudes_spec.shape[1])
    amplitudes_np = amplitudes_spec.cpu().numpy().T

    plt.figure(figsize=(15, 10))
    plt.imshow(amplitudes_np, aspect="auto", extent=[angles.min(), angles.max(),
                frequencies.min(), frequencies.max()], origin="lower", cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.xlabel("DOA (degrees)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram: Frequency vs DOA (Feature Index {index})")

    # Save the figure instead of showing it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"spectrogram_index_{index}_{timestamp}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Spectrogram plot saved to {output_path}")

if __name__ == "__main__":
    # Load STFT Mat file
    data_stft = scipy.io.loadmat(STFT_MAT_FILE)
    W_STFT_timeFixed = torch.tensor(data_stft["W_STFT_timeFixed"][INDEX]).unsqueeze(0).to(DEVICE)

    # Load Feature Mat file
    data_feature = scipy.io.loadmat(FEATURE_MAT_FILE)
    angle_x = float(data_feature["angle_x"])
    angle_n1 = float(data_feature["angle_n1"])
    angle_n2 = float(data_feature["angle_n2"])

    # Compute and plot beampattern
    amplitudes_sq, amplitudes_spec = compute_beampattern_amplitudes(W_STFT_timeFixed)
    plot_beampattern(amplitudes_sq, angle_x, angle_n1, angle_n2, INDEX)
    plot_spectrogram(amplitudes_spec, INDEX)

    print("Beampattern computation complete.")
