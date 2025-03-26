import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import hydra
from hydra.core.config_store import ConfigStore
import sys
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary modules
from utils import Preprocesing, Postprocessing, return_as_complex
from LoadPreTrainedModel import loadPreTrainedModel
from ComputeLoss import Loss
from config import CUNETConfig

# Hydra config setup
cs = ConfigStore.instance()
cs.store(name="cunet_config", node=CUNETConfig)

# Constants
DEVICE_ID = 1  # Set this based on cfg.device.device_num
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
WIN_LEN = 512
HOP_LEN = WIN_LEN // 4
SAMPLE_RATE = 16000
MIC_REF = 4  # Reference microphone

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: CUNETConfig): 
    """
    Runs beamforming model on a given feature file and generates the beampattern.
    """
    # Load the trained model
    model = loadPreTrainedModel(cfg)
    model.to(DEVICE)
    model.eval()

    # Path to a single test feature file
    feature_path = "/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Test/my_feature_vector_76.mat"

    # Process feature and generate beampattern
    process_feature_and_generate_beampattern(feature_path, model, cfg)

def process_feature_and_generate_beampattern(feature_path, model, cfg):
    """
    Given a feature file path, process it through the model, extract W_STFT_timeFixed, 
    compute beamforming, and generate beampattern plots.
    """
    print(f"Processing feature: {feature_path}")

    # Load the feature file
    data = scipy.io.loadmat(feature_path)
    feature = torch.from_numpy(data['feature']).float().to(DEVICE).unsqueeze(0)  # Shape: (1, T, M)

    # Perform STFT
    Y = Preprocesing(feature, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)
    Y = return_as_complex(Y)

    # Forward pass through the model
    with torch.no_grad():
        W_timeChange, X_hat_Stage1, Y_out, W_STFT_timeFixed, X_hat_Stage2, W_Stage2, skip_Stage1, skip_Stage2 = model(Y, DEVICE)

    # Compute and plot beampattern using W_STFT_timeFixed
    amplitudes_sq, amplitudes_spec = compute_beampattern_amplitudes(W_STFT_timeFixed)
    plot_beampattern(amplitudes_sq)
    plot_spectrogram(amplitudes_spec)

    print("Beampattern computation complete.")

def compute_beampattern_amplitudes(W_STFT_timeFixed):
    """
    Given W_STFT_timeFixed, compute the power beampattern and the energy spectrogram per DOA angle.
    """
    amplitudes_sq = torch.zeros(1, 181, device=DEVICE)
    base_path = "/dsi/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround/"
    amplitudes_spec = torch.zeros(181, 257, device=DEVICE)

    # Loop over angles 0 to 180 degrees
    for i in range(181):
        # Load surround feature for the current angle
        surround_file = os.path.join(base_path, f"my_surround_feature_vector_angle_{i}.mat")
        data = scipy.io.loadmat(surround_file)
        feature_vector = torch.from_numpy(data['feature']).float().to(DEVICE).unsqueeze(0)

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
        Z_energy = torch.sum(torch.abs(Z_STFT)**2, dim=1).squeeze(0)
        amplitudes_spec[i, :] = Z_energy

    # Convert to dB scale
    maximum = torch.max(amplitudes_sq)
    amplitudes_sq = 10 * torch.log10(amplitudes_sq / maximum)
    amplitudes_spec = 10 * torch.log10(amplitudes_spec)
    return amplitudes_sq, amplitudes_spec

def plot_beampattern(amplitudes_sq):
    """
    Plots the polar beam pattern given power (in dB) versus angle.
    """
    angles = np.arange(181)
    angles_rad = np.deg2rad(angles)
    mapped_angles = np.pi / 2 - angles_rad
    amplitudes_np = amplitudes_sq.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    ax.plot(mapped_angles, amplitudes_np, label="Power (dB)")

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1])
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc='upper right')
    plt.title("Beamforming Beampattern")
    plt.show()

def plot_spectrogram(amplitudes_spec):
    """
    Plots the spectrogram (frequency vs DOA) given energy (in dB).
    """
    angles = np.arange(181)
    frequencies = np.linspace(0, SAMPLE_RATE/2, amplitudes_spec.shape[1])
    amplitudes_np = amplitudes_spec.cpu().numpy().T

    plt.figure(figsize=(15, 10))
    plt.imshow(amplitudes_np, aspect='auto', extent=[angles.min(), angles.max(),
                frequencies.min(), frequencies.max()], origin='lower', cmap='viridis')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('DOA (degrees)')
    plt.ylabel('Frequency (Hz)')
    plt.title("Spectrogram: Frequency vs DOA")
    plt.show()

if __name__ == "__main__":
    main()
