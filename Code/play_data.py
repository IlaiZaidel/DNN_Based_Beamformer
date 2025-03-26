import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/TEST_STFT_domain_results_01_12_2024__19_52_52_0.mat'
mat_data = scipy.io.loadmat(file_path)

# Check keys in the .mat file to find STFT and metadata
print(mat_data.keys())

# Replace 'stft_matrix' and 'fs' with the correct variable names from your .mat file 
# (['__header__', '__version__', '__globals__', 'Y_STFT', 'X_STFT', 'W_STFT_timeChange',
#  'W_STFT_timeFixed', 'W_Stage2', 'X_hat_Stage1', 'X_hat_Stage2', 'skip_Stage1', 'skip_Stage2']
stft_matrix = mat_data['X_STFT']  # Example variable name
fs = mat_data.get('fs', 16000)  # Sampling rate (default to 16 kHz if not provided)

# Compute the magnitude spectrogram
magnitude_spectrogram = np.abs(stft_matrix)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(
    20 * np.log10(magnitude_spectrogram + 1e-10),  # Log scale for better visualization
    aspect='auto',
    origin='lower',
    extent=[0, magnitude_spectrogram.shape[1], 0, fs / 2]
)
plt.colorbar(label='Magnitude (dB)')
plt.xlabel('Time (frames)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.show()

# import scipy.io

# file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/TEST_STFT_domain_results_01_12_2024__19_52_52_0.mat'
# mat_data = scipy.io.loadmat(file_path)

# # Print keys to understand the file structure
# print(mat_data.keys())