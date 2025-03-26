import scipy.io
import soundfile as sf
import numpy as np

# Load the .mat file using the correct path
#mat_file_path = "/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Train" # 'feature', 'fulloriginal', 'fullnoise', 'target_s'
#mat_file_path ="/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_03_2025/TEST_STFT_domain_results_25_03_2025__12_30_27_0.mat"
#mat_file_path ="/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_03_2025/TEST_time_domain_results_25_03_2025__12_30_27_0.mat"
mat_file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/ADI_2_noises_25_03/TEST_time_domain_results_25_03_2025__21_21_24_0.mat"
mat_data = scipy.io.loadmat(mat_file_path)
print("Keys in mat_data:", mat_data.keys())

# Extract the desired signal (e.g., 'y')
audio_signals = mat_data['x_hat_stage1'] #  'y', 'x_hat_stage1', 'x_hat_stage2'
print(f"Audio signals shape: {audio_signals.shape}")
print(f"Audio signals dtype: {audio_signals.dtype}")

# If the audio is complex, process it
if np.iscomplexobj(audio_signals):
    print("Audio signals are complex, taking real part.")
    audio_signals = np.real(audio_signals)

# Convert to float32 for writing to file
audio_signals = audio_signals.astype(np.float32)

# Save each audio example separately
num_examples = audio_signals.shape[0]#0  # Number of examples
sample_rate = 16000  # Define the sample rate

for i in range(num_examples):
    audio_example = audio_signals[i, :] # Extract the i-th example   audio_signals[i, :]
    output_file = f"Adi_25_03_x_hat_stage1_{i}.wav"  # Name each output file uniquely #31_01_TEMP_x_hat_stage1_audio_{i+1}.wav
    sf.write(output_file, audio_example, sample_rate)  # Save as WAV file
    print(f"Saved audio example {i} to {output_file}")

print("All audio examples have been saved.")


