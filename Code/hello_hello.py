import scipy.io
import soundfile as sf
import numpy as np
# Load the .mat file using the correct path
#mat_file_path = "/dsi/gannot-lab1/datasets/Ilai_data/Two_Directional_Noises_Train" # 'feature', 'fulloriginal', 'fullnoise', 'target_s'
#mat_file_path ="/home/dsi/43rrrilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_03_2025/TEST_STFT_domain_results_25_03_2025__12_30_27_0.mat"
#mat_file_path ="/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_03_2025/TEST_time_domain_results_25_03_2025__12_30_27_0.mat"
mat_file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/23_12_Augmented_Langrangian_first_try/TEST_time_domain_results_23_12_2025__11_15_34_0.mat"

mat_data = scipy.io.loadmat(mat_file_path)
print("Keys in mat_data:", mat_data.keys())

# Normalize each example to have max absolute value of 1
def normalize(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    else:
        return signal  # Return unmodified if max is zero (e.g., silence)

# Extract the desired signal (e.g., 'y')
audio_signals = mat_data['x_hat_stage1_left'] #  'y', 'x_hat_stage1_left', 'x_hat_stage2'
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

# for i in range(num_examples):
#     audio_example = audio_signals[i, :] # Extract the i-th example   audio_signals[i, :]
#     #audio_example = normalize(audio_example)
#     output_file = f"13_08_x_hat_stage1_{i}.wav"  # Name each output file uniquely #31_01_TEMP_x_hat_stage1_audio_{i+1}.wav
#     sf.write(output_file, audio_example, sample_rate)  # Save as WAV file
#     print(f"Saved audio example {i} to {output_file}")
for i in range(num_examples):
    sig = audio_signals[i]

    # Case: multichannel → stereo from first & last mic
    if sig.ndim == 2 and sig.shape[1] >= 2:
        stereo_sig = np.stack([sig[:, 0], sig[:, -1]], axis=-1)  # shape (T, 2)
        output_file = f"23_12_Augmented_Langrangian_first_try_y_{i}.wav"
        sf.write(output_file, stereo_sig, sample_rate)
        print(f"Saved stereo audio example {i} to {output_file}")

    # Case: mono
    else:
        output_file = f"23_12_Augmented_Langrangian_first_try_x_hat_stage1_{i}.wav"
        sf.write(output_file, sig, sample_rate)
        print(f"Saved mono audio example {i} to {output_file}")


print("All audio examples have been saved.")

