import torch

def changeTime(signal, length):
    """
    Truncates the signal to a specified length.

    Args:
        signal (torch.Tensor): Input signal.
        length          (int): Desired length of the signal.

    Returns:
        torch.Tensor: Modified signal.
    """
    B, samples, M = signal.shape
    if samples > length:
        signal = signal[:, 0:length, :]
    signal = torch.squeeze(signal, dim=2)
    return signal

def Preprocesing(y, win_len, fs, T, R, device):
    """
    Applies preprocessing (truncates the signal and STFT) to the input signal.

    Args:
        y (torch.Tensor): Input signal in the time domain.
        win_len    (int): Window length for STFT.  
        fs         (int): Sample rate.
        T          (int): Length of the signal in the time domain.
        R          (int): hop_length - the length of the non-intersecting portion of window length.
        device (torch.device): Device to perform the operations on.

    Returns:
        Y     (torch.Tensor): Preprocessed signal in the STFT domain.
    """
    # Truncates the signal to a specified length.
    B, samples, M = y.size()
    if samples > fs * T:
        y = y[:,0:fs*T,:]                                  # y = B,T*fs,M = B,8,64000 
    y = y.permute(0,2,1).contiguous().view(B*M,-1)         # y = B*M,T*fs = B*8,64000 

    # STFT transformation 
    w_analysis = torch.hamming_window(win_len).to(device)
    Y = torch.stft(y, n_fft=win_len, hop_length=int(R), win_length=win_len, window=w_analysis, center=False, return_complex = False)# , return_complex=False)  #ILAI: I added the return_complex=True
    #print("Shape of Y:", Y.size()) 
    B_M, F, L, C = Y.size()      # Y = B*M,F,L,C = B*8,257,497,2   _, F, L = Y.size()

    Y = Y.permute(0,1,3,2).contiguous().view(B,M,F*C,L)    # Y = B,M,F*C,L = B,8,257*2,497
    #print("Shape of Y:", Y.size()) 
    return Y

def Postprocessing(X_hat, R, win_len, device):
    """
    Applies postprocessing (ISTFT) to the estimated source signal.

    Args:
        X_hat   (torch.Tensor): Single channel estimated source signal in the STFT domain.
        R       (int): hop_length - the length of the non-intersecting portion of window length.
        win_len (int): Window length for ISTFT.
        device  (torch.device): Device to perform the operations on.

    Returns:
        x_hat (torch.Tensor): Postprocessed signal in the time domain.
    """
    #X_hat = torch.view_as_real(X_hat) # X_hat = B,F,L,2 = B,257,497,2
  # Convert real tensor with 2 channels to complex tensor
    # Ensure that X_hat is complex (if it's not already complex)
 # Ensure that X_hat is complex (if it's not already complex)
    if X_hat.is_complex():
        X_hat_complex = X_hat
    else:
        # Create complex tensor from real (real part, imaginary part)
        X_hat_complex = torch.complex(X_hat[..., 0], X_hat[..., 1])  # real part and imaginary part

    # ISTFT transformation
    w_analysis = torch.hamming_window(win_len).to(device)
    x_hat = torch.istft(X_hat_complex, n_fft=win_len, hop_length=int(R), win_length=win_len, window=w_analysis, center=False, return_complex=False)

    return x_hat

def beamformingOpreation(Y, mic_ref, W=0):
    """
    Applies beamforming operation to the input signal with the calculated weights.

    Args:
        Y (torch.Tensor): Input signal in the STFT domain.
        mic_ref    (int): Reference microphone index.
        W (torch.Tensor or int, optional): Beamforming weights. Defaults to 0 (uniform weights).

    Returns:
        X_hat (torch.Tensor): Estimated source signal in the STFT domain.
        Y     (torch.Tensor): Input signal in the STFT domain.
        W     (torch.Tensor or int): Updated beamforming weights. 
    """
    B, M, F, L = Y.size()        # Y = B,M,F,L = B,8,514,497
    Y = Y.view(B, M, F // 2, 2, L).permute(0,1,2,4,3).contiguous() #  Y = B,M,F//2,L,2 = B,8,257,497,2
    Y = torch.view_as_complex(Y) # Y = B,M,F//2,L = B,8,257,497

    # If we got no input weights, we will take the y recorded at the reference microphone by 
    # setting W as 1 in the reference channel and 0 for the other channels.
    if type(W) == int:  
        W = torch.zeros_like(Y)
        W[:, mic_ref-1, :, :] = W[:, mic_ref-1, :, :] + 1

    # BeamformingOpreation 
    X_hat = torch.mul(torch.conj(W), Y) # X_hat = B,M,F,L = B,8,257,497
    X_hat = torch.sum(X_hat, dim=1)     # X_hat = B,F,L = B,257,497

    return X_hat, Y, W

import scipy.io
from scipy.io.wavfile import write
import numpy as np

import soundfile as sf
def save_feature_as_wav(mat_file_path, matrix_name, output_wav_path='my_output_file.wav', mic_index=3, sample_rate=16000):
    """
    Load a matrix from a .mat file, extract a feature vector, normalize it, and save it as a WAV file.

    Parameters:
    - mat_file_path (str): Path to the .mat file containing the matrix.
    - matrix_name (str): Name of the matrix to extract from the .mat file (e.g., 'feature', 'fulloriginal').
    - output_wav_path (str): Path to save the output WAV file.
    - mic_index (int): Index of the microphone channel to extract from the matrix.
    - sample_rate (int): Sample rate for the WAV file. Default is 16000 Hz.
    """
    # Load the .mat file
    data = scipy.io.loadmat(mat_file_path)
    
    # Extract the matrix
    matrix = data[matrix_name]

    if matrix.shape[1] == 1:
        mic_index = 0

    # Extract the feature vector for the specified microphone index
    mic_data = matrix[:, mic_index]
    
    # Normalize the data to the range [-1, 1]
    mic_data_normalized = mic_data #/ np.max(np.abs(mic_data))
    
    # Convert to 16-bit PCM format
    mic_data_int16 = (mic_data_normalized * 32767).astype(np.int16)
    
    # Save the data to a WAV file
    write(output_wav_path, sample_rate, mic_data_int16)
    
    print(f"WAV file saved as {output_wav_path}")

# def save_vector_as_wav(feature, output_wav_path='my_output_file.wav', sample_rate=16000):
#     """
#     Save a feature vector as a WAV file.

#     Parameters:
#     - feature (array-like): A 1D numpy array or PyTorch tensor of audio data.
#     - output_wav_path (str): Path to save the output WAV file. Default is 'my_output_file.wav'.
#     - sample_rate (int): Sample rate for the WAV file. Default is 16000 Hz.
#     """
#     import numpy as np
#     import torch

#     # Convert PyTorch tensor to NumPy array if necessary
#     if isinstance(feature, torch.Tensor):
#         feature = feature.detach().cpu().numpy()
    
#     # Flatten the feature array if it is not already 1D
#     feature = np.squeeze(feature)
    
#     # Normalize the data to the range [-1, 1]
#     mic_data_normalized = feature # / np.max(np.abs(feature))
    
#     # Convert to 16-bit PCM format
#     mic_data_int16 = (mic_data_normalized * 32767).astype(np.int16)
    
#     # Save the data to a WAV file
#     from scipy.io.wavfile import write
#     write(output_wav_path, sample_rate, mic_data_int16)
    
#     print(f"WAV file saved as {output_wav_path}")

def save_vector_as_wav(feature, output_wav_path='my_output_file.wav', sample_rate=16000):
    """
    Save a feature vector as a WAV file.

    Parameters:
    - feature: PyTorch tensor or NumPy array.
    - output_wav_path (str): Path to save the output WAV file.
    - sample_rate (int): Sample rate for the WAV file.
    """
    # Convert PyTorch tensor to NumPy if necessary
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()

    # Ensure it's a 1D NumPy array
    feature = np.squeeze(feature)  # Remove unnecessary dimensions
    if feature.ndim != 1:
        raise ValueError(f"Expected a 1D array, but got shape {feature.shape}")

    # Ensure data type is float32 (soundfile requires float32 or int16)
    feature = feature.astype(np.float32)

    # Save as WAV file
    sf.write(output_wav_path, feature, sample_rate)
    print(f"WAV file saved as {output_wav_path}")

    
def return_as_complex(Y_stft):
    
    B, M, F, L = Y_stft.size()
    Y_STFT = Y_stft.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()
    Y = torch.view_as_complex(Y_STFT)
    return Y


def snr(signal: torch.Tensor, noise: torch.Tensor):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB, averaged across all batches.

    Args:
        signal (torch.Tensor): Tensor of shape (batch_size, signal_length) - clean signal.
        noise (torch.Tensor): Tensor of shape (batch_size, signal_length) - noise.

    Returns:
        torch.Tensor: The average SNR in dB across the batch.
    """
    # Compute power of signal and noise
    signal_power = torch.mean(signal ** 2, dim=1)  # Shape: (batch_size,)
    noise_power = torch.mean(noise ** 2, dim=1)    # Shape: (batch_size,)

    # Compute SNR for each sample in batch
    snr_per_sample = 10 * torch.log10(signal_power / noise_power)  # Shape: (batch_size,)

    # Average SNR across batch
    avg_snr = torch.mean(snr_per_sample)  # Scalar

    return avg_snr

def snr_np(signal: np.ndarray, noise: np.ndarray):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB, averaged across samples or batches.

    Args:
        signal (np.ndarray): Array of shape (signal_length,) or (batch_size, signal_length) – clean signal.
        noise (np.ndarray): Array of shape (signal_length,) or (batch_size, signal_length) – noise.

    Returns:
        float: The average SNR in dB.
    """
    # If inputs are 1D, compute directly
    if signal.ndim == 1:
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / noise_power + 1e-10)
    
    # If inputs are 2D (batch_size, signal_length), compute SNR per batch and average
    elif signal.ndim == 2:
        signal_power = np.mean(signal ** 2, axis=1)
        noise_power = np.mean(noise ** 2, axis=1)
        snr_per_sample = 10 * np.log10(signal_power / noise_power + 1e-10)
        return np.mean(snr_per_sample)
    
    else:
        raise ValueError("Input signal must be 1D or 2D.")