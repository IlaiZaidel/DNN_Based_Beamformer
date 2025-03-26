import sys
import os
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import scipy.io
from utils import Postprocessing
device_ids = 2
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
device = "cpu"




def compute_mse_from_stft(path_stft, win_len=512, device="cpu"):
    """
    Computes the Mean Square Error (MSE) between the original signal and the stage 2 reconstructed signal
    from an STFT domain results file.
    
    Args:
        path_stft (str): Path to the STFT .mat file.
        win_len (int): Window length for the STFT. Defaults to 512.
        device (str): Device to run the computation on ("cpu" or "cuda:X"). Defaults to "cpu".
    
    Returns:
        float: The computed Mean Square Error (MSE).
    """
    # Load the .mat file
    data = scipy.io.loadmat(path_stft)
    
    # Extract relevant data
    X_STFT = torch.tensor(data['X_STFT'][3])  # Shape (8, 514, 497)
    X_hat_Stage2_STFT = torch.tensor(data['X_hat_Stage2'])  # Shape (8, 257, 497)

    # Ensure tensors are on the correct device
    device = torch.device(device)
    X_STFT = X_STFT.to(device)
    X_hat_Stage2_STFT = X_hat_Stage2_STFT.to(device)
    
    # Reshape and convert X_STFT into complex form
    X_STFT_contiguous = X_STFT.contiguous()
    X_STFT_reshaped = X_STFT_contiguous.reshape(X_STFT.shape[0], X_STFT.shape[1] // 2, X_STFT.shape[2], 2)
    X_complex = torch.complex(X_STFT_reshaped[..., 0], X_STFT_reshaped[..., 1])  # Shape (B, F, L)

    # Define STFT parameters
    R = win_len // 4  # Hop length

    # Perform postprocessing to obtain time-domain signals
    x_original = Postprocessing(X_complex, R, win_len, device)  # Shape (8, 64000)
    x_hat_stage2 = Postprocessing(X_hat_Stage2_STFT, R, win_len, device)  # Shape (8, 64000)

    # Compute MSE
    mse = torch.mean((x_original - x_hat_stage2) ** 2)

    return mse.item()



'''
For 04_01_2025 Model:
'''
# First Reference
path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/04_01_2025/TEST_STFT_domain_results_05_01_2025__12_30_22_1.mat'
mse = compute_mse_from_stft(path)*100
print(f"Mean Square Error 04_01_2025 First Reference: {mse}")

path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/04_01_2025/TEST_STFT_domain_results_06_01_2025__17_17_02_0.mat'
mse = compute_mse_from_stft(path)*100
print(f"Mean Square Error 04_01_2025 Second Reference: {mse}")




'''
For 06_01_2025 Model:
'''
# First Reference
path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/06_01_2025/TEST_STFT_domain_results_06_01_2025__15_47_16_2.mat'
mse = compute_mse_from_stft(path)*100
print(f"Mean Square Error 06_01_2025 First Reference: {mse}")

path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/06_01_2025_second/TEST_STFT_domain_results_07_01_2025__10_34_34_0.mat'
mse = compute_mse_from_stft(path)*100
print(f"Mean Square Error 06_01_2025 Second Reference: {mse}")






print('done')