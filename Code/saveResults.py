import os
import scipy.io as sio
from datetime import datetime


def saveResults(Y, X_stft,
                W_Stage1_left, W_Stage1_right,
                X_hat_Stage1_C_left, X_hat_Stage1_C_right,
                y, x_hat_stage1_left, x_hat_stage1_right,
                results_path, i, fs):
    """
    Save STFT-domain and time-domain results as .mat files.
    Compatible with Ilai's DNN Beamformer project output format.
    """

    # Ensure directory exists
    os.makedirs(results_path, mode=0o777, exist_ok=True)

    # Timestamp
    now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    ####### Save STFT Domain #######
    Y_STFT = Y.cpu().detach().numpy()
    X_STFT = X_stft.cpu().detach().numpy()

    W_Stage1_left_np  = W_Stage1_left.cpu().detach().numpy()
    W_Stage1_right_np = W_Stage1_right.cpu().detach().numpy()

    X_hat_Stage1_C_left_np  = X_hat_Stage1_C_left.cpu().detach().numpy()
    X_hat_Stage1_C_right_np = X_hat_Stage1_C_right.cpu().detach().numpy()

    stft_filename = os.path.join(results_path, f"TEST_STFT_domain_results_{now}_{i}.mat")
    stft_dict = {
        "Y_STFT": Y_STFT,
        "X_STFT": X_STFT,
        "W_Stage1_left": W_Stage1_left_np,
        "W_Stage1_right": W_Stage1_right_np,
        "X_hat_Stage1_C_left": X_hat_Stage1_C_left_np,
        "X_hat_Stage1_C_right": X_hat_Stage1_C_right_np,
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    sio.savemat(stft_filename, stft_dict)

    ####### Save Time Domain #######
    y_np                  = y.cpu().detach().numpy()
    x_hat_stage1_left_np  = x_hat_stage1_left.cpu().detach().numpy()
    x_hat_stage1_right_np = x_hat_stage1_right.cpu().detach().numpy()

    time_filename = os.path.join(results_path, f"TEST_time_domain_results_{now}_{i}.mat")
    time_dict = {
        "y": y_np,
        "x_hat_stage1_left": x_hat_stage1_left_np,
        "x_hat_stage1_right": x_hat_stage1_right_np,
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    sio.savemat(time_filename, time_dict)
