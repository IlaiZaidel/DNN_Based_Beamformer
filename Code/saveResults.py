import os
import scipy.io as sio
from datetime import datetime

def saveResults(
    Y,
    FIRST_SPEAKER_stft,
    SECOND_SPEAKER_stft,
    W_Stage1_left,
    X_hat_Stage1_C_left,
    y,
    first_speaker,
    second_speaker,
    x_hat_stage1_left,
    results_path,
    i,
    fs,
):
    """
    Save STFT-domain and time-domain results as .mat files.

    Stored (STFT-domain):
        - Y_STFT                 : mixture STFT
        - FIRST_SPEAKER_STFT     : first speaker STFT
        - SECOND_SPEAKER_STFT    : second speaker STFT
        - W_Stage1_left          : beamformer weights (left)
        - X_hat_Stage1_C_left    : enhanced STFT (left)

    Stored (time-domain):
        - y                      : mixture (time-domain)
        - first_speaker          : clean first speaker (time-domain)
        - second_speaker         : clean second speaker (time-domain)
        - x_hat_stage1_left      : enhanced signal (time-domain, left)
    """

    os.makedirs(results_path, mode=0o777, exist_ok=True)

    now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    # ---------- STFT domain ----------
    Y_STFT_np                = Y.cpu().detach().numpy()
    FIRST_SPEAKER_STFT_np    = FIRST_SPEAKER_stft.cpu().detach().numpy()
    SECOND_SPEAKER_STFT_np   = SECOND_SPEAKER_stft.cpu().detach().numpy()
    W_Stage1_left_np         = W_Stage1_left.cpu().detach().numpy()
    X_hat_Stage1_C_left_np   = X_hat_Stage1_C_left.cpu().detach().numpy()

    stft_filename = os.path.join(
        results_path, f"TEST_STFT_domain_results_{now}_{i}.mat"
    )
    stft_dict = {
        "Y_STFT": Y_STFT_np,
        "FIRST_SPEAKER_STFT": FIRST_SPEAKER_STFT_np,
        "SECOND_SPEAKER_STFT": SECOND_SPEAKER_STFT_np,
        "W_Stage1_left": W_Stage1_left_np,
        "X_hat_Stage1_C_left": X_hat_Stage1_C_left_np,
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    sio.savemat(stft_filename, stft_dict)

    # ---------- Time domain ----------
    y_np                 = y.cpu().detach().numpy()
    first_speaker_np     = first_speaker.cpu().detach().numpy()
    second_speaker_np    = second_speaker.cpu().detach().numpy()
    x_hat_stage1_left_np = x_hat_stage1_left.cpu().detach().numpy()

    time_filename = os.path.join(
        results_path, f"TEST_time_domain_results_{now}_{i}.mat"
    )
    time_dict = {
        "y": y_np,
        "first_speaker": first_speaker_np,
        "second_speaker": second_speaker_np,
        "x_hat_stage1_left": x_hat_stage1_left_np,
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    sio.savemat(time_filename, time_dict)
