import scipy.io
import soundfile as sf
import numpy as np
import os

mat_file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/25_10_2025_PINK_DUAL/TEST_time_domain_results_26_10_2025__09_49_50_0.mat"
mat_file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_10_2025_BABBLE_ATTEN_MIX_SNR_3/TEST_time_domain_results_29_10_2025__15_09_44_0.mat"
out_prefix = "AAA_PAPER_28_10_2025_BABBLE_ATTEN_MIX_3_SNR_FOR_PAPER"
out_dir = "/home/dsi/ilaiz/"
key = "x_hat_stage1"   # choose "x_hat_stage1" or "y"

mat_data = scipy.io.loadmat(mat_file_path)
print("Keys:", mat_data.keys())

fs = 16000
if "fs" in mat_data:
    fs = int(np.array(mat_data["fs"]).squeeze())

if key == "x_hat_stage1":
    left  = mat_data["x_hat_stage1_left"].astype(np.float32)
    right = mat_data["x_hat_stage1_right"].astype(np.float32)

    # normalize by same factor
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left  /= max_val
        right /= max_val

    num_examples = left.shape[0]
    for i in range(num_examples):
        stereo = np.stack([left[i], right[i]], axis=-1)  # (T,2)
        fname = os.path.join(out_dir, f"{out_prefix}__x_hat_stage1_{i}.wav")
        sf.write(fname, stereo, fs)
        print(f"Saved {fname}, shape={stereo.shape}, sr={fs}")

elif key == "y":
    y = mat_data["y"].astype(np.float32)
    num_examples = y.shape[0]
    for i in range(num_examples):
        stereo = np.stack([y[i, :, 0], y[i, :, -1]], axis=-1)  # first & last mic
        fname = os.path.join(out_dir, f"{out_prefix}_y_{i}.wav")
        sf.write(fname, stereo, fs)
        print(f"Saved {fname}, shape={stereo.shape}, sr={fs}")
