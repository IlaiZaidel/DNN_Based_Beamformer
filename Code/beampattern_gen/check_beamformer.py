import os
import sys
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from RTF_covariance_whitening import (
    covariance_whitening, fix_covariance_whitening
)

from RTF_from_clean_speech import RTF_from_clean_speech
from utils import Preprocesing, save_feature_as_wav, return_as_complex, Postprocessing, save_vector_as_wav
import os
import sys
import torch
import numpy as np
import scipy.io
import soundfile as sf
# Constants
DEVICE_ID = 2
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
WIN_LEN = 512
HOP_LEN = WIN_LEN // 4
SAMPLE_RATE = 16000
MIC_REF = 4


#path =  '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/reference_feature_vector_0.mat'
path =  '/dsi/gannot-lab1/datasets/Ilai_data/WhiteNoiseOnly_Train/my_feature_vector_0.mat'
#path =  '/dsi/gannot-lab1/datasets/Ilai_data/Test_set/feature_vector_0.mat'
i = 0
data = scipy.io.loadmat(path)
'''
dict_keys(['__header__', '__version__', '__globals__', 'beta', 'x_position', 'n_position', 'mic_position',
'room_dim', 'angleOrientation', 'angle_x', 'angle_n', 'radius', 'feature', 'fulloriginal', 'fullnoise',
'target_s'])
'''


# Extract signals
feature_vector_time = torch.from_numpy(data['feature']).float().to(DEVICE)
original_time = torch.from_numpy(data['fulloriginal']).float().to(DEVICE)
fullnoise = torch.from_numpy(data['fullnoise']).float().to(DEVICE)
white_noise = feature_vector_time - original_time
original_s = torch.from_numpy(data['target_s']).float().to(DEVICE)
residual_noise = fullnoise - white_noise
def save_tensor_as_wav(tensor, filename, sample_rate):
    tensor = tensor.squeeze().cpu().numpy()
    if tensor.ndim > 1:
        tensor = tensor[:, 0]  # Take the first channel if multi-channel
    sf.write(filename, tensor, sample_rate)
    print(f"Saved {filename}")

# Save audio signals
save_tensor_as_wav(feature_vector_time, 'my_feature_vector0.wav', SAMPLE_RATE)
save_tensor_as_wav(original_time, 'my_feature_vector0_original.wav', SAMPLE_RATE)
save_tensor_as_wav(fullnoise, 'my_feature_vector0_fullnoise.wav', SAMPLE_RATE)
save_tensor_as_wav(white_noise, 'my_feature_vector0_white_noise.wav', SAMPLE_RATE)
save_tensor_as_wav(residual_noise, 'my_feature_vector0_residual_noise.wav', SAMPLE_RATE)
# save_feature_as_wav(feature_vector_time, output_wav_path='my_feature_vector0.wav', sample_rate=16000)
# save_feature_as_wav(original_time, output_wav_path='my_feature_vector0_original.wav', sample_rate=16000)
# save_feature_as_wav(fullnoise, output_wav_path='my_feature_vector0_fullnoise.wav', sample_rate=16000)
# save_feature_as_wav(white_noise, output_wav_path='my_feature_vector0_white_noise.wav', sample_rate=16000)


Y_STFT = Preprocesing(feature_vector_time, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE) # torch.Size([1, 8, 514, 497])
X_STFT = Preprocesing(original_time, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)       # torch.Size([1, 8, 514, 497]) #fulloriginal
Noise_dir_STFT = Preprocesing(fullnoise, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)   # torch.Size([1, 8, 514, 497])
White_noise = Preprocesing(white_noise, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)    # torch.Size([1, 8, 514, 497])
Speech_STFT =  Preprocesing(original_s, WIN_LEN, SAMPLE_RATE, 4, HOP_LEN, DEVICE)    # torch.Size([1, 1, 514, 497])

Y = return_as_complex(Y_STFT)
X = return_as_complex(X_STFT)
Noise_Dir = return_as_complex(Noise_dir_STFT)
White_Noise = return_as_complex(White_noise)  # torch.Size([1, 8, 257, 497])
Speech_STFT = return_as_complex(Speech_STFT)  # torch.Size([1, 1, 257, 497])

Speech_STFT = Speech_STFT.squeeze() # torch.Size([257, 497])

#a = RTF_from_clean_speech(X)



#mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/14_01_2025/TEST_STFT_domain_results_14_01_2025__11_00_56_0.mat'
#mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/Ilai_Results_Non_Reverberant_Environment/28_01_2025/TEST_STFT_domain_results_28_01_2025__12_55_47_0.mat'
mat_file_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/31_01_2025_temp/TEST_STFT_domain_results_30_01_2025__21_57_24_0.mat'
data = scipy.io.loadmat(mat_file_path)

W_Stage1 = torch.tensor(data['W_STFT_timeFixed'][0]).unsqueeze(0) # torch.Size([1, 8, 257, 1])
#W_Stage2 = torch.tensor(data['W_Stage2'][0]).unsqueeze(0)         # torch.Size([1, 257, 497])
X_hat_Stage1 = torch.tensor(data['X_hat_Stage1'][0]).unsqueeze(0) #torch.Size([1, 257, 497])
##### Beamforming Output #####

wx = torch.mul(torch.conj(W_Stage1), X)
wx = torch.sum(wx, dim=1) #.squeeze(-1)  wa is shaped torch.Size([1, 257, 497])
#wx_stage2 = torch.mul(torch.conj(W_Stage2), wx) #torch.Size([1, 257, 497])


wy = torch.mul(torch.conj(W_Stage1), Y)
wy = torch.sum(wy, dim=1) #.squeeze(-1)  wa is shaped torch.Size([1, 257, 497])
#wy_stage2 = torch.mul(torch.conj(W_Stage2), wy) #torch.Size([1, 257, 497])

wn_dir = torch.mul(torch.conj(W_Stage1),Noise_Dir)
wn_dir = torch.sum(wn_dir, dim=1) #.squeeze(-1)  wa is shaped torch.Size([1, 257, 497])
#wn_dir_stage2 = torch.mul(torch.conj(W_Stage2), wn_dir) #torch.Size([1, 257, 497])

wn_white = torch.mul(torch.conj(W_Stage1),White_Noise)
wn_white = torch.sum(wn_white, dim=1) #.squeeze(-1)  wa is shaped torch.Size([1, 257, 497])
#wn_white_stage2 = torch.mul(torch.conj(W_Stage2), wn_white) #torch.Size([1, 257, 497])


x_hat_Stage1 = Postprocessing(X_hat_Stage1, HOP_LEN, WIN_LEN, DEVICE) # torch.Size([1, 64000])
x_stage2 = Postprocessing(wx, HOP_LEN, WIN_LEN, DEVICE) # torch.Size([1, 64000])
y_stage2 = Postprocessing(wy, HOP_LEN, WIN_LEN, DEVICE) # torch.Size([1, 64000])
noise_dir_stage2 = Postprocessing(wn_dir, HOP_LEN, WIN_LEN, DEVICE) #torch.Size([1, 64000])
white_noise_stage2 = Postprocessing(wn_white, HOP_LEN, WIN_LEN, DEVICE) #torch.Size([1, 64000])


save_vector_as_wav(x_hat_Stage1, output_wav_path='TEMP_31_01_B_x_hat_stage1_file.wav', sample_rate=16000)
save_vector_as_wav(x_stage2, output_wav_path='TEMP_31_01_B_x_stage2_file.wav', sample_rate=16000)
save_vector_as_wav(y_stage2, output_wav_path='TEMP_31_01_B_y_stage2_file.wav', sample_rate=16000)
save_vector_as_wav(noise_dir_stage2, output_wav_path='TEMP_31_01_B_noise_dir_stage2_file.wav', sample_rate=16000)
save_vector_as_wav(white_noise_stage2, output_wav_path='TEMP_31_01_B_white_noise_stage2_file.wav', sample_rate=16000)
save_vector_as_wav(original_s, output_wav_path='TEMP_31_01_B_original_file.wav', sample_rate=16000)



print('done')