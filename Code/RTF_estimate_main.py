
import numpy as np

from numpy import linalg as LA
from RTF_CW_estimation import RTF_Cov_W
from RTF_CW_estimation import noise_estimation
from RTF_CW_estimation import mix_estimation
from RTF_CW_estimation import covariance_whitening
import scipy.io
import torch

M = 8
L = 1000*M
F = 2

noise = np.random.random((M,F,L))

Rnn = np.zeros((F, M, M))
for i in range(F):
    Rnn[i] = np.matmul(noise[:,i,:], noise[:,i,:].conj().T)/L
#print(Rnn[0])
a = np.random.randn(M,F)
a[0,:] = 1



# sigS = np.random.uniform(1, 5, size=(F, L))
sigS = np.random.standard_normal((F,L))

# Calculate after_rtf using broadcasting
after_rtf = a[:, :, np.newaxis] * sigS[np.newaxis, :, :]

# Add the noise
Y = after_rtf + noise


sigma_s = np.zeros(F)

Rs = np.zeros((F, M, M))  # Signal covariance matrix (F x M x M)
for i in range(F):
    sigma_s[i] = np.matmul(sigS[i,:],sigS[i,:].conj().T)/L
    Rs[i] = (sigma_s[i] ** 2) * np.outer(a[:, i], a[:, i].conj())




# Combined covariance matrix (signal + noise)
Ry = Rs + Rnn



Rnn_cw = noise_estimation(noise,F, M, int(L))

Ryy_cw = mix_estimation(Y, F, M, L)

a_cw = RTF_Cov_W(Rnn_cw, Ryy_cw, M, F, L)

#a_cw =  covariance_whitening(Y, F, M, L)

print("Rnn[0] (true RTF):")
print(Rnn[0])
print("Rnn[0] (true RTF):")
print(Rnn_cw[0])

print("Ryy[0] (true RTF):")
print(Ry[0])
print("Ryy_cw[0] (true RTF):")
print(Ryy_cw[0])

print("a (true RTF):")
print(a)
print("a_cw (estimated RTF):")
print(a_cw)



sigS2 =  np.random.standard_normal((F,L))
Ln = int(0.2*L)
sigS2[:,:Ln] = 0

# Calculate after_rtf using broadcasting
after_rtf_2 = a[:, :, np.newaxis] * sigS2[np.newaxis, :, :]

# Add the noise
Y_2 = after_rtf_2 + noise
a_cw_2 =  covariance_whitening(Y_2, F, M, L)

print("a (true RTF):")
print(a)
print("a_cw (estimated RTF):")
print(a_cw_2)

# # Path to the .mat file
# file_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/my_feature_vector_0.mat"

# # Load the .mat file
# data = scipy.io.loadmat(file_path)

# # Check the keys in the .mat file
# print(data.keys())
#  #dict_keys(['__header__', '__version__', '__globals__', 'beta', 'x_position', 'mic_position', 'room_dim', 'angleOrientation', 'angle_x', 'radius',
# #'n1_position', 'angle_n1', 'radius_n1', 'n2_position', 'angle_n2', 'radius_n2', 'n3_position', 'angle_n3', 'radius_n3', 'n4_position', 'angle_n4', 'radius_n4', 'n5_position',
# #  'angle_n5', 'radius_n5', 'n6_position', 'angle_n6', 'radius_n6', 'n7_position', 'angle_n7', 'radius_n7', 'n8_position', 'angle_n8', 'radius_n8', 'n9_position', 'angle_n9', 
# # 'radius_n9', 'n10_position', 'angle_n10', 'radius_n10', 'feature', 'fulloriginal', 'fullnoise', 'target_s'])
# feature_vector = data['feature']
# #y = feature_vector[:,4].reshape(64000, 1)
# # Print or check the loaded data
# #print(feature_vector)
# # Print the size (shape) of the feature_vector
# print("Size of feature_vector:", feature_vector.shape)
# y_tensor = torch.from_numpy(feature_vector).squeeze()
# win_len = 512
# w_analysis = torch.hamming_window(win_len)
# #y_stft = torch.stft(y_tensor, n_fft=512, hop_length=int(512/4), win_length=512, window=w_analysis, center=False, return_complex = True)

# # abs_tensor = torch.abs(y_stft) 
# # non_zero_mask = (abs_tensor.sum(dim=0) != 0)  # Sum across the F dimension and check non-zero columns
# # first_non_zero_index = torch.where(non_zero_mask)[0][0].item()  # Get the first non-zero index
# # print(f"The L-axis stops being zero starting from index: {first_non_zero_index}") # 
# # Ln = 59
# # Perform STFT for each column and stack results
# stft_results = []
# for m in range(M):
#     stft_result = torch.stft(
#         y_tensor[:, m],  # Take one column at a time
#         n_fft=512,
#         hop_length=512 // 4,
#         win_length=512,
#         window=w_analysis,
#         center=False,
#         return_complex=True
#     )
#     stft_results.append(stft_result)

# # Stack results along M dimension
# y_stft = torch.stack(stft_results, dim=0)  # Shape: (M, F, L)
# print("Size of feature_vector:", y_stft.shape)
# #print(y_stft)

# F = y_stft.shape[1]
# L = y_stft.shape[2]

# a_speech_cw = covariance_whitening(y_stft.numpy(), F, M, L)
# print(a_speech_cw)

# original_speech = data['target_s']
# original_speech = torch.from_numpy(original_speech).squeeze()
# s_stft  = torch.stft(original_speech, n_fft=512, hop_length=int(512/4), win_length=512, window=w_analysis, center=False, return_complex = True)
# print(s_stft.shape)
# print(a_speech_cw.shape)
# after_rtf_2 = a_speech_cw[:, :, np.newaxis] * s_stft[np.newaxis, :, :].numpy()
# after_rtf_2 = torch.from_numpy(after_rtf_2).squeeze()
# #mic_reference = torch.istft(after_rtf_2, n_fft=win_len, hop_length=int(512/4), win_length=win_len, window=w_analysis, center=False, return_complex = True)


# impulse_response = data['impulse_response']
# print(impulse_response.shape)

# import matplotlib.pyplot as plt


# # Assuming impulse_response is a numpy array with shape (7200, 8)
# # For demonstration, create a dummy array:
# impulse_response = np.random.rand(7200, 8)

# # Select the 4th column (index 3 since Python is 0-based indexing)
# column_4 = impulse_response[:, 1]
# # print(column_4.shape)
# # Plotting the 4th column
# plt.plot(column_4)
# plt.title("4th Column of Impulse Response")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.grid(True)
# plt.savefig("plot.png")
# print("Plot saved as plot.png")