import torch
import numpy as np
from RTF_covariance_whitening import RTF_Cov_W
from RTF_covariance_whitening import noise_estimation
from RTF_covariance_whitening import mix_estimation
from RTF_covariance_whitening import covariance_whitening
import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation

# PyTorch implementation
M = 8
L = 1000 * M
F = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate noise tensor
noise = torch.rand((M, F, L), dtype=torch.cfloat, device=device)

# Compute Rnn
Rnn = torch.zeros((F, M, M), dtype=torch.cfloat, device=device)
for i in range(F):
    Rnn[i] = (noise[:, i, :] @ noise[:, i, :].conj().transpose(0, 1)) / L

# Generate a random RTF vector `a`
a = torch.randn((M, F), dtype=torch.cfloat, device=device)
a[0, :] = 1

# Generate signal `sigS`
sigS = torch.randn((F, L), dtype=torch.cfloat, device=device)
Ln = int(0.2*L)
sigS[:,:Ln] = 0

# Calculate `after_rtf` using broadcasting
after_rtf = a[:, :, None] * sigS[None, :, :]

# Add the noise
Y = after_rtf + noise

# Compute signal covariance matrix `Rs`
sigma_s = torch.zeros(F, dtype=torch.cfloat, device=device)
Rs = torch.zeros((F, M, M), dtype=torch.cfloat, device=device)
for i in range(F):
    sigma_s[i] = (sigS[i, :] @ sigS[i, :].conj().T) / L
    Rs[i] = (sigma_s[i] ** 2) * torch.outer(a[:, i], a[:, i].conj())

# Combined covariance matrix (signal + noise)
Ry = Rs + Rnn


# Estimate noise covariance matrix
Rnn_cw = noise_estimation(Y, F, M, Ln)

# Estimate mixed covariance matrix
Ryy_cw = mix_estimation(Y, F, M, L)

# Compute RTF
a_cw = RTF_Cov_W(Rnn_cw, Ryy_cw, M, F, L)
a_cw = covariance_whitening(Y, F, M, L)
# Print results
print("Rnn[0] (true RTF):")
print(Rnn[0])
print("Rnn_cw[0] (estimated RTF):")
print(Rnn_cw[0])

print("Ryy[0] (true RTF):")
print(Ry[0])
print("Ryy_cw[0] (estimated RTF):")
print(Ryy_cw[0])

print("a (true RTF):")
print(a)
print("a_cw (estimated RTF):")
print(a_cw)

# Compute the Mean Square Error
mse = torch.mean(torch.abs(a - a_cw)**2)

# Print the result
print(f"Mean Square Error: {mse.item()}")