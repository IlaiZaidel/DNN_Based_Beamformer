import torch
import numpy as np
###########################################
# CORRECT RTF ESTIMATION
###########################################
def RTF_Cov_W(Rnn, Ryy, B, M, F, L):

    mic_ref = 4

    # Preallocate tensors
    a = torch.zeros((B, M, F), dtype=torch.cfloat, device=Rnn.device)

    # Precompute square root and inverse square root matrices for Rnn
    eigenvalues_Rnn, eigenvectors_Rnn = torch.linalg.eigh(Rnn)  # Batch eigendecomposition (B, F, M, M)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_Rnn.to(torch.cfloat))
    inv_sqrt_eigenvalues = 1 / sqrt_eigenvalues

    Rn1_2 = eigenvectors_Rnn @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors_Rnn.conj().transpose(-2, -1)
    invRn1_2 = eigenvectors_Rnn @ torch.diag_embed(inv_sqrt_eigenvalues) @ eigenvectors_Rnn.conj().transpose(-2, -1)

    # Calculate Rz
    Rz = invRn1_2 @ Ryy @ invRn1_2.conj().transpose(-2, -1)

    # Eigen decomposition of Rz
    eigenvalues_Rz, eigenvectors_Rz = torch.linalg.eigh(Rz)
    i_max_k = torch.argmax(eigenvalues_Rz, dim=-1)  # Index of the max eigenvalue (B, F)
    # Use advanced indexing to select eigenvectors corresponding to max eigenvalues
    batch_indices = torch.arange(B, device=Rnn.device).view(B, 1)  # Shape: (B, 1)
    freq_indices = torch.arange(F, device=Rnn.device).view(1, F)  # Shape: (1, F)

    # Select eigenvectors corresponding to the max eigenvalue
    phi_k = eigenvectors_Rz[batch_indices, freq_indices, :, i_max_k]  # Shape: (B, F, M)
    # Compute the steering vector
    temp_k = (Rn1_2.conj().transpose(-2, -1) @ phi_k.unsqueeze(-1)).squeeze(-1)  # (B, F, M)
    a = temp_k / temp_k[:, :, mic_ref-1].unsqueeze(-1)

    # Change the shape of `a` to (B, M, F)
    a = a.permute(0, 2, 1)  # (B, M, F)
    return a


def noise_estimation(noise, Ln):

    # Extract the relevant slices of the noise tensor
    n_k_sliced = noise[:, :, :, :Ln]  # Shape: (B, M, F, Ln)
    
    # Compute Rnn in a vectorized manner
    Rnn = torch.einsum('bmfl,bnfl->bfmn', n_k_sliced, n_k_sliced.conj()) / Ln  # Shape: (B, F, M, M)
    # Ensure the output is of type torch.complex64
    Rnn = Rnn.to(torch.complex64)
    return Rnn

def mix_estimation(Y, L, Ln):

    #Ln = 62  # Assumed pre-computed
    # Extract the relevant slices of the mixture tensor
    Y_k_sliced = Y[:, :, :, Ln:L]  # Shape: (B, M, F, L-Ln)
    
    # Compute Ryy in a vectorized manner
    Ryy = torch.einsum('bmfl,bnfl->bfmn', Y_k_sliced, Y_k_sliced.conj()) / (L - Ln)  # Shape: (B, F, M, M)
    Ryy = Ryy.to(torch.complex64)
    return Ryy

def covariance_whitening(Y, noise_time_only):
    B, M, F, L = Y.size()  
    fs = 16000
    hop_size = 128
    Ln = int(noise_time_only * fs // hop_size)
    Rnn = noise_estimation(Y,  Ln)
    Ryy = mix_estimation(Y, L, Ln)
    a_cw = RTF_Cov_W(Rnn, Ryy, B, M, F, L)
    a_cw = a_cw.unsqueeze(-1) 
    return a_cw

def fix_covariance_whitening(a_cw):
    B, M, F, L = a_cw.size()

    # Compute the absolute values
    abs_values = torch.abs(a_cw[:, :, :, 0])  # Shape: (B, M, F)

    # Compute thresholds for all batches and microphones
    thresholds = 3 * torch.mean(abs_values, dim=-1, keepdim=True)  # Shape: (B, M, 1)

    # Create a mask for values exceeding the threshold
    mask = abs_values > thresholds  # Shape: (B, M, F)

    # Generate random values (-1 or 1) for all elements
    random_values = 2 * torch.bernoulli(0.5 * torch.ones_like(abs_values)) - 1  # Shape: (B, M, F)
    random_values = random_values.to(torch.cfloat)

    # Apply the random values to the elements that exceed the threshold
    a_cw[:, :, :, 0][mask] = random_values[mask]

    return a_cw
