import torch



def RTF_Estimation(Rxx, B, M, F, L):

    mic_ref = 4
    # Rxx shape is torch.Size([8, 257, 8, 8])
    # Preallocate tensors
    a = torch.zeros((B, M, F), dtype=torch.cfloat, device=Rxx.device)


    # Eigen decomposition of Rxx
    eigenvalues_Rxx, eigenvectors_Rxx = torch.linalg.eigh(Rxx) # eigenvalues is torch.Size([8, 257, 8]) and eigenvectors is torch.Size([8, 257, 8, 8])
    i_max_k = torch.argmax(eigenvalues_Rxx, dim=-1)  # Index of the max eigenvalue (B, F) 8,257
    # Use advanced indexing to select eigenvectors corresponding to max eigenvalues
    batch_indices = torch.arange(B, device=Rxx.device).view(B, 1)  # Shape: (B, 1)
    freq_indices = torch.arange(F, device=Rxx.device).view(1, F)  # Shape: (1, F)

    # Select eigenvectors corresponding to the max eigenvalue
    phi_k = eigenvectors_Rxx[batch_indices, freq_indices, :, i_max_k]  # Shape: (B, F, M)
    # Compute the steering vector

    a = phi_k / phi_k[:, :, mic_ref-1].unsqueeze(-1)

    # Change the shape of `a` to (B, M, F)
    a = a.permute(0, 2, 1)  # (B, M, F)
    return a


def clean_speech_estimation(Y, B, F, M, L):

    # Compute Rxx in a vectorized manner
    Rxx = torch.einsum('bmfl,bnfl->bfmn', Y, Y.conj()) / (L )  # Shape: (B, F, M, M)
    Rxx = Rxx.to(torch.complex64)
    return Rxx

def RTF_from_clean_speech(Y):
    B, M, F, L = Y.size()  

    Rxx = clean_speech_estimation(Y, B, F, M, L)
    a = RTF_Estimation(Rxx, B, M, F, L)
    a = a.unsqueeze(-1) 
    return a

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
