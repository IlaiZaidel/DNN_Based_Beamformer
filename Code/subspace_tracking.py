

import torch
import numpy as np

from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening, noise_estimation
@torch.no_grad()

def pastd_rank1_whitened(Y, noise_time_only, beta=0.95):
    """
    Run rank-1 PASTd on each time frame (after whitening) for each batch and frequency bin.

    Inputs:
        Y: (B, M, F, L) complex STFT
        Rnn: (B, F, M, M) noise covariance matrix per freq
        beta: forgetting factor

    Returns:
        W: (B, F, M, L - Ln) dominant eigenvectors at each frame
    """
    B, M, F, L = Y.shape
    fs = 16000
    hop_size = 128
    Ln = int(noise_time_only * fs // hop_size)
    Rnn = noise_estimation(Y,  Ln) #torch.Size([16, 514, 8, 8])
    I = torch.eye(M, dtype=Rnn.dtype, device=Rnn.device).expand_as(Rnn)
    Rnn = Rnn + 1e-6 * I  # Diagonal loading
    T = L - Ln
    device = Y.device
    dtype = Y.dtype

    # Whitening matrices: inv_sqrt(Rnn)
    eigvals, eigvecs = torch.linalg.eigh(Rnn)  # (B, F, M)
    eigvals = eigvals.to(dtype=torch.complex64)  # <- this is critical
    inv_sqrt_vals = 1 / torch.sqrt(eigvals)
    inv_sqrt_vals = inv_sqrt_vals.to(torch.complex64)



    inv_Rnn12 = eigvecs @ torch.diag_embed(inv_sqrt_vals) @ eigvecs.conj().transpose(-2, -1)  # (B, F, M, M) torch.Size([16, 257, 8, 8])

    # Prepare output
    W = torch.zeros((B, F, M, T), dtype=dtype, device=device)

    # Initialize w and d
    w = torch.randn(B, F, M, dtype=dtype, device=device)
    w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
    d = torch.full((B, F), 1e-3, dtype=dtype, device=device)

    # Time loop
    for t in range(T):
        y_t = Y[:, :, :, Ln + t]  # (B, M, F)
        y_t = y_t.permute(0, 2, 1)  # (B, F, M)
        y_t = y_t.to(torch.complex64)
        # Whitening: x̃ = inv(Rnn)^½ @ y
        x_t = torch.einsum('bfmn,bfn->bfm', inv_Rnn12, y_t)  # (B, F, M) torch.Size([16, 257, 8])
        x_t = x_t.to(torch.complex64)
        w = w.to(torch.complex64)
        # Project and update
        y_proj = torch.einsum('bfm,bfm->bf', w.conj(), x_t)  # (B, F)
        d = beta * d + torch.abs(y_proj) ** 2
        gain = y_proj.conj() / d  # (B, F)
        residual = x_t - w * y_proj.unsqueeze(-1)  # (B, F, M)
        w = w + gain.unsqueeze(-1) * residual
        w = w / torch.linalg.norm(w, dim=-1, keepdim=True)

        # Save output
        W[:, :, :, t] = w

    return W, eigvals, eigvecs  # shape: (B, F, M, T)


def rtf_from_subspace_tracking(W, eigvals, eigvecs,noise_only_time, mic_ref):
    
    # Make sure eigvals are float32 BEFORE turning complex
    eigvals = eigvals.to(torch.float32)  # 
    eigvecs = eigvecs.to(torch.complex64)
    W = W.to(torch.complex64)

    sqrt_vals = torch.sqrt(eigvals).to(torch.complex64)  # now complex64, not complex128
    Rnn12 = eigvecs @ torch.diag_embed(sqrt_vals) @ eigvecs.conj().transpose(-2, -1)  # complex64

    a_hat = torch.einsum('bfmn,bfnt->bfmt', Rnn12, W)  # now both operands are complex64
    a_hat = a_hat / a_hat[:, :, mic_ref, :].unsqueeze(-2)  # torch.Size([8, 257, 8, 435])
    return a_hat



