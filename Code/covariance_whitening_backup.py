import torch
import numpy as np

def RTF_Cov_W(Rnn, Ryy, B, M, F, L):
    mic_ref = 4
    Rn1_2 = torch.zeros_like(Rnn)
    invRn1_2 = torch.zeros_like(Rnn)
    Rz = torch.zeros_like(Rnn)

    a = torch.zeros((B, M, F), dtype=torch.cfloat, device=Rnn.device)

    for b in range(B):
        for k in range(F):
            eigenvalues, eigenvectors = torch.linalg.eigh(Rnn[b, k])
            Rn1_2[b, k] = eigenvectors @ torch.diag(torch.sqrt(eigenvalues.to(torch.cfloat))) @ eigenvectors.conj().T

            inv_sqrt_eigenvalues = torch.diag(1 / torch.sqrt(eigenvalues.to(torch.cfloat)))
            invRn1_2[b, k] = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.conj().T

            Rz[b, k] = invRn1_2[b, k] @ Ryy[b, k] @ invRn1_2[b, k].conj().T

            eigenvalues, eigenvectors = torch.linalg.eigh(Rz[b, k])
            i_max_k = torch.argmax(eigenvalues)
            phi_k = eigenvectors[:, i_max_k]

            temp_k = Rn1_2[b, k].conj().T @ phi_k
            a[b, :, k] = temp_k / temp_k[mic_ref-1]

    return a

def noise_estimation(noise, B, F, M, Ln):
    Rnn = torch.zeros((B, F, M, M), dtype=torch.cfloat, device=noise.device)
    for b in range(B):
        for k in range(F):
            n_k_sliced = noise[b, :, k, :Ln]
            Rnn[b, k] = (n_k_sliced @ n_k_sliced.conj().T) / Ln
    return Rnn

def mix_estimation(Y, B, F, M, L):
    Ln = 62 #floor(noise_tim_fn*fs/R);
    Ryy = torch.zeros((B, F, M, M), dtype=torch.cfloat, device=Y.device)
    for b in range(B):
        for k in range(F):
            Y_k_sliced = Y[b, :, k, Ln:L]
            Ryy[b, k] = (Y_k_sliced @ Y_k_sliced.conj().T) / (L - Ln)
    return Ryy

def covariance_whitening(Y):
    B, M, F, L = Y.size()  
    Ln = 62
    Rnn = noise_estimation(Y, B, F, M, Ln)
    Ryy = mix_estimation(Y, B, F, M, L)
    a_cw = RTF_Cov_W(Rnn, Ryy, B, M, F, L)
    a_cw = a_cw.unsqueeze(-1) 
    return a_cw

def fix_covariance_whitening(a_cw):
    B, M, F, L = a_cw.size()
    K =F
    #a_cw = covariance_whitening(Y)
    # # Assuming G_f is already computed: shape (B, M, F)
    # # Hermitian symmetry to extend G_f
    # # G_f_full = torch.cat([G_f, torch.conj(G_f[0,:, 255:1:-1])], dim=2)

    # # Hermitian symmetry to extend G_f
    # G_f_full = torch.cat([G_f, torch.conj(G_f[:, :, 1:256].flip(dims=[2]))], dim=2)

    # # IFFT to time domain along frequency axis
    # g_f = torch.fft.ifft(G_f_full, dim=2)

    # # Truncate in the frequency domain
    # G_f_trc_full = torch.fft.fft(g_f, dim=2)
    # G_f = G_f_trc_full[:, :, :K] 

    # Expand along a new dimension if needed
#    G = G_f.unsqueeze(-1)  # Shape: (B, M, K, 1)

    for b in range(B):  # Iterate over batches
        for m in range(M):  # Iterate over microphones
            # Get the absolute values for the current batch and microphone
            abs_values = torch.abs(a_cw[b, m, :, 0])  # Shape: (F,)
            
            # Compute the threshold for the current microphone in the current batch
            threshold = 3 * torch.mean(abs_values)  # Scalar threshold
            
            # Find the indices where the absolute value exceeds the threshold
            indices = torch.where(abs_values > threshold)[0]  # Shape: (num_indices,)
            
            # Generate random values (-1 or 1) for the selected indices
            random_values = 2 * torch.bernoulli(0.5 * torch.ones(len(indices), device=a_cw.device)) - 1  # Shape: (num_indices,)
            random_values = random_values.to(torch.cfloat) 
            # Assign the random values to the selected indices
            a_cw[b, m, indices, :] = random_values.unsqueeze(-1)  # Match shape (num_indices, 1)

    return a_cw