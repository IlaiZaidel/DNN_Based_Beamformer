import torch
import numpy as np
from numpy import linalg as LA
from RTF_CW_estimation import RTF_CW
# Dimensions
M = 5 # Number of microphones
F = 1  # Number of frequency bins
L = 6000  # Number of frames


# RTF vector
a = np.random.randn(M, F)
a[0, :] = 1

# Speech signal
s =  np.random.uniform(1, 5, size=(F, L))


Ln = int(0.2*L)

s[:, :Ln] = 0

# Create Noise:
target_eigenvalues = np.random.uniform(0.4, 1.3, M)  # Eigenvalues ~ 1 ± 0.1

# Generate noise matrix
noise = np.zeros((M, F, L), dtype=np.float64)

for f in range(F):
    # Generate a random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(M, M))  # Ensure orthogonal matrix
    
    # Construct diagonal matrix with target eigenvalues
    Lambda = np.diag(target_eigenvalues)
    
    # Generate covariance matrix with desired eigenvalues
    R = Q @ Lambda @ Q.T
    
    # Fill noise along the time frames
    for l in range(L):
        noise[:, f, l] = np.random.multivariate_normal(np.zeros(M), R)




# Noisy signal
#noise = np.random.random((M, F, L))*10
s_expanded = s[np.newaxis, :, :]
result = a[:, :, np.newaxis] * s_expanded

Y = result + noise
Ryy = np.zeros((F, M, M))
for k in range(F):
    Y_k_sliced = Y[:,k,Ln:L]
    Ryy[k] = np.matmul(Y_k_sliced, Y_k_sliced.conj().T)/(L-Ln)


Rnn = np.zeros((F, M, M))
for k in range(F):
    Rnn[k] = np.matmul(noise[:,k,:], noise[:,k,:].conj().T)#/L

# for k in range(F):
#     sum = np.zeros((M,1))
#     for l in range(L):
#         noise_vector = noise[:, k, l].reshape(-1, 1)
#         sum = sum + np.matmul(noise_vector, noise_vector.conj().T)
        
#     Rnn[k] = sum/L







Rnn_CW, a_cw, R_yy_cw, R_z = RTF_CW(Y, Rnn)

print("a is: ")
print(a)

print("a_cw is: ")
print(a_cw)




print("Rnn[0] is ")
print(np.array2string( Rnn[0], formatter={'float_kind': lambda x: f"{x:7.3f}"}))
print("Rnn_CW[0] is ")
print(np.array2string( Rnn_CW[0], formatter={'float_kind': lambda x: f"{x:7.3f}"}))

print("Ryy[0] is ")
print(np.array2string( Ryy[0], formatter={'float_kind': lambda x: f"{x:7.3f}"}))
print("Ryy_CW[0] is ")
print(np.array2string( R_yy_cw[0], formatter={'float_kind': lambda x: f"{x:7.3f}"}))

print("Rz[0] is ")
print(np.array2string( R_z[0], formatter={'float_kind': lambda x: f"{x:7.3f}"}))


Rn1_2 = np.zeros_like(Rnn)
invRn1_2 = np.zeros_like(Rnn)


for k in range(F):
 
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Rnn[k])  
    Rn1_2[k] = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inv_sqrt_eigenvalues = np.diag(1 / np.sqrt(eigenvalues))
    invRn1_2[k] = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.conj().T


# Check eigenvalues of Rnn
for k in range(F):
    eigenvalues = np.linalg.eigvals(Rnn[k])
    print(f"Eigenvalues of Rnn[{k}]: {eigenvalues}")

# Check eigenvalues of Rz
for k in range(F):
    eigenvalues = np.linalg.eigvals(R_z[k])
    print(f"Eigenvalues of R_z[{k}]: {eigenvalues}")

# Compare unnormalized a_CW
for k in range(F):
    eigenvalues, eigenvectors = np.linalg.eigh(R_z[k])
    i_max_k = np.argmax(eigenvalues)
    phi_k = eigenvectors[:, i_max_k]
    temp_k = np.matmul(Rn1_2[k].conj().T, phi_k)
    print(f"Unnormalized a_CW[:, {k}]: {temp_k}")