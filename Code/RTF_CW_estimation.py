
###########################################
#NOT THE CORRECT ONE - DO NOT USE
###########################################
import torch
import numpy as np
from numpy import linalg as LA
# Not the updated one!!!!
def RTF_CW(Y, good_Rnn):

    # Assuming the size of Y is: Y = M,F,L = 8,257*2,497, where L is the number of frames
    # I'm assuming no batch index B like Y = B,M,F*C,L = B,8,257*2,497

    M, F, L = Y.shape
    Ln = int(0.2 * L)

    Ryy = np.zeros((F, M, M))
    for k in range(F):
        Y_k_sliced = Y[:,k,Ln:L]
        Ryy[k] = np.matmul(Y_k_sliced, Y_k_sliced.conj().T)/(L-Ln)

    Rnn = np.zeros((F, M, M))
    for k in range(F):
        Y_k_sliced = Y[:,k,:Ln]
        Rnn[k] = np.matmul(Y_k_sliced, Y_k_sliced.conj().T)/Ln



# Calculation of sqrt of Rnn:
    # Delete Later:
    Rnn = good_Rnn
    Rn1_2 = np.zeros_like(Rnn)
    invRn1_2 = np.zeros_like(Rnn)

    for k in range(F):
        # Eigenvalue decomposition
        epsilon = 1e-3
        Rnn[k] += epsilon * np.eye(M)
        eigenvalues, eigenvectors = np.linalg.eig(Rnn[k])  
        Rn1_2[k] = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
        inv_sqrt_eigenvalues = np.diag(1 / np.sqrt(eigenvalues))
        invRn1_2[k] = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.conj().T

    # Covariance matrix of Z:
        
    R_z = np.zeros((F, M, M))

    for k in range(F):
        R_z[k] = invRn1_2[k] @ Ryy[k] @ Rn1_2[k].conj().T

# This works correctly:
    # Eigenvalue decomposition of R_z
    #eigenvalues, eigenvectors = LA.eig(R_z)  # eigenvalues size is F,M, eigenvector size is F,M,M

    a_CW = np.zeros((M, F))

    for k in range(F):
         # Eigenvalue decomposition of R_z
        eigenvalues, eigenvectors = LA.eigh(R_z[k])  # eigenvalues size is M, eigenvector size is M,M
        i_max_k = np.argmax(eigenvalues)
        phi_k = eigenvectors[:,i_max_k]  # Eigenvector corresponding to the maximum eigenvalue
        
        temp_k = np.matmul(Rn1_2[k].conj().T, phi_k)
        #temp2 = np.dot(Rn1_2[k].conj().T[0, :], phi_k)
        a_CW[:,k] = temp_k/temp_k[0] # RTF estimator


    return Rnn, a_CW, Ryy, R_z

# This is the correct one:
#-----------------------------------------------
def RTF_Cov_W(Rnn, Ryy, M, F, L):

    Rn1_2 = np.zeros_like(Rnn)
    invRn1_2 = np.zeros_like(Rnn)
    Rz= np.zeros_like(Rnn)
    
    a = np.zeros((M, F))

    for k in range(F):
        eigenvalues, eigenvectors = np.linalg.eig(Rnn[k])  
        Rn1_2[k] = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
        inv_sqrt_eigenvalues = np.diag(1 / np.sqrt(eigenvalues))
        invRn1_2[k] = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.conj().T

        Rz[k] = invRn1_2[k] @ Ryy[k] @ invRn1_2[k].conj().T
    
     
        # Eigenvalue decomposition of R_z
        eigenvalues, eigenvectors = LA.eigh(Rz[k])  # eigenvalues size is M, eigenvector size is M,M
        i_max_k = np.argmax(eigenvalues)
        phi_k = eigenvectors[:,i_max_k]  # Eigenvector corresponding to the maximum eigenvalue
        
        temp_k = np.matmul(Rn1_2[k].conj().T, phi_k)
        a[:,k] = temp_k/temp_k[0] # RTF estimator

    
    
    return a

#-------------------------------------------
def noise_estimation(noise, F, M, Ln):
    
    Rnn = np.zeros((F, M, M))
    for k in range(F):
        n_k_sliced = noise[:,k,:Ln]
        Rnn[k] = np.matmul(n_k_sliced, n_k_sliced.conj().T)/Ln

    return Rnn


#-------------------------------------------
def mix_estimation(Y, F, M, L):
    Ln=58 #int(0.2*L) # Delete
    Ryy = np.zeros((F, M, M))
    for k in range(F):
        Y_k_sliced = Y[:,k,Ln:L] # Change to L
        Ryy[k] = np.matmul(Y_k_sliced, Y_k_sliced.conj().T)/(L-Ln) # Change to L

    return Ryy


def covariance_whitening(Y, F, M, L):
    Ln = 58 #int(0.2*L)
    Rnn = noise_estimation(Y, F, M, Ln)
    Ryy = mix_estimation(Y, F, M, L)

    a_cw = RTF_Cov_W(Rnn, Ryy, M, F, L)

    return a_cw