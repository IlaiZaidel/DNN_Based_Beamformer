import torch
import torch.nn as nn
from UnetModel import UNET
from Beamformer import beamformingOperationStage1, MaskOperationStage2
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from utils import beamformingOpreation, Postprocessing, Preprocesing, return_as_complex, snr
from RTF_from_clean_speech import RTF_from_clean_speech
from subspace_tracking import pastd_rank1_whitened, rtf_from_subspace_tracking

class ExNetBFPF(nn.Module):
    """
    ExNet-BF+PF Model
    """
    def __init__(self,modelParams):
        """
        Args:
            modelParams -  Model Parameters configuration, contains:
                activationSatge1    (str): The activation function at the end of stage1.
                activationSatge2    (str): The activation function at the end of stage2.
                channelsStage1      (int): Number of input channels for stage1.
                channelsStage2      (int): Number of input channels for stage2.
                numFeatures         (int): Number of features (number of frames) at the end of stage1.
                EnableSkipAttention (int): Flag to enable attention in the skip connections

        Returns:
            W_timeChange   (torch.Tensor): The estimeted weights that change in time at the end of stage1.
            X_hat_Stage1_C (torch.Tensor): The estimeted signal at the end of the stage1 in the STFT domain.
            Y              (torch.Tensor): The noisy (input) signal in the STFT domain.
            W_Stage1       (torch.Tensor): The estimated weights at the end of stage1.
            X_hat_Stage2_C (torch.Tensor): The estimated signal at the end of the stage2 in the STFT domain.
            W_Stage2       (torch.Tensor): The estimated weights at the end of stage2.
            skip_Stage1    (torch.Tensor): The attention mask of the final layer of stage1.
            skip_Stage2    (torch.Tensor): The attention mask of the final layer of stage2.
        """
        super(ExNetBFPF, self).__init__()
        # Create the first and second stage UNET
        self.unet_multiChannel  = UNET(modelParams.channelsStage1,modelParams.activationSatge1,
                                       modelParams.EnableSkipAttention, use_rtf=True)
        # Ilai 2/8/25
        # Delete this
        # self.unet_singleChannel = UNET(modelParams.channelsStage2,modelParams.activationSatge2,
        #                                modelParams.EnableSkipAttention, use_rtf=False)
        # Delete
        self.modelParams = modelParams
    
    def forward(self, Y,  device):

       # Y.shape - torch.Size([16, 8, 514, 497])   
        noise_only_time = self.modelParams.noise_only_time



        #RTF Etimation of directional noise

        Y_stft =  return_as_complex(Y) #torch.Size([16, 8, 257, 497])
        R = 128
        win_len = 512
        fs = 16000

        Ln = int(noise_only_time * fs // R)
        W, eigvals, eigvecs = pastd_rank1_whitened(Y_stft, noise_only_time, beta=0.995)
        
        a_hat = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time)
        B, F, M, T = a_hat.shape
        L = T + Ln
        #W = W.detach() 
        # Pad zeros: shape (B, F, M, Ln)
        zeros_pad = torch.zeros((B, F, M, Ln), dtype=a_hat.dtype, device=a_hat.device)

        # Concatenate zeros on the left (along time axis = dim=-1)
        rtf_tracking = torch.cat([zeros_pad, a_hat], dim=-1)  # (B, F, M, L)
        rtf_tracking = rtf_tracking.permute(0, 2, 1, 3)
        rtf_real = torch.cat([rtf_tracking.real, rtf_tracking.imag], dim=2) #torch.Size([16, 8, 514, L])

        
        # RTF tracking results:
        '''
        rtf_cw  = covariance_whitening(Y_stft, noise_only_time)
        rtf_cw.shape:              torch.Size([8, 8, 257, 1])
        rtf_tracking.shape:        torch.Size([8, 8, 257, 497])
        rtf_cw[0,:,0,0]:           tensor([ 2.0228+0.j,  1.7768+0.j,  1.3280+0.j,  1.0000+0.j,  0.2264+0.j, -0.4458+0.j, -0.9935+0.j, -1.0482+0.j])
        rtf_tracking[0,:,0,496]:   tensor([ 2.0944+5.3008e-07j,  1.8427+7.1657e-07j,  1.2614+1.5627e-07j, 1.0000-6.6213e-09j,  0.3411-3.6671e-07j, -0.3477-7.5018e-07j,
                                    -0.8158-1.1464e-06j, -0.8811-9.6857e-07j])
        '''



        #Stage 1: Multi-Channel speech enhancement

        W_timeChange,skip_Stage1 = self.unet_multiChannel(Y.float(),rtf_real) # W changes over time # W_timeChange.shape - > torch.Size([16, 8, 514, 497])



        # Ilai, 30/07/2025
        # This needs to change
        # Apply mean to W_timeChange for getting timeFxed weights 
        W_timeFixed = torch.mean(W_timeChange,3)  
        
        # Force frequencies 0 and pi to be real 
        W_timeFixed[:,:,1]  = torch.zeros_like(W_timeFixed[:,:,1])
        W_timeFixed[:,:,-1] = torch.zeros_like(W_timeFixed[:,:,-1])
        
        # Beamforming Operation 
        # X_hat_Stage1_C,X_hat_satge1,Y,W_Stage1 = beamformingOperationStage1(Y, W_timeFixed)     
        X_hat_Stage1_C,X_hat_satge1,Y,W_Stage1 = beamformingOperationStage1(Y, W_timeChange)   

        # rtf_zeros = torch.zeros_like(rtf_real)
        # Stage 2: Single-Channel speech enhancement
        # IlaiZ 02/08 Dont need
        # W_Stage2,skip_Stage2 = self.unet_singleChannel(X_hat_satge1.float())
        
        # # Mask Operation
        # X_hat_Stage2_C,_,_ = MaskOperationStage2(X_hat_Stage1_C, W_Stage2)       
        
        # return W_timeChange,X_hat_Stage1_C,Y,W_Stage1,X_hat_Stage2_C,W_Stage2,skip_Stage1,skip_Stage2
        B, _, F, L = Y.shape  # Y is shape [B, M, F, L]
        W_Stage2 = torch.zeros((B, 1, F, L), dtype=W_timeChange.dtype, device=W_timeChange.device)

        return W_timeChange,X_hat_Stage1_C,Y,W_Stage1,None,W_Stage2,skip_Stage1,None