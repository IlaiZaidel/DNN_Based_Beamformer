import torch
import torch.nn as nn
from UnetModel import UNET, UNETDualInput, UNETDualInput_Two_Speakers
from Beamformer import beamformingOperationStage1, MaskOperationStage2
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from utils import beamformingOpreation, Postprocessing, Preprocesing, return_as_complex, snr, true_rtf_from_all_rirs_batch, true_rtf_from_rirs_bmk
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
        # self.unet_multiChannel_left  = UNET(modelParams.channelsStage1,modelParams.activationSatge1,
        #                                modelParams.EnableSkipAttention, use_rtf=True)
        # self.unet_multiChannel_right  = UNET(modelParams.channelsStage1,modelParams.activationSatge1,
        #                         modelParams.EnableSkipAttention, use_rtf=True)
        
        self.unet_multiChannel_left  = UNETDualInput_Two_Speakers(
            rtf_in_ch=modelParams.channelsStage1,    # 8
            mix_in_ch=modelParams.channelsStage1,    # 8
            out_channels=modelParams.channelsStage1, # 8
            activation=modelParams.activationSatge1,
            EnableSkipAttention=modelParams.EnableSkipAttention,
            stem_each=8  # you can set 32 if you want a slightly richer stem
        )

        # self.unet_multiChannel_right = UNETDualInput(
        #     rtf_in_ch=modelParams.channelsStage1,
        #     mix_in_ch=modelParams.channelsStage1,
        #     out_channels=modelParams.channelsStage1,
        #     activation=modelParams.activationSatge1,
        #     EnableSkipAttention=modelParams.EnableSkipAttention,
        #     stem_each= 8
        # )
        # Ilai 2/8/25
        # Delete this
        # self.unet_singleChannel = UNET(modelParams.channelsStage2,modelParams.activationSatge2,
        #                                modelParams.EnableSkipAttention, use_rtf=False)
        # Delete
        self.modelParams = modelParams
        
    
    def forward(self, Y,rir_first, rir_second,  device, mode="test"):

       # Y.shape - torch.Size([16, 8, 514, 497])   
        noise_only_time = self.modelParams.noise_only_time
        mic_ref_left = self.modelParams.mic_ref_left
        mic_ref_right = self.modelParams.mic_ref_right
        mic_REF = self.modelParams.mic_ref
        DUAL_MODEL = self.modelParams.DUAL_MODEL
        #RTF Etimation of directional noise
        beta_past = self.modelParams.beta_past
        Y_stft =  return_as_complex(Y) #torch.Size([16, 8, 257, 497])
        R = 128
        win_len = 512
        fs = 16000
        B, M,F, T = Y_stft.shape

        Ln = int(noise_only_time * fs // R)
        real_rtf_left  = torch.zeros((B, M, 2*F, T), dtype=Y.dtype, device=Y_stft.device)
        real_rtf_right = torch.zeros((B, M, 2*F, T), dtype=Y.dtype, device=Y_stft.device)
        
        USE_TRUE_RIRS = self.modelParams.use_true_rirs

        B_full, M, F_pos, T_full = Y_stft.shape  # e.g., (16, 8, 257, 497)
        T_eff = T_full - Ln                      # 497 - Ln
        F = F_pos                                # 257
        F_fft = win_len                          # 512

### SEE GiTHUB if you to implement it for time varying RIRS

        if(USE_TRUE_RIRS):
            all_RTFs_left =  true_rtf_from_rirs_bmk(rir_first, win_len=win_len, ref_mic=mic_REF) #torch.Size([8, 8, 512])
            all_RTFs_right =  true_rtf_from_rirs_bmk(rir_second, win_len=win_len, ref_mic=mic_REF)
            # all_RTFs_left =  true_rtf_from_all_rirs_batch(rir_first, win_len, mic_ref_left) #all_RTFs_left.shape torch.Size([8, 8, 512, 434])
            # all_RTFs_right =  true_rtf_from_all_rirs_batch(rir_second, win_len, mic_ref_right)
            #Stage 1: Multi-Channel speech enhancement
            true_RTF_c_left = all_RTFs_left[:, :, :257]
            true_RTF_c_right = all_RTFs_right[:, :, :257]
            rtf_feat_left = torch.cat(
                [true_RTF_c_left.real, true_RTF_c_left.imag],
                dim=2
            )
            rtf_feat_right = torch.cat(
                [true_RTF_c_right.real, true_RTF_c_right.imag],
                dim=2
            )

            real_rtf_left_per_batch = rtf_feat_left.unsqueeze(-1).expand(-1, -1, 2*F, T_eff)
            real_rtf_right_per_batch = rtf_feat_right.unsqueeze(-1).expand(-1, -1, 2*F, T_eff)

            # assume Fpos = 257
            Fpos = true_RTF_c_left.shape[2]
            # L = true_RTF_c_left.shape[-1]
            zero_tail  = torch.zeros((B, M,F, 1), device=Y_stft.device)
            # make zero padding (same as for a_hat_left)
            # zeros_pad = torch.zeros_like(true_RTF_c_left[:, :, :, :L//10])  # example: 10% padding, or match whatever your code does
            zeros_pad = torch.zeros((B, M,2*F, Ln), device=Y_stft.device)
            # zero_tail  = torch.zeros((B, M,F, 1), device=Y_stft.device)
            # Concatenate zeros on time axis (dim=-1)
            true_rtf_left  = torch.cat([zeros_pad, real_rtf_left_per_batch],  dim=-1)   # (B, M, Fpos, L_pad)
            true_rtf_right = torch.cat([zeros_pad,real_rtf_right_per_batch], dim=-1)

            # Reorder to (B, Fpos, M, L)
            # true_rtf_left  = true_rtf_left.permute(0, 2, 1, 3)
            # true_rtf_right = true_rtf_right.permute(0, 2, 1, 3)

            # Concatenate real and imaginary parts along frequency dimension
            # real_rtf_first = torch.cat([true_rtf_left.real,  true_rtf_left.imag],  dim=1).permute(0,2,1,3)  # (B, 2*Fpos, M, L)
            # real_rtf_second = torch.cat([true_rtf_right.real, true_rtf_right.imag], dim=1).permute(0,2,1,3) 


        # elif(DUAL_MODEL):
        #     W, eigvals, eigvecs = pastd_rank1_whitened(Y_stft, noise_only_time, beta_past)
        #     # ----- RTF left reference microphone ----- #
        #     a_hat_left = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time, mic_ref_left)
        #     B, F, M, T = a_hat_left.shape
        #     L = T + Ln
        #     zeros_pad = torch.zeros((B, F, M, Ln), dtype=a_hat_left.dtype, device=a_hat_left.device)

        #     # Concatenate zeros on the left (along time axis = dim=-1)
        #     rtf_tracking_left = torch.cat([zeros_pad, a_hat_left], dim=-1).permute(0,2,1,3)  # (B, F, M, L)
        #     real_rtf_left = torch.cat([rtf_tracking_left.real, rtf_tracking_left.imag], dim=2) #torch.Size([16, 8, 514, L])

        #     # ----- RTF right reference microphone ----- #
        #     a_hat_right = rtf_from_subspace_tracking(W, eigvals, eigvecs, noise_only_time, mic_ref_right)

        #     zeros_pad = torch.zeros((B, F, M, Ln), dtype=a_hat_right.dtype, device=a_hat_right.device)

        #     # Concatenate zeros on the left (along time axis = dim=-1)
        #     rtf_tracking_right = torch.cat([zeros_pad, a_hat_right], dim=-1)  # (B, F, M, L)
        #     rtf_tracking_right = rtf_tracking_right.permute(0, 2, 1, 3)
        #     real_rtf_right = torch.cat([rtf_tracking_right.real, rtf_tracking_right.imag], dim=2) #torch.Size([16, 8, 514, L])


        

        # rtf_real_left_tracking, rtf_real_right_tracking
        W_timeChange_left,skip_Stage1 = self.unet_multiChannel_left(Y.float(),true_rtf_left,true_rtf_right, DUAL_MODEL) # W changes over time # W_timeChange.shape - > torch.Size([16, 8, 514, 497])

        # W_timeChange_right,skip_Stage1 = self.unet_multiChannel_right(Y.float(),real_rtf_right, DUAL_MODEL) # W changes over time # W_timeChange.shape - > torch.Size([16, 8, 514, 497])

        # Ilai, 30/07/2025
        # This needs to change
        # Apply mean to W_timeChange for getting timeFxed weights 
        # 
        W_timeFixed = torch.mean(W_timeChange_left,3)  
        
        # Force frequencies 0 and pi to be real 
        # W_timeFixed[:,:,1]  = torch.zeros_like(W_timeFixed[:,:,1])
        # W_timeFixed[:,:,-1] = torch.zeros_like(W_timeFixed[:,:,-1])
        Y_copy = Y
        # Beamforming Operation 
        # X_hat_Stage1_C,X_hat_satge1,Y,W_Stage1 = beamformingOperationStage1(Y, W_timeFixed)     
        X_hat_Stage1_C_left ,X_hat_satge1_left,Y_copy,W_Stage1_left = beamformingOperationStage1(Y_copy, W_timeFixed)   
        # X_hat_Stage1_C_right ,X_hat_satge1_right, Y,W_Stage1_right = beamformingOperationStage1(Y, W_timeChange_right)   
        # rtf_zeros = torch.zeros_like(rtf_real)
        # Stage 2: Single-Channel speech enhancement
        # IlaiZ 02/08 Dont need
        # W_Stage2,skip_Stage2 = self.unet_singleChannel(X_hat_satge1.float())
        
        # # Mask Operation
        # X_hat_Stage2_C,_,_ = MaskOperationStage2(X_hat_Stage1_C, W_Stage2)       
        
        # return W_timeChange,X_hat_Stage1_C,Y,W_Stage1,X_hat_Stage2_C,W_Stage2,skip_Stage1,skip_Stage2
        B, _, F, L = Y.shape  # Y is shape [B, M, F, L]
        # W_Stage2 = torch.zeros((B, 1, F, L), dtype=W_timeChange_left.dtype, device=W_timeChange_left.device)

        # return W_timeChange,X_hat_Stage1_C,Y,W_Stage1,None,W_Stage2,skip_Stage1,None
    
        return W_Stage1_left,  X_hat_Stage1_C_left, Y