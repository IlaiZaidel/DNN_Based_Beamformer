import torch
from utils import beamformingOpreation, Postprocessing, Preprocesing, return_as_complex, snr
from ComputeLoss import Loss
import criterionFile
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
import torch.nn as nn
from RTF_from_clean_speech import RTF_from_clean_speech
from torch import tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


def compute_loss(x, X_stft, Y, X_hat_Stage1, x_hat_stage2, W_Stage1, W_Stage2, fullLabels_x, fullnoise_first, fullnoise_second, win_len, fs, T, R, device, cfg_loss, args):
    """
    Compute the loss for the given inputs based on the provided arguments.

    Args:
        x: Labeled target signal in the time domain (normalized).
        X_stft: STFT of the labeled target signal.
        Y: Noisy signal in the STFT domain.
        W_Stage1: Beamformer weights from stage 1.
        W_Stage2: Beamformer weights from stage 2.
        fullLabels_x: Target signal in the time domain (batch, time, channels).
        fullnoise: Directional noise signal in the time domain (batch, time, channels).
        win_len: Window length for STFT.
        fs: Sampling frequency.
        T: Frame duration.
        R: Hop size.
        device: Torch device (CPU or GPU).
        cfg_loss: Configuration for the loss function.
        args: Additional arguments with settings for enabling loss terms.

    Returns:
        total_loss: Computed total loss value.
    """
    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    total_loss = 0
    cost_distortionless = torch.tensor(0.0)
    cost_minimum_variance_white = torch.tensor(0.0)
    cost_minimum_variance_dir = torch.tensor(0.0)
    SNR_output = torch.tensor(0.0)
    si_sdr_loss = torch.tensor(0.0)
    cost_minimum_variance_two = torch.tensor(0.0)
    # Base loss (e.g., MAE between x and x_hat_stage2)
    loss_L1 = Loss(x, x_hat_stage2, cfg_loss)* args.Enable_cost_L1 # Value example: 764.1401
    total_loss += loss_L1


    # L2 Norm

    criterion_L2 = criterionFile.criterionL2

    # Calculate L1 loss
    loss_L2 = criterion_L2(x, x_hat_stage2, cfg_loss.norm)* args.Enable_cost_L2
    total_loss += loss_L2


    # Cost function (beamforming-based regularization)
    if args.EnableCost:
        WX, _, _ = beamformingOpreation(X_stft, args.mic_ref, W_Stage1)
        wx = Postprocessing(WX, R, win_len, device)
        max_wx = torch.max(abs(wx), dim=1).values
        wx = (wx.T / max_wx).T
        
        criterion_L1 = criterionFile.criterionL1
        cost_wx = criterion_L1(wx.float(), x.float(), cfg_loss.norm) * 10000
        total_loss += cost_wx


    # Distortionless response (DR) cost
    if args.beta_dr>0:
        #a = covariance_whitening(Y)
        a = RTF_from_clean_speech(X_stft)
        a = fix_covariance_whitening(a)  # Output: B, M, F, L

        wa = torch.mul(torch.conj(W_Stage1), a)
        wa = torch.sum(wa, dim=1) # torch.Size([8, 257, 1])
        wa_stage2 = torch.mul(torch.conj(W_Stage2), wa) # torch.Size([8, 257, 497])
        # Loss is computed only after stage 1:
        ones = torch.ones_like(wa)
        abs_squared_diff = torch.abs(wa - ones) ** 2 #torch.Size([8, 257, 1])
        sum_per_frame = torch.sum(abs_squared_diff, dim=1) #sum_per_frame.shape
        cost_distortionless = torch.mean(sum_per_frame)*args.beta_dr # Value example: 338.544
        total_loss += cost_distortionless

    # Minimum variance (MV) cost
    if args.beta_mv_dir>0:
        noise_only = fullnoise_first.to(device) # torch.Size([8, 64000, 8])
        noise_stft = Preprocesing(noise_only, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])

        noise_stage1, _, _ = beamformingOpreation(noise_stft, args.mic_ref, W_Stage1) # torch.Size([8, 257, 497])
        #noise_stage2 = torch.mul(torch.conj(W_Stage2), noise_stage1) # torch.Size([8, 257, 497]) (B,F,L)
        # Loss is computed only after stage 1:
        abs_squared_noise = torch.abs(noise_stage1) ** 2
        sum_per_frame = torch.sum(abs_squared_noise, dim=1) # torch.Size([8, 497])
        cost_minimum_variance_dir = torch.mean(sum_per_frame)  #mean ovre time  #Value example 3164
        total_loss += cost_minimum_variance_dir * args.beta_mv_dir

        
    

    if args.beta_mv_white>0:
        dir_noise = fullnoise_first.to(device) # torch.Size([8, 64000, 8]) 
        DIR_noise_stft = Preprocesing(dir_noise, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])
        DIR_noise_stft = return_as_complex(DIR_noise_stft) # torch.Size([8, 8, 257, 497])
        #X_STFT = return_as_complex(X_stft)  # torch.Size([8, 8, 257, 497])
        White_Noise_STFT = Y-X_stft-DIR_noise_stft  # torch.Size([8, 8, 257, 497])
        white_noise_stage1 = torch.mul(torch.conj(W_Stage1), White_Noise_STFT)
        white_noise_stage1 = torch.sum(white_noise_stage1, dim=1) # torch.Size([8, 257, 497])
        #white_noise_stage2 = torch.mul(torch.conj(W_Stage2), white_noise_stage1) # torch.Size([8, 257, 497]) (B,F,L)
        # Loss is computed only after stage1:
        abs_squared_white_noise = torch.abs(white_noise_stage1) ** 2
        sum_per_frame_white = torch.sum(abs_squared_white_noise, dim=1) # torch.Size([8, 497])
        cost_minimum_variance_white = torch.mean(sum_per_frame_white)  #mean ovre time  #Value example 1600
        total_loss += cost_minimum_variance_white * args.beta_mv_white
    
    
    if args.beta_SIR> 0:
        noise_only = fullnoise_first.to(device) # torch.Size([8, 64000, 8])
        noise_stft = Preprocesing(noise_only, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])

        noise_stage1, _, _ = beamformingOpreation(noise_stft, args.mic_ref, W_Stage1) # torch.Size([8, 257, 497])
        #SNR COMPARING FOR DEBUGGING
        Speech_stage1 = torch.mul(torch.conj(W_Stage1), X_stft) # B,M,F,L = B,8,257,497
        Speech_stage1 = torch.sum(Speech_stage1, dim=1)         # B,F,L = B,257,497

        speech_stage1_time = Postprocessing(Speech_stage1,R,win_len,device) #torch.Size([8, 64000])
        noise_stage1_time = Postprocessing(noise_stage1,R,win_len,device) #torch.Size([8, 64000])
        speech = fullLabels_x[:,:,3]

        SNR = snr(speech, noise_only[:,:,3])
        #SNR_noises = snr(noise_only, noise_stage1_time)
        SNR_output = snr(speech_stage1_time, noise_stage1_time)
        #white_noise_time = Postprocessing(White_Noise_STFT[:,3,:,:],R,win_len,device)#torch.Size([8, 64000])
        #SNR_white_noise = snr(speech,white_noise_time)

        #just to see what happens

        total_loss -= SNR_output *args.beta_SIR

# This also works for 2 directional noise
    if args.white_only_enable>0:
        white_noise_stft = Y-X_stft # torch.Size([8, 8, 257, 497])
        white_noise_stage1 = torch.mul(torch.conj(W_Stage1), white_noise_stft) # torch.Size([8, 8, 257, 497])
        white_noise_stage1 = torch.sum(white_noise_stage1, dim=1) # torch.Size([8, 257, 497]) ( B = 8 ) # Beamforming operation
        #white_noise_stage2 = torch.mul(torch.conj(W_Stage2), white_noise_stage1) # torch.Size([8, 257, 497]) (B,F,L)
        # Loss is computed only after stage1:
        abs_squared_white_noise = torch.abs(white_noise_stage1) ** 2 
        sum_per_frame_white = torch.sum(abs_squared_white_noise, dim=1) # torch.Size([8, 497]) # Energy in the frequency domain
        cost_minimum_variance_white = torch.mean(sum_per_frame_white)  #mean ov
        total_loss += cost_minimum_variance_white * args.white_only_enable


# Two directional noises
    if args.two_dir_noise>0:
        two_directional_noises = fullnoise_first.to(device) + fullnoise_second.to(device) # torch.Size([8, 64000, 8])
        two_noises_stft = Preprocesing(two_directional_noises, win_len, fs, T, R, device)#  torch.Size([8, 8, 514, 497]) 
        two_noises_stft =  return_as_complex(two_noises_stft) #torch.Size([8, 8, 257, 497])
        two_directional_noises_stage1 = torch.mul(torch.conj(W_Stage1), two_noises_stft) # torch.Size([8, 8, 257, 497])
        two_directional_noises_stage1 = torch.sum(two_directional_noises_stage1, dim=1) # torch.Size([8, 257, 497]) ( B = 8 ) # Beamforming operation
        #white_noise_stage2 = torch.mul(torch.conj(W_Stage2), white_noise_stage1) # torch.Size([8, 257, 497]) (B,F,L)
        # Loss is computed only after stage1:
        abs_squared_directional = torch.abs(two_directional_noises_stage1) ** 2 
        sum_per_frame_directional = torch.sum(abs_squared_directional, dim=1) # torch.Size([8, 497]) # Energy in the frequency domain
        cost_minimum_variance_two = torch.mean(sum_per_frame_directional)  #mean ov
        total_loss += cost_minimum_variance_two * args.two_dir_noise



    ## SI-SDR
    if args.beta_SISDR> 0:

        x_hat_stage1 = Postprocessing(X_hat_Stage1,R,win_len,device)

        si_sdr_loss = si_sdr(x,x_hat_stage1)

        total_loss -=si_sdr_loss*args.beta_SISDR


    return total_loss,loss_L1, cost_distortionless, cost_minimum_variance_dir, cost_minimum_variance_white, SNR_output,si_sdr_loss, cost_minimum_variance_two
