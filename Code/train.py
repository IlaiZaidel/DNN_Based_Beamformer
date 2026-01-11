import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation,return_as_complex, true_rtf_from_rirs_bmk
from test import test
from ComputeLoss import Loss
import criterionFile
from tqdm import tqdm
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from loss_function import compute_loss
import wandb
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
#from covariance_whitening_backup import covariance_whitening, fix_covariance_whitening
def train(model, args, results_path, train_loader, val_loader, optimizer, device, cfg_loss, debug, rho: float, lam_pass: torch.Tensor, lam_null: torch.Tensor, epoch):
     
    # Parameters
    fs = args.fs
    win_len = args.win_length
    T = args.T
    R = eval(args.R)
    mic_ref = args.mic_ref
    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    # Init Variables 
    epoch_train_loss, epoch_val_loss = 0 ,0
    noise_only_time = args.noise_only_time

    model.train()
    # y, d, d1, d2, babble, v, all_rirs
    #for i, (y, labels_x, fullnoise) in tqdm(enumerate(train_loader)): # on batch # 20.3 - I need to change it back to one noise only
    for i, (y, labels_x, first_speaker, second_speaker, babble, white_noise, rir_first, rir_second) in tqdm(enumerate(train_loader),total=len(train_loader)): # on batch
        # Extract Data
        y = y.to(device)                    # y = B,T*fs,M - noisy signal in the time domain
        fullLabels_x = labels_x.to(device)  # x = B,T*fs,M - target signal in the time domain
        first_speaker = first_speaker.to(device)
        second_speaker = second_speaker.to(device)
        labels_x = torch.unsqueeze(fullLabels_x[:,:,mic_ref-1],2) # x_ref - B,T*fs,1 - target signal ref in the time domain  
        
        

        # Perform STFT and Padding
        Y = Preprocesing(y, win_len, fs, T, R, device)                  # Y = B,M,2*F,L - noisy signal in the STFT domain, torch.Size([8, 8, 514, 497])
        FIRST_SPEAKER = Preprocesing(first_speaker, win_len, fs, T, R, device)   
        SECOND_SPEAKER = Preprocesing(second_speaker, win_len, fs, T, R, device)   
        
        Y_complex =  return_as_complex(Y) #torch.Size([16, 8, 257, 497])
        FIRST_SPEAKER_stft =  return_as_complex(FIRST_SPEAKER)
        SECOND_SPEAKER_stft =  return_as_complex(SECOND_SPEAKER)
        
        
        BABBLE =  Preprocesing(babble.to(device)  , win_len, fs, T, R, device)   
        BABBLE_stft =  return_as_complex(BABBLE)
        # optimizer.zero_grad()
        # Ln = int(noise_only_time * fs // R)


        # B, M, F_pos, T_full = Y_complex.shape  # e.g., (16, 8, 257, 497)
        # T_eff = T_full - Ln                      # 497 - Ln
        # F = F_pos                                # 257
        # F_fft = win_len                          # 512
        all_RTFs_first =  true_rtf_from_rirs_bmk(rir_first, win_len=win_len, ref_mic=mic_ref).to(device)     #torch.Size([8, 8, 512])
        all_RTFs_second=  true_rtf_from_rirs_bmk(rir_second, win_len=win_len, ref_mic=mic_ref).to(device)    
        # # all_RTFs_left =  true_rtf_from_all_rirs_batch(rir_first, win_len, mic_ref_left) #all_RTFs_left.shape torch.Size([8, 8, 512, 434])
        # # all_RTFs_right =  true_rtf_from_all_rirs_batch(rir_second, win_len, mic_ref_right)
        # #Stage 1: Multi-Channel speech enhancement
        true_RTF_c_left = all_RTFs_first[:, :, :257]
        true_RTF_c_right = all_RTFs_second[:, :, :257]
        # # rtf_feat_left = torch.cat(
        # #     [true_RTF_c_left.real, true_RTF_c_left.imag],
        # #     dim=2
        # # )
        # # rtf_feat_right = torch.cat(
        # #     [true_RTF_c_right.real, true_RTF_c_right.imag],
        # #     dim=2
        # # )

        # real_rtf_left_per_batch = true_RTF_c_left.unsqueeze(-1).expand(-1, -1, -1, T_eff)
        # real_rtf_right_per_batch = true_RTF_c_right.unsqueeze(-1).expand(-1, -1, -1, T_eff)


        # # make zero padding (same as for a_hat_left)
        # # zeros_pad = torch.zeros_like(true_RTF_c_left[:, :, :, :L//10])  # example: 10% padding, or match whatever your code does
        # zeros_pad = torch.zeros((B, M,F, Ln), device=Y_complex.device)
        # # zero_tail  = torch.zeros((B, M,F, 1), device=Y_stft.device)
        # # Concatenate zeros on time axis (dim=-1)
        # true_rtf_first  = torch.cat([zeros_pad, real_rtf_left_per_batch],  dim=-1)   # (B, M, Fpos, L_pad)
        # true_rtf_second = torch.cat([zeros_pad,real_rtf_right_per_batch], dim=-1)
        
        loss = torch.tensor(0.0, device=device)

        ##########################################
        ####### Model
        ##########################################
        W_Stage1_left,  X_hat_Stage1_C_left, Y = model(Y,rir_first, rir_second , device, mode="train")


        FIRST_SPEAKER_output,_,_ =  beamformingOpreation(FIRST_SPEAKER_stft,mic_ref, W_Stage1_left) # No beamformer W in the function, it takes only signal as recorded/
        first_speaker_output_time = Postprocessing(FIRST_SPEAKER_output,R,win_len,device)
        

        SECOND_SPEAKER_output,_,_ =  beamformingOpreation(SECOND_SPEAKER_stft,mic_ref, W_Stage1_left) # No beamformer W in the function, it takes only signal as recorded/
        second_speaker_output_time = Postprocessing(SECOND_SPEAKER_output,R,win_len,device)

        BABBLE_output,_,_ =  beamformingOpreation(BABBLE_stft,mic_ref, W_Stage1_left) # No beamformer W in the function, it takes only signal as recorded/
        babble_output_time = Postprocessing(BABBLE_output,R,win_len,device)

        
        # # Preprocessing & Postprocessing for the labeled signal
        # X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])
        # X_stft = return_as_complex(X_stft) #torch.Size([8, 8, 257, 497])
        
        
        # X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref) # No beamformer W in the function, it takes only signal as recorded/
        # x = Postprocessing(X_stft_mic_ref,R,win_len,device) # speech as recorded in reference microphone
        # max_x = torch.max(abs(x),dim=1).values
        # # x = (x.T/max_x).T

        FIRST_RTF_output =  torch.sum( torch.mul(torch.conj(W_Stage1_left[:,:,:,0]), true_RTF_c_left), dim=1) # X_hat = B,M,F,L = B,8,257,497
        SECOND_RTF_output =  torch.sum( torch.mul(torch.conj(W_Stage1_left[:,:,:,0]), true_RTF_c_right), dim=1) 
        
        ####### CONSTRAINT DEFINITIONS #######

        

        def phi_quadlog(t: torch.Tensor) -> torch.Tensor:
            # Boolean mask for quadratic region t >= -1/2
            quad_region = t >= -0.5   # bool tensor

            # Quadratic branch (used when quad_region is True)
            quad = 0.5 * t**2 + t

            # Log branch (used when quad_region is False)
            # Clamp to keep -2*t strictly positive and avoid log(0)
            log_input = torch.clamp(-2 * t, min=1e-12)
            log_part = -0.25 * torch.log(log_input) - 3/8

            # Elementwise branch selection
            return torch.where(quad_region, quad, log_part)


        # no_AL = epoch >= 2
        # no_null = epoch  >= 4

         ##### Constraints - Time domain       
        r_pass = (si_sdr(first_speaker_output_time, first_speaker[:,:,mic_ref-1]) - 15)/10 #(first_speaker_output_time - first_speaker[:,:,mic_ref-1]).pow(2).mean()     #si_sdr(first_speaker_output_time, first_speaker[:,:,mic_ref-1]) - 15
                            # want 0
        margin_db = -30.0
        eps = 1e-8
        r_null = (second_speaker_output_time.pow(2).mean()) # torch.relu(10 * torch.log10(second_speaker_output_time.pow(2).mean() + eps) - margin_db        )
        # (second_speaker_output_time.pow(2).mean())   # want 0 # si_sdr(second_speaker_output_time, torch.zeros_like(second_speaker_output_time, device=second_speaker_output_time.device)) - 15   
                
        
        # constraints (complex STFT)
        # S1_ref = FIRST_SPEAKER_stft[:, args.mic_ref, :, :]  #torch.Size([8, 257, 497])  # [B,F,L]
        # r_pass = FIRST_SPEAKER_output - S1_ref                   # want 0
        # r_null = SECOND_SPEAKER_output                           # want 0
        # S1_ref = FIRST_SPEAKER_stft[:, args.mic_ref, :, :]  #torch.Size([8, 257, 497])  # [B,F,L]
        # r_pass = FIRST_RTF_output - torch.ones_like(FIRST_RTF_output,  device=Y_complex.device)                   # want 0
        # r_null = SECOND_RTF_output                           # want 0
        # choose f(θ) (example: maximize SI-SDR of mixture output)

        if args.beta_SISDR> 0:

            X_hat_Stage1_left_time = Postprocessing(X_hat_Stage1_C_left,R,win_len,device) #torch.Size([16, 64000])

            si_sdr_loss_first= si_sdr(first_speaker[:,:,args.mic_ref-1],X_hat_Stage1_left_time)
            # second_speaker= Postprocessing(SECOND_SPEAKER_stft[:, args.mic_ref_left,:,:],R,win_len,device) #X_stft.shape is torch.Size([16, 8, 257, 497])
            si_sdr_loss_second = si_sdr(second_speaker[:,:,args.mic_ref-1],X_hat_Stage1_left_time)

            penalty_null = args.beta_SISDR * phi_quadlog(si_sdr_loss_second + 15)

        X_hat_time = Postprocessing(X_hat_Stage1_C_left, R, win_len, device) #torch.Size([8, 64000])
        f_theta = -si_sdr(first_speaker[:, :, args.mic_ref-1], X_hat_time)  #babble_output_time.pow(2).mean()   #(X_hat_time - first_speaker[:,:,mic_ref-1]).pow(2).mean()    #-si_sdr(first_speaker[:, :, args.mic_ref-1], X_hat_time)  # minimize negative

        # lin_pass = torch.real((lam_pass.conj() * r_pass))
        # lin_null = torch.real((lam_null * r_null))

        # pen_pass =  0.5 * rho * r_pass.abs().pow(2) 
        # pen_null = 0.5 * rho * r_null.abs().pow(2) 
        # # AL terms
        # lin = torch.real((lam_pass.conj() * r_pass)) + torch.real((lam_null * r_null))
        # pen = 0.5 * rho * (r_pass.abs().pow(2) + r_null.abs().pow(2))

        
        # if no_AL==0 :
        #     loss = f_theta + penalty_null
        # else:
        #     if no_null == 0:
        #         loss = f_theta + lin_pass + pen_pass + penalty_null
        #     else:
        #         loss = f_theta + (lin + pen) + penalty_null

        r_pass = torch.mean(torch.sum(torch.abs(FIRST_RTF_output - 1.0)**2, dim = 1))
        r_null = torch.mean(torch.sum(torch.abs(SECOND_RTF_output)**2, dim = 1))


        # want r <= delta
        delta_pass = 1e-3
        delta_null = 1e-3

        g_pass = r_pass - delta_pass   # want <= 0
        g_null = r_null - delta_null   # want <= 0
        no_AL = epoch >= 0
        no_null = epoch  >= 2
        pen = 0.5 * (no_AL*torch.relu(g_pass) + no_null*torch.relu(g_null))

        loss =  pen



        # loss = f_theta + no_AL*phi_quadlog(r_pass-0.1) + no_null*phi_quadlog(r_null-0.1)


        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()










        # if args.beta_SISDR> 0:

        #     X_hat_Stage1_left_time = Postprocessing(X_hat_Stage1_C_left,R,win_len,device) #torch.Size([16, 64000])
        #     first_speaker_2= Postprocessing(FIRST_SPEAKER_stft[:, args.mic_ref-1,:,:],R,win_len,device) #X_stft.shape is torch.Size([16, 8, 257, 497])

        #     si_sdr_loss_first_noisy = si_sdr(first_speaker[:,:,args.mic_ref-1],X_hat_Stage1_left_time)
        #     si_sdr_loss_first= si_sdr(first_speaker[:,:,args.mic_ref-1],X_hat_Stage1_left_time)
        #     # second_speaker= Postprocessing(SECOND_SPEAKER_stft[:, args.mic_ref_left,:,:],R,win_len,device) #X_stft.shape is torch.Size([16, 8, 257, 497])
        #     si_sdr_loss_second = si_sdr(second_speaker[:,:,args.mic_ref-1],X_hat_Stage1_left_time)

            # penalty_null = args.beta_SISDR * phi_quadlog(si_sdr_loss_second + 15)
            # penalty_pass = args.beta_SISDR * phi_quadlog(-si_sdr_loss_first + 5)
            # total_SI_SDR = -50*si_sdr_loss_first_noisy + penalty_pass + penalty_null # -si_sdr_loss_first + penalty #  si_sdr(first_speaker[:,:,args.mic_ref_left] + second_speaker[:,:,args.mic_ref_left],X_hat_Stage1_left_time) #

            # loss += total_SI_SDR



        # loss, loss_L1, cost_distortionless, cost_minimum_variance_dir, \
        # cost_minimum_variance_white, SNR_output, si_sdr_loss_left,si_sdr_loss_right, \
        # cost_minimum_variance_two, loss_W_L1,  loss_L2_stft_left,loss_L2_stft_right  = compute_loss(
        #     x, X_stft, Y, X_hat_Stage1_C_left, 
        #     W_Stage1_left,X_hat_Stage1_C_right,W_Stage1_right ,fullLabels_x, fullnoise_first,
        #     fullnoise_second, white_noise, win_len, fs, T, R,
        #     device, cfg_loss, args
        # )
        # W_Stage1_left, W_Stage1_right, X_hat_Stage1_C_left, X_hat_Stage1_C_right
        # Log metrics to wandb
        wandb.log({
            "batch_loss_train": loss.item(),
            "f(X)": f_theta.item(),
            # "lin": lin.item(),
            "penalty": pen.item(),
            # "loss_L1": loss_L1.item(),
            # "cost_distortionless": cost_distortionless.item(),
            # "cost_minimum_variance_dir": cost_minimum_variance_dir.item(),
            # "cost_minimum_variance_white": cost_minimum_variance_white.item(),
            # "SNR_output": SNR_output.item(),
            "SI-SDR_FIRST": si_sdr_loss_first.item(),
            "SI-SDR_SECOND": si_sdr_loss_second.item(),
            # "cost_minimum_variance_two" : cost_minimum_variance_two.item(),
            # "loss_W_L1" : loss_W_L1.item(),
            # "loss_L2_stft_left" : loss_L2_stft_left.item(),
            # "loss_L2_stft_right" : loss_L2_stft_right.item()
            "r_pass_rms": float(r_pass.abs().item()),
            "r_null_rms": float(r_null.abs().item()),
            # "lam_pass_rms": float(lam_pass.abs().item()),
            # "lam_null_rms": float(lam_null.abs().item()),
        })

        # ---------------
        # Backward & Optimize
        loss.backward()
        if debug:
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name, param.grad)
        optimizer.step()        
        epoch_train_loss += loss.item() 

        # with torch.no_grad():

        #     lam_pass += rho * r_pass.detach()  # [F,L]
        #     lam_null += no_AL*rho * r_null.detach()

        # with torch.no_grad():
        #     if no_AL:          # epoch >= 2
        #         lam_pass += rho * r_pass.detach()

        #         if no_null:    # epoch >= 4  (your naming is reversed; see note below)
        #             lam_null += rho * r_null.detach()

                    
    # VAL    
    epoch_val_loss = test(model, args, results_path, val_loader, device, cfg_loss, 0)
   
    return epoch_train_loss,epoch_val_loss,  lam_pass, lam_null