import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation, return_as_complex
from ComputeLoss import Loss
from saveResults import saveResults
import criterionFile
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from loss_function import compute_loss
def test(model, args, results_path, test_loader, device, cfg_loss, iftest):

    # Parameters
    fs = args.fs
    win_len = args.win_length
    T = args.T
    R = eval(args.R)
    mic_ref = args.mic_ref
 
    # Init Variables 
    epoch_test_loss = 0 
        
    model.eval()   
       
    with torch.no_grad():
        for i, (y, labels_x, fullnoise_first,white_noise, all_rirs) in enumerate(test_loader):
            # Extract Data
            y = y.to(device)                    # y = B,T*fs,M - noisy signal in the time domain
            fullLabels_x = labels_x.to(device)  # x = B,T*fs,M - target signal in the time domain
            labels_x = torch.unsqueeze(fullLabels_x[:,:,mic_ref-1],2) # x_ref - B,T*fs,1 - target signal ref in the time domain  
            fullnoise_second =  torch.tensor(0.0, device=device)
            # Perform STFT and Padding
            Y = Preprocesing(y, win_len, fs, T, R, device)  # Y = B,M,2*F,L - noisy signal in the STFT domain

            ##########################################
            # RTF calculation, maybe delete 
            noise_only_time = args.noise_only_time
            # RTF Etimation:
            Y_complex =  return_as_complex(Y) #torch.Size([16, 8, 257, 497])

            # with torch.no_grad():
            #     rtf_cw = covariance_whitening(Y_complex, noise_only_time)
            #     rtf_cw = fix_covariance_whitening(rtf_cw)
            #     rtf_real = torch.cat([rtf_cw.real, rtf_cw.imag], dim=2)
            # ##########################################
            
            # Forward
            # W_timeChange,X_hat_Stage1,Y,W_Stage1,X_hat_Stage2,W_Stage2,skip_Stage1,skip_Stage2 = model(Y, device)
            W_Stage1_left, W_Stage1_right, X_hat_Stage1_C_left, X_hat_Stage1_C_right, Y = model(Y,all_rirs, device, mode="test")
            # W_Stage2 is shape torch.Size([8, 1, 514, 497])
            # B,M,F,L = W_Stage2.size()
            # W_Stage2 = W_Stage2.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()
            # W_Stage2= torch.view_as_complex(W_Stage2) # torch.Size([8, 1, 257, 497])
            # # Squeeze the singleton dimension (M = 1) to align with noise_stage1
            # W_Stage2 = W_Stage2.squeeze(1)  # Now W_Stage2 has shape [8, 257, 497]


            # Perform ISTFT and norm for x_hat before PF
            x_hat_stage1_B_norm_left = Postprocessing(X_hat_Stage1_C_left,R,win_len,device)
            max_x_left = torch.max(abs(x_hat_stage1_B_norm_left),dim=1).values
            x_hat_stage1_left = (x_hat_stage1_B_norm_left.T/max_x_left).T

            x_hat_stage1_B_norm_right = Postprocessing(X_hat_Stage1_C_right,R,win_len,device)
            max_x_right = torch.max(abs(x_hat_stage1_B_norm_right),dim=1).values
            x_hat_stage1_right = (x_hat_stage1_B_norm_right.T/max_x_right).T


        #     # Perform ISTFT and norm for x_hat
        #     x_hat_stage2_B_norm = Postprocessing(X_hat_Stage2,R,win_len,device)
        #   #  max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
        #     x_hat_stage2_time = x_hat_stage2_B_norm #(x_hat_stage2_B_norm.T/max_x).T            

            # Preprocessing & Postprocessing for the labeled signal
            X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device) 
            X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref)
            x = Postprocessing(X_stft_mic_ref,R,win_len,device)
            max_x = torch.max(abs(x),dim=1).values
           # x = (x.T/max_x).T

            # Calculate the loss function 
            #loss = Loss(x,x_hat_stage2,cfg_loss)*args.Enable_cost_mae        
            #loss,loss_mae, cost_distortionless, cost_minimum_variance_dir = compute_loss(x, X_stft, Y, x_hat_stage2, W_Stage1, W_Stage2, fullLabels_x, fullnoise, win_len, fs, T, R, device, cfg_loss, args)
            X_stft = return_as_complex(X_stft) #torch.Size([8, 8, 257, 497])
            loss, loss_L1, cost_distortionless, cost_minimum_variance_dir, \
            cost_minimum_variance_white, SNR_output, si_sdr_loss_left,si_sdr_loss_right, \
            cost_minimum_variance_two, loss_W_L1,_,_ = compute_loss(
                x, X_stft, Y, X_hat_Stage1_C_left, 
                W_Stage1_left,X_hat_Stage1_C_right,W_Stage1_right ,fullLabels_x, fullnoise_first,
                fullnoise_second, white_noise, win_len, fs, T, R,
                device, cfg_loss, args
            )
            # Backward & Optimize     
            epoch_test_loss += loss.item() 

            # Normalizing: 

            x = (x.T/max_x).T
            # max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
            # x_hat_stage2 = (x_hat_stage2_B_norm.T/max_x).T            
            # x_hat_stage2_time = torch.zeros_like(x, device=device)
            # X_hat_Stage2= torch.zeros_like(X_hat_Stage1, device=device)
            # Save results
            if iftest == 1:
                saveResults(Y,X_stft,W_Stage1_left, W_Stage1_right,X_hat_Stage1_C_left,X_hat_Stage1_C_right,
                            y,x_hat_stage1_left, x_hat_stage1_right,results_path,i,fs)           

    return epoch_test_loss