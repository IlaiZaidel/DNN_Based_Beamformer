import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation,return_as_complex
from test import test
from ComputeLoss import Loss
import criterionFile
from tqdm import tqdm
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from loss_function import compute_loss
import wandb
#from covariance_whitening_backup import covariance_whitening, fix_covariance_whitening
def train(model, args, results_path, train_loader, val_loader, optimizer, device, cfg_loss, debug):
     
    # Parameters
    fs = args.fs
    win_len = args.win_length
    T = args.T
    R = eval(args.R)
    mic_ref = args.mic_ref
    
    # Init Variables 
    epoch_train_loss, epoch_val_loss = 0 ,0
    
    model.train()
    
    #for i, (y, labels_x, fullnoise) in tqdm(enumerate(train_loader)): # on batch # 20.3 - I need to change it back to one noise only
    for i, (y, labels_x, fullnoise_first, white_noise) in tqdm(enumerate(train_loader),total=len(train_loader)): # on batch
        # Extract Data
        y = y.to(device)                    # y = B,T*fs,M - noisy signal in the time domain
        fullLabels_x = labels_x.to(device)  # x = B,T*fs,M - target signal in the time domain  
        labels_x = torch.unsqueeze(fullLabels_x[:,:,mic_ref-1],2) # x_ref - B,T*fs,1 - target signal ref in the time domain  
        
        fullnoise_second = torch.tensor(0.0, device=device)

        # Perform STFT and Padding
        Y = Preprocesing(y, win_len, fs, T, R, device)                  # Y = B,M,2*F,L - noisy signal in the STFT domain, torch.Size([8, 8, 514, 497])
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        ##########################################
        noise_only_time = args.noise_only_time
        # RTF Etimation:
        Y_complex =  return_as_complex(Y) #torch.Size([16, 8, 257, 497])

        with torch.no_grad():
            rtf_cw = covariance_whitening(Y_complex, noise_only_time)
            rtf_cw = fix_covariance_whitening(rtf_cw)
            rtf_real = torch.cat([rtf_cw.real, rtf_cw.imag], dim=2)
        ##########################################
        ##########################################
        # Calculate total number of samples in the noise-only segment
        num_samples = int(noise_only_time * fs)
        # Calculate number of STFT frames
        noise_frames = 1 + (num_samples - win_len) // R
        ##########################################
        ##########################################
        #noise_STFT = 
        ##########################################



        # Forward
        # model = W_timeChange,X_hat_Stage1_C,Y,W_Stage1,X_hat_Stage2_C,W_Stage2,skip_Stage1,skip_Stage2
        W_timeChange,X_hat_Stage1,Y,W_Stage1,X_hat_Stage2,W_Stage2,_,_ = model(Y, device) # Y now has the size of torch.Size([8, 8, 257, 497]) (viewed as complex)

        # W_Stage2 is shape torch.Size([8, 1, 514, 497])
        B,M,F,L = W_Stage2.size()

        # Ilai Z 03/8 - Deleted this
        # W_Stage2 = W_Stage2.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()
        # W_Stage2= torch.view_as_complex(W_Stage2) # torch.Size([8, 1, 257, 497])
        # # Squeeze the singleton dimension (M = 1) to align with noise_stage1
        # W_Stage2 = W_Stage2.squeeze(1)  # Now W_Stage2 has shape [8, 257, 497]


        # X_hat_Stage1 is shape of 8,257,497
        # W_Stage1 is with size of: 8, 8, 257, 1
        # Perform ISTFT and norm for x_hat before PF
        x_hat_stage1_B_norm = Postprocessing(X_hat_Stage1,R,win_len,device)
        max_x = torch.max(abs(x_hat_stage1_B_norm),dim=1).values
        x_hat_stage1 = (x_hat_stage1_B_norm.T/max_x).T 
        
        # Perform ISTFT and norm for x_hat
        # IlaiZ 03/08 - Deleted this
        # x_hat_stage2_B_norm = Postprocessing(X_hat_Stage2,R,win_len,device)
        # max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
        # x_hat_stage2_time=x_hat_stage2_B_norm #(x_hat_stage2_B_norm.T/max_x).T       
        
        # Preprocessing & Postprocessing for the labeled signal
        X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])

        X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref) # No beamformer W in the function, it takes only signal as recorded/
        x = Postprocessing(X_stft_mic_ref,R,win_len,device) # speech as recorded in reference microphone
        max_x = torch.max(abs(x),dim=1).values
       # x = (x.T/max_x).T
        X_stft = return_as_complex(X_stft) #torch.Size([8, 8, 257, 497])
        loss,loss_L1, cost_distortionless, cost_minimum_variance_dir, cost_minimum_variance_white, SNR_output ,si_sdr_loss,cost_minimum_variance_two,loss_W_L1= compute_loss(x, X_stft, Y, X_hat_Stage1, X_hat_Stage2,None, W_Stage1, W_Stage2, fullLabels_x, fullnoise_first, fullnoise_second, white_noise, win_len, fs, T, R, device, cfg_loss, args)
        # Log metrics to wandb
        wandb.log({
            "batch_loss_train": loss.item(),
            "loss_L1": loss_L1.item(),
            "cost_distortionless": cost_distortionless.item(),
            "cost_minimum_variance_dir": cost_minimum_variance_dir.item(),
            "cost_minimum_variance_white": cost_minimum_variance_white.item(),
            "SNR_output": SNR_output.item(),
            "SI-SDR": si_sdr_loss.item(),
            "cost_minimum_variance_two" : cost_minimum_variance_two.item(),
            "loss_W_L1" : loss_W_L1.item()
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

    # VAL    
    epoch_val_loss = test(model, args, results_path, val_loader, device, cfg_loss, 0)
   
    return epoch_train_loss,epoch_val_loss