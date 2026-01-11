import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation, return_as_complex
from ComputeLoss import Loss
from saveResults import saveResults
import criterionFile
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from loss_function import compute_loss
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

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

        for i, (y, labels_x, first_speaker, second_speaker, babble, white_noise, rir_first, rir_second) in enumerate(test_loader): # on batch
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
            
            # Zero the parameter gradients


            
            loss = torch.tensor(0.0, device=device)

            ##########################################
            ####### Model
            ##########################################
            W_Stage1_left,  X_hat_Stage1_C_left, Y = model(Y,rir_first, rir_second , device, mode="train")

            
            # # Preprocessing & Postprocessing for the labeled signal
            # X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device) # torch.Size([8, 8, 514, 497])
            # X_stft = return_as_complex(X_stft) #torch.Size([8, 8, 257, 497])
            
            
            # X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref) # No beamformer W in the function, it takes only signal as recorded/
            # x = Postprocessing(X_stft_mic_ref,R,win_len,device) # speech as recorded in reference microphone
            # max_x = torch.max(abs(x),dim=1).values
            # # x = (x.T/max_x).T
            
            si_sdr = ScaleInvariantSignalDistortionRatio().to(device)

            if args.beta_SISDR> 0:

                X_hat_Stage1_left_time = Postprocessing(X_hat_Stage1_C_left,R,win_len,device) #torch.Size([16, 64000])
                # first_speaker_2= Postprocessing(FIRST_SPEAKER_stft[:, args.mic_ref_left,:,:],R,win_len,device) #X_stft.shape is torch.Size([16, 8, 257, 497])
                si_sdr_loss_first = si_sdr(first_speaker[:,:,args.mic_ref_left],X_hat_Stage1_left_time)

                # second_speaker= Postprocessing(SECOND_SPEAKER_stft[:, args.mic_ref_left,:,:],R,win_len,device) #X_stft.shape is torch.Size([16, 8, 257, 497])
                si_sdr_loss_second = si_sdr(second_speaker[:,:,args.mic_ref_left],X_hat_Stage1_left_time)

                total_SI_SDR = si_sdr_loss_first + si_sdr_loss_second

                loss -=total_SI_SDR*args.beta_SISDR


            # Backward & Optimize     
            epoch_test_loss += loss.item() 

            # Normalizing: 

            # x = (x.T/max_x).T
            # max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
            # x_hat_stage2 = (x_hat_stage2_B_norm.T/max_x).T            
            # x_hat_stage2_time = torch.zeros_like(x, device=device)
            # X_hat_Stage2= torch.zeros_like(X_hat_Stage1, device=device)
            # Save results
            if iftest == 1:
                # Here I assume:
                #   - you want to save the full batch as-is
                #   - first_speaker/second_speaker are time-domain (same as input)
                saveResults(
                    Y=Y,
                    FIRST_SPEAKER_stft=FIRST_SPEAKER_stft,
                    SECOND_SPEAKER_stft=SECOND_SPEAKER_stft,
                    W_Stage1_left=W_Stage1_left,
                    X_hat_Stage1_C_left=X_hat_Stage1_C_left,
                    y=y,
                    first_speaker=first_speaker,
                    second_speaker=second_speaker,
                    x_hat_stage1_left=X_hat_Stage1_left_time,
                    results_path=results_path,
                    i=i,
                    fs=fs,
                )

       

    return epoch_test_loss