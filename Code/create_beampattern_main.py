import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
#from sklearn.model_selection import train_test_split
from generate_dataset_beampattern import GeneretedInputOutputBeamPattern
import hydra
import os
from datetime import datetime
import matplotlib.pyplot as plt
from config import CUNETConfig 
from hydra.core.config_store import ConfigStore
from ExNetBFPFModel import ExNetBFPF
from LoadPreTrainedModel import loadPreTrainedModel
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter   
import wandb  # Added
import json
from tqdm import tqdm
import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation, return_as_complex
from ComputeLoss import Loss
from saveResults import saveResults
import criterionFile
from RTF_covariance_whitening import covariance_whitening, fix_covariance_whitening
from loss_function import compute_loss
cs = ConfigStore.instance()
cs.store(name="cunet_config", node=CUNETConfig)

@hydra.main(config_path = "conf", config_name = "config")
def main(cfg: CUNETConfig): 


    
    model = loadPreTrainedModel(cfg)

    # Defining the device 
    device_ids = [cfg.device.device_num]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids = device_ids)
    
    model.to(device)
    
    # Test - Data Loader
    # Path of reference_feature_vactor_0.mat
    test_path = '/home/dsi/ilaiz/DNN_Based_Beamformer/Code/beampattern_gen/'
    test_data = GeneretedInputOutputBeamPattern(test_path,cfg.params.mic_ref)
    test_loader = DataLoader(test_data, batch_size = cfg.model_hp.batchSize, shuffle = cfg.model_hp.test_loader_shuffle)
    
    # Testing
    test_loss, W_Stage1, W_Stage2 = test(model, cfg.params, cfg.paths.results_path, test_loader, device, cfg.loss, 1) 
    test_loss = test_loss / len(test_loader) 
    
    print(f"Test_loss: {test_loss}")
    



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
        for i, (y, labels_x, fullnoise) in enumerate(test_loader):
            # Extract Data
            y = y.to(device)                    # y = B,T*fs,M - noisy signal in the time domain
            fullLabels_x = labels_x.to(device)  # x = B,T*fs,M - target signal in the time domain
            labels_x = torch.unsqueeze(fullLabels_x[:,:,mic_ref-1],2) # x_ref - B,T*fs,1 - target signal ref in the time domain  
            
            # Perform STFT and Padding
            Y = Preprocesing(y, win_len, fs, T, R, device)  # Y = B,M,2*F,L - noisy signal in the STFT domain
            
            # Forward
            W_timeChange,X_hat_Stage1,Y,W_Stage1,X_hat_Stage2,W_Stage2,skip_Stage1,skip_Stage2 = model(Y, device)

            # W_Stage2 is shape torch.Size([8, 1, 514, 497])
            B,M,F,L = W_Stage2.size()
            W_Stage2 = W_Stage2.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()
            W_Stage2= torch.view_as_complex(W_Stage2) # torch.Size([8, 1, 257, 497])
            # Squeeze the singleton dimension (M = 1) to align with noise_stage1
            W_Stage2 = W_Stage2.squeeze(1)  # Now W_Stage2 has shape [8, 257, 497]


            # Perform ISTFT and norm for x_hat before PF
            x_hat_stage1_B_norm = Postprocessing(X_hat_Stage1,R,win_len,device)
            max_x = torch.max(abs(x_hat_stage1_B_norm),dim=1).values
            x_hat_stage1 = (x_hat_stage1_B_norm.T/max_x).T

            # Perform ISTFT and norm for x_hat
            x_hat_stage2_B_norm = Postprocessing(X_hat_Stage2,R,win_len,device)
            max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
            x_hat_stage2 = (x_hat_stage2_B_norm.T/max_x).T            

            # Preprocessing & Postprocessing for the labeled signal
            X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device) 
            X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref)
            x = Postprocessing(X_stft_mic_ref,R,win_len,device)
            max_x = torch.max(abs(x),dim=1).values
            x = (x.T/max_x).T

            # Calculate the loss function 
            #loss = Loss(x,x_hat_stage2,cfg_loss)*args.Enable_cost_mae   
            X_stft = return_as_complex(X_stft) #torch.Size([8, 8, 257, 497])     
            
            loss,loss_L1, cost_distortionless, cost_minimum_variance_dir, cost_minimum_variance_white, SNR_output ,si_sdr_loss= compute_loss(x, X_stft, Y, X_hat_Stage1, x_hat_stage2, W_Stage1, W_Stage2, fullLabels_x, fullnoise, win_len, fs, T, R, device, cfg_loss, args)
            # Backward & Optimize     
            epoch_test_loss += loss.item() 
                    
            # Save results
            if iftest == 1:
                saveResults(Y,X_stft,skip_Stage1,skip_Stage2,W_Stage1,W_timeChange,W_Stage2,X_hat_Stage1,X_hat_Stage2,
                            y[:,:,mic_ref-1],x_hat_stage1,x_hat_stage2,results_path,i,fs)           

    return epoch_test_loss, W_Stage1, W_Stage2
    # wandb.finish()  # Mark the end of wandb logging
if __name__ == '__main__':
    print('start')
    main()
    print('ilai zaidel is done')