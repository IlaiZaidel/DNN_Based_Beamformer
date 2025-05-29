import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
#from sklearn.model_selection import train_test_split
from generate_dataset import GeneretedInputOutput,GeneretedData
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

cs = ConfigStore.instance()
cs.store(name="cunet_config", node=CUNETConfig)

@hydra.main(config_path = "conf", config_name = "config")
def main(cfg: CUNETConfig): 

    # ------- Only for WandB -------
    # Initialize wandb with the sanitized config
    wandb.init(
        # set the wandb project where this run will be logged
        project="DNN Based Beamformer",

        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.optimizer.learning_rate,
        "epochs": cfg.model_hp.epochs,

        }
    )

    if cfg.SavedModel == 0: # If train the model 
        # Load Data (Train & Val)
        train_path = cfg.paths.train_path
        train_data=GeneretedData(cfg, mode='train')
        # Create train & validation set 
        # train_set, val_set = train_test_split(train_data,train_size = cfg.model_hp.train_size_spilt,test_size = cfg.model_hp.val_size_spilt)  
        train_set, val_set = torch.utils.data.random_split(train_data,[cfg.model_hp.train_size_spilt, cfg.model_hp.val_size_spilt])  
        train_loader = DataLoader(train_set, batch_size = cfg.model_hp.batchSize, shuffle = cfg.model_hp.data_loader_shuffle,num_workers=16)
        val_loader = DataLoader(val_set, batch_size = cfg.model_hp.batchSize, shuffle = cfg.model_hp.data_loader_shuffle,num_workers=16)
    
        # Defining the model
        model = ExNetBFPF(cfg.modelParams)  #  #ExNetBFPF(cfg.modelParams)   #loadPreTrainedModel(cfg)#
    else: # If to load saved weights
        model = loadPreTrainedModel(cfg)

    # Defining the device 
    device_ids = [cfg.device.device_num]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids = device_ids)
    
    model.to(device)
    
    # Defining the optimizer
    if cfg.optimizer.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    else:
        print("error no optimzier has been configrues - got optimzier {}".format(cfg.optimizer.optimizer))
    
    if cfg.SavedModel == 0: # If train the model 
        # Training
        train_loss = []
        val_loss = []

        writer = SummaryWriter(log_dir = cfg.paths.log_path)

        for epoch in tqdm(range(cfg.model_hp.epochs)):
            epoch_train_loss,epoch_val_loss = train(model, cfg.params, cfg.paths.results_path, train_loader, val_loader, optimizer, device, cfg.loss, cfg.debug) 
            train_loss.append(epoch_train_loss / len(train_loader)) # divide by number of batches, to get mean loss for each epoch
            val_loss.append(epoch_val_loss / len(val_loader))       # divide by number of batches, to get mean loss for each epoch 
            
            print(f"Epoch {epoch+1}/{cfg.model_hp.epochs}, train_loss: {train_loss[epoch]}, val_loss: {val_loss[epoch]}")
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss[epoch],
                "val_loss": val_loss[epoch]
            })



            # Saved results in log file (TensorBorad)
            writer.add_scalar("Loss/train",epoch_train_loss / len(train_loader),epoch)
            writer.add_scalar("Loss/val",epoch_val_loss / len(val_loader),epoch)

            # Keeps the weights obtained at the end of the training        
            os.makedirs(cfg.paths.modelData_path, mode=0o777, exist_ok=True)
            PATH = cfg.paths.modelData_path + 'trained_model_dataset_withoutRev.pt'
            torch.save(model, PATH)

                    # Log loss curves to wandb
        wandb.log({
            "train_loss_curve": wandb.plot.line_series(
                xs=list(range(1, cfg.model_hp.epochs + 1)),
                ys=[train_loss, val_loss],
                keys=["Train Loss", "Validation Loss"],
                title="Loss Curves",
                xname="Epochs"
            )
        })

        writer.flush()
        writer.close()
          
    # Test - Data Loader
    # test_path = cfg.paths.test_path
    # test_data = GeneretedInputOutput(test_path,cfg.params.mic_ref)
    test_data = GeneretedData(cfg, mode='test')
    test_loader = DataLoader(test_data, batch_size = cfg.model_hp.batchSize, shuffle = cfg.model_hp.test_loader_shuffle)
    
    # Testing
    test_loss = test(model, cfg.params, cfg.paths.results_path, test_loader, device, cfg.loss, 1) 
    test_loss = test_loss / len(test_loader) 
    
    print(f"Test_loss: {test_loss}")
    
    if cfg.SavedModel == 0:
        # Plots
        plt.figure() 
        plt.plot(train_loss) 
        plt.plot(val_loss) 
        plt.xlabel('Epochs') 
        plt.ylabel('Loss') 
        plt.legend(['train loss','val loss'])
        plt.grid(True)
        plt.title('Loss')
        now = (datetime.now()).strftime("%d_%m_%Y__%H_%M_%S")
        path = cfg.paths.results_path + 'Loss_' + now + '.jpg'
        plt.savefig(path)


    # wandb.finish()  # Mark the end of wandb logging
if __name__ == '__main__':
    print('start')
    main()