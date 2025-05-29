from torch.utils.data import Dataset
import scipy.io as sio
import os
import pandas as pd
import rir_generator
import numpy as np 
import ast
import torchaudio
import torchaudio.functional as F
import scipy.signal as signal
import torch
from fft_conv_pytorch import fft_conv
import soundfile as sf
from utils import snr_np
class GeneretedInputOutput(Dataset):
    '''Generated Input and Output to the model'''

    def __init__(self,path,mic_ref):
        '''
        Args:
            path (string): Directory with all the features.
            mic_ref (int): Reference microphone
        '''
        self.path = path
        self.mic_ref = mic_ref - 1 

    def __len__(self): 
        # Return the number of examples we have in the folder in the path
        file_list = os.listdir(self.path)  
        return len(file_list)

    def __getitem__(self,i):
        # Get audio file name 
        new_path = self.path + 'my_feature_vector_' + str(i) + '.mat'   # The name of the file ## Change it to feature_vector!
        train = sio.loadmat(new_path) 

        # Loads data
        self.y = train['feature']
        try:
            self.x = train['original']
        except:
            self.x = train['fulloriginal']     
        
        # 1 Noise
        #self.fullnoise = train['fullnoise']


        # 20.03 - Two noise terms:
        self.fullnoise_first = train['fullnoise_first']
        self.fullnoise_second = train['fullnoise_second']


        
        # Normalizing the input and output signals
        max_y = abs(self.y[:,self.mic_ref]).max(0)
        self.y = self.y/max_y
        self.x = self.x//max_y
        self.fullnoise_first = self.fullnoise_first/max_y #/max_noise
        self.fullnoise_second = self.fullnoise_second/max_y #/max_noise
        return self.y,self.x, self.fullnoise_first, self.fullnoise_second


class GeneretedData(Dataset):
    '''Generated Input and Output to the model'''

    # def __init__(self,cfg):
    #     '''
    #     Args:
    #         path (string): Directory with all the features.
    #         mic_ref (int): Reference microphone
    #     '''
    #     self.cfg = cfg
    #     self.mic_ref = cfg.params.mic_ref - 1 

    #     self.df = pd.read_csv(cfg.dataset.df_path)
    #     self.rir =rir_generator

    def __init__(self,cfg, mode='train'):
            '''
            Args:
                cfg: Configuration object containing dataset and parameter settings.
                mode (str): 'train' or 'test', used to select correct CSV path
            '''
            self.cfg = cfg
            self.mic_ref = cfg.params.mic_ref - 1 

            # Choose train or test CSV based on mode
            if mode == 'train':
                csv_path = cfg.dataset.df_path_train
            elif mode == 'test':
                csv_path = cfg.dataset.df_path_test
            else:
                raise ValueError("mode should be either 'train' or 'test'")

            self.df = pd.read_csv(csv_path)
            self.rir = rir_generator


    def __len__(self): 
        # Return the number of examples we have in the folder in the path
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get audio file name
        line = self.df.iloc[idx]
        new_path = line["speaker_path"]  # The name of the file ## Change it to feature_vector!
        
        mic_positions = np.array(ast.literal_eval(line["mic_positions"]))
        source_position = [line["speaker_x"], line["speaker_y"], line["speaker_z"]]
        noise_1_position = [line["noise1_x"], line["noise1_y"], line["noise1_z"]]
        noise_2_position = [line["noise2_x"], line["noise2_y"], line["noise2_z"]]
        room_dim = [line["room_x"], line["room_y"], line["room_z"]]
        beta = line["beta"]
        beta = np.array([beta] * 6)
        n = line["n"]
        M = mic_positions.shape[0]  # Number of microphones

        # Generate RIRs (Room Impulse Responses)
        h_n = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, source_position, room_dim, beta, n,order=0)).T
        a_n_1 = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, noise_1_position, room_dim, beta, n,order=0)).T
        a_n_2 = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, noise_2_position, room_dim, beta, n,order=0)).T
        
        # Load speaker data using soundfile
        x, fs = sf.read(new_path)  # x: (samples,) or (samples, channels)
        
        # Ensure x is 2D with shape (samples, M)
        x = np.tile(x[:, None], (1, M))  # Shape: (samples, M)
        
        # ---------------------------
        # Efficient Convolution for Direct Signal
        # ---------------------------
        d = np.array([signal.fftconvolve(x[:, m], h_n[m, :], mode='full')[:x.shape[0]] for m in range(M)]).T
        
        # Zero-pad the beginning to simulate the noise period
        T = self.cfg.params.T  # Total duration in seconds
        noise_time = 0.5  # Seconds of noise at the start
        zero_pad = np.zeros((int(fs * noise_time), M))
        d = np.vstack((zero_pad, d[:int(fs * (T - noise_time)), :]))

        # ---------------------------
        # Efficient Noise Generation
        # ---------------------------
        # Generate AR noise process for both noise sources
        w = np.random.randn(int(T * fs), 2)  # Shape: (T*fs, 2)
        AR = [1, -0.7] # Low pass: [1, -0.7], High pass: [1, 0.7]
        noise_processes = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

        # Apply impulse responses using matrix multiplication
        # n_1 = signal.lfilter(a_n_1, [1], noise_processes[:, 0])  # (T*fs, M)
        # n_2 = signal.lfilter(a_n_2, [1], noise_processes[:, 1])  # (T*fs, M)
        n1_list = [signal.lfilter(a_n_1[c, :], [1], noise_processes[:, 0]) for c in range(a_n_1.shape[0])]
        n_1 = np.stack(n1_list, axis=1)  # Shape: (T*fs, M)

        n2_list = [signal.lfilter(a_n_2[c, :], [1], noise_processes[:, 1]) for c in range(a_n_2.shape[0])]
        n_2 = np.stack(n2_list, axis=1)  # Shape: (T*fs, M)
        # ---------------------------
        # SNR Scaling for Noise Components
        # ---------------------------
        SNR_noise = 3  # Desired SNR for noise
        mic_ref = self.mic_ref

        dVSn = np.sqrt(
            (np.sum(d[:, mic_ref] ** 2) * 10 ** (-SNR_noise / 10)) /
            (np.sum((n_1[:, mic_ref] + n_2[:, mic_ref]) ** 2) + 1e-10)
        )
        n_1 *= dVSn
        n_2 *= dVSn

        # ---------------------------
        # White Noise Addition
        # ---------------------------
        SNR_white = 30  # 30 dB for white noise
        flagwhiteNoise = 1  # Set this flag as needed (1 to include white noise, 0 otherwise)
        
        v = np.random.randn(*d.shape)  # White noise with the same shape as d
        dVSv = np.sqrt(
            (np.sum(d[:, mic_ref] ** 2) * 10 ** (-SNR_white / 10)) /
            (np.sum(v[:, mic_ref] ** 2) + 1e-10)
        ) if flagwhiteNoise else 0
        v *= dVSv

        # ---------------------------
        # Final Output Mixture
        # ---------------------------
        y = d + n_1 + v # d + n_1 + n_2 + v # Only First Noise!

        max_y = np.abs(y).max()
        y /= max_y
        d /= max_y
        n_1 /= max_y
        n_2 /= max_y
        v /= max_y 
        return y, d, n_1, n_2, v

