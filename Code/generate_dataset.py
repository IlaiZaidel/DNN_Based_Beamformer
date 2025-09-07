from torch.utils.data import Dataset
import scipy.io as sio
import os, random
import pandas as pd
# import rir_generator
# at top of generate_dataset.py
from rir_generator import generate as rir_generat
import numpy as np 
import ast
import torchaudio
import torchaudio.functional as F
import scipy.signal as signal
import torch
import glob
#from fft_conv_pytorch import fft_conv
import soundfile as sf
from utils import snr_np
import sys

import os, sys, importlib

# SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator"  # folder that holds signal_generator.so
# if SIGGEN_DIR not in sys.path:
#     sys.path.insert(0, SIGGEN_DIR)   # put it FIRST so it wins

# # If something named 'signal_generator' was already imported, purge it
# if 'signal_generator' in sys.modules:
#     del sys.modules['signal_generator']

# import signal_generator as sg
# print("signal_generator loaded from:", getattr(sg, "__file__", "UNKNOWN (namespace pkg?)"))
# from signal_generator import SignalGenerator



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


# class GeneretedData(Dataset):
#     '''Generated Input and Output to the model'''

#     # def __init__(self,cfg):
#     #     '''
#     #     Args:
#     #         path (string): Directory with all the features.
#     #         mic_ref (int): Reference microphone
#     #     '''
#     #     self.cfg = cfg
#     #     self.mic_ref = cfg.params.mic_ref - 1 

#     #     self.df = pd.read_csv(cfg.dataset.df_path)
#     #     self.rir =rir_generator

#     def __init__(self,cfg, mode='train'):
#             '''
#             Args:
#                 cfg: Configuration object containing dataset and parameter settings.
#                 mode (str): 'train' or 'test', used to select correct CSV path
#             '''
#             self.cfg = cfg
#             self.mic_ref = cfg.params.mic_ref - 1 

#             # Choose train or test CSV based on mode
#             if mode == 'train':
#                 csv_path = cfg.dataset.df_path_train
#             elif mode == 'test':
#                 csv_path = cfg.dataset.df_path_test
#             else:
#                 raise ValueError("mode should be either 'train' or 'test'")

#             self.df = pd.read_csv(csv_path)
#             self.rir = rir_generator


#     def __len__(self): 
#         # Return the number of examples we have in the folder in the path
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         # Get audio file name
#         line = self.df.iloc[idx]
#         new_path = line["speaker_path"]  # The name of the file ## Change it to feature_vector!
        
#         mic_positions = np.array(ast.literal_eval(line["mic_positions"]))
#         source_position = [line["speaker_x"], line["speaker_y"], line["speaker_z"]]
#         noise_1_position = [line["noise1_x"], line["noise1_y"], line["noise1_z"]]
#         noise_2_position = [line["noise2_x"], line["noise2_y"], line["noise2_z"]]
#         room_dim = [line["room_x"], line["room_y"], line["room_z"]]
#         beta = line["beta"]
#         beta = np.array([beta] * 6)
#         n = line["n"]
#         M = mic_positions.shape[0]  # Number of microphones

#         # Generate RIRs (Room Impulse Responses)
#         h_n = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, source_position, room_dim, beta, n,order=0)).T
#         a_n_1 = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, noise_1_position, room_dim, beta, n,order=0)).T
#         a_n_2 = np.array(self.rir.generate(self.cfg.params.ck, self.cfg.params.fs, mic_positions, noise_2_position, room_dim, beta, n,order=0)).T
        
#         # Load speaker data using soundfile
#         x, fs = sf.read(new_path)  # x: (samples,) or (samples, channels)
        
#         # Ensure x is 2D with shape (samples, M)
#         x = np.tile(x[:, None], (1, M))  # Shape: (samples, M)
        
#         # ---------------------------
#         # Efficient Convolution for Direct Signal
#         # ---------------------------
#         d = np.array([signal.fftconvolve(x[:, m], h_n[m, :], mode='full')[:x.shape[0]] for m in range(M)]).T
        
#         # Zero-pad the beginning to simulate the noise period
#         T = self.cfg.params.T  # Total duration in seconds
#         noise_time = self.cfg.modelParams.noise_only_time # Seconds of noise at the start
#         zero_pad = np.zeros((int(fs * noise_time), M))
#         d = np.vstack((zero_pad, d[:int(fs * (T - noise_time)), :]))

#         # ---------------------------
#         # Efficient Noise Generation
#         # ---------------------------
#         # Generate AR noise process for both noise sources
#         w = np.random.randn(int(T * fs), 2)  # Shape: (T*fs, 2)
#         AR = [1, -0.7] # Low pass: [1, -0.7], High pass: [1, 0.7]
#         noise_processes = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

#         # Apply impulse responses using matrix multiplication
#         # n_1 = signal.lfilter(a_n_1, [1], noise_processes[:, 0])  # (T*fs, M)
#         # n_2 = signal.lfilter(a_n_2, [1], noise_processes[:, 1])  # (T*fs, M)
#         n1_list = [signal.lfilter(a_n_1[c, :], [1], noise_processes[:, 0]) for c in range(a_n_1.shape[0])]
#         n_1 = np.stack(n1_list, axis=1)  # Shape: (T*fs, M)

#         n2_list = [signal.lfilter(a_n_2[c, :], [1], noise_processes[:, 1]) for c in range(a_n_2.shape[0])]
#         n_2 = np.stack(n2_list, axis=1)  # Shape: (T*fs, M)
#         # ---------------------------
#         # SNR Scaling for Noise Components
#         # ---------------------------
#         SNR_noise = 3  # Desired SNR for noise
#         mic_ref = self.mic_ref

#         dVSn = np.sqrt(
#             (np.sum(d[:, mic_ref] ** 2) * 10 ** (-SNR_noise / 10)) /
#             (np.sum((n_1[:, mic_ref] + n_2[:, mic_ref]) ** 2) + 1e-10)
#         )
#         n_1 *= dVSn
#         n_2 *= dVSn

#         # ---------------------------
#         # White Noise Addition
#         # ---------------------------
#         SNR_white = 30  # 30 dB for white noise
#         flagwhiteNoise = 1  # Set this flag as needed (1 to include white noise, 0 otherwise)
        
#         v = np.random.randn(*d.shape)  # White noise with the same shape as d
#         dVSv = np.sqrt(
#             (np.sum(d[:, mic_ref] ** 2) * 10 ** (-SNR_white / 10)) /
#             (np.sum(v[:, mic_ref] ** 2) + 1e-10)
#         ) if flagwhiteNoise else 0
#         v *= dVSv

#         # ---------------------------
#         # Final Output Mixture
#         # ---------------------------
#         y = d + n_1 + v # d + n_1 + n_2 + v # Only First Noise!

#         max_y = np.abs(y).max()
#         y /= max_y
#         d /= max_y
#         n_1 /= max_y
#         n_2 /= max_y
#         v /= max_y 
#         return y, d, n_1, v




# class GeneretedDataTracking(Dataset):
#     """Online data generation with moving speaker"""

#     def __init__(self, cfg, mode='train'):
#         self.cfg = cfg
#         self.mic_ref = cfg.params.mic_ref - 1

#         csv_path = cfg.dataset.df_path_train if mode == 'train' else cfg.dataset.df_path_test
#         self.df = pd.read_csv(csv_path)
#         # self.rir = rir_generator

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         line = self.df.iloc[idx]
#         M = len(ast.literal_eval(line["mic_positions"]))
#         fs, T, hop = self.cfg.params.fs, self.cfg.params.T, 32
#         L = [line[f"room_{axis}"] for axis in "xyz"]

#         mic_positions = np.array(ast.literal_eval(line["mic_positions"]))
#         speaker_start_x = line["speaker_start_x"]
#         speaker_start_y = line["speaker_start_y"]
#         speaker_start_z = line["speaker_start_z"]
#         start = np.array([speaker_start_x, speaker_start_y, speaker_start_z])

#         speaker_stop_x = line["speaker_stop_x"]
#         speaker_stop_y = line["speaker_stop_y"]
#         speaker_stop_z = line["speaker_stop_z"]
#         stop = np.array([speaker_stop_x, speaker_stop_y, speaker_stop_z])

#         sp_path = np.zeros((T*fs, 3))
#         rp_path = np.zeros((T*fs, 3, M))
#         for i in range(0, T*fs, hop):
#             alpha = i / (T*fs)
#             sp = start + alpha * (stop - start)
#             sp_path[i:i+hop] = sp
#             for m in range(M):
#                 rp_path[i:i+hop, :, m] = mic_positions[m]

#         x, _ = sf.read(line["speaker_path"])
#         if x.ndim > 1:
#             x = x[:, 0]
#         x = x[:T*fs]
 
#         # result = gen.generate(list(x), 340.0, fs, rp_path.tolist(), sp_path.tolist(), L, [line["beta"]]*6, 1024, mtype="o", order=2, hp_filter=True)
#         gen = SignalGenerator()
#         use_cuda = torch.cuda.is_available()
#         generate_fn = gen.generate_cuda if use_cuda else gen.generate

#         result = generate_fn(
#             list(x),               # input_signal
#             340.0,                 # c
#             fs,                    # fs
#             rp_path.tolist(),      # r_path: [T][3][M]
#             sp_path.tolist(),      # s_path: [T][3]
#             L,                     # room dims [3]
#             [line["beta"]]*6,      # beta (6 coeffs) OR [TR]
#             1024,                  # nsamples
#             "o",                   # mtype
#             2,                     # order
#             3,                     # dim
#             [],                    # orientation
#             True,                   # hp_filter
#             mode="fast"
#         )
#         print("passed the signal generator")
#         d = np.vstack((np.zeros((int(fs * self.cfg.modelParams.noise_only_time), M)), np.array(result.output).T[:int(fs * (T - self.cfg.modelParams.noise_only_time)), :]))

#         w = np.random.randn(int(T * fs), 2)
#         AR = [1, -0.7]
#         noise_processes = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

#         a_n_1 = np.array(rir_generat(self.cfg.params.ck, fs, mic_positions, [line[f"noise1_{axis}"] for axis in "xyz"], L, [line["beta"]]*6, line["n"], order=0)).T
#         a_n_2 = np.array(rir_generat(self.cfg.params.ck, fs, mic_positions, [line[f"noise2_{axis}"] for axis in "xyz"], L, [line["beta"]]*6, line["n"], order=0)).T

#         n1 = np.stack([signal.lfilter(a_n_1[c], [1], noise_processes[:, 0]) for c in range(M)], axis=1)
#         n2 = np.stack([signal.lfilter(a_n_2[c], [1], noise_processes[:, 1]) for c in range(M)], axis=1)

#         dVSn = np.sqrt((np.sum(d[:, self.mic_ref] ** 2) * 10 ** (-3 / 10)) / (np.sum((n1[:, self.mic_ref] + n2[:, self.mic_ref]) ** 2) + 1e-10))
#         n1 *= dVSn
#         n2 *= dVSn

#         v = np.random.randn(*d.shape)
#         dVSv = np.sqrt((np.sum(d[:, self.mic_ref] ** 2) * 10 ** (-30 / 10)) / (np.sum(v[:, self.mic_ref] ** 2) + 1e-10))
#         v *= dVSv

#         y = d + n1 + v

#         max_y = np.abs(y).max()
#         return y/max_y, d/max_y, n1/max_y, v/max_y

class GeneratedDataTrackingFromClean(Dataset):
    """
    Uses pre-generated clean MAT files and adds two spatial colored noises + white noise.
    File mapping: index == CSV row index (zero-based).
    Expects files named: <clean_prefix><idx:07d>.mat inside cfg.dataset.clean_dir
    """

    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        csv_path = cfg.dataset.df_path_train if mode == "train" else cfg.dataset.df_path_test
        self.clean_dir = cfg.paths.train_path if mode == "train" else cfg.paths.test_path
        self.clean_prefix = "clean_example_"
        self.df = pd.read_csv(csv_path)
        base = "/dsi/gannot-lab1/datasets/AudioSet_noise"
        sub  = "train" if mode == "train" else "test"
        self.noise_dir = os.path.join(base, sub, "Air conditioning")
        self.noise_wavs = sorted(glob.glob(os.path.join(self.noise_dir, "*.wav")))

    def __len__(self):
        return len(self.df)

    def _mat_path(self, idx: int) -> str:
        fname = f"{self.clean_prefix}{idx:07d}.mat"
        return os.path.join(self.clean_dir, fname)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        fs = self.cfg.params.fs
        ck = self.cfg.params.ck
        # ---------- Load clean (moving speaker) ----------
        mat_path = self._mat_path(idx)
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing clean MAT for idx={idx}: {mat_path}")
        data = sio.loadmat(mat_path)
        if "clean" not in data:
            raise KeyError(f"'clean' not in {mat_path}")

        d_full = np.array(data["clean"], dtype=np.float64)  # (samples, M) expected
        if d_full.ndim != 2:
            raise ValueError(f"clean has wrong shape in {mat_path}: {d_full.shape}")
        # fix orientation if needed
        if d_full.shape[0] < d_full.shape[1]:
            d_full = d_full.T

        M = d_full.shape[1]
        N = int(self.cfg.params.T * self.cfg.params.fs)
        if d_full.shape[0] < N:
            reps = int(np.ceil(N / max(1, d_full.shape[0])))
            d_full = np.tile(d_full, (reps, 1))
        d_full = d_full[:N, :]

        # noise-only head for direct speech
        nlead = int(self.cfg.params.fs * self.cfg.params.noise_only_time)
        keep = N - nlead
        d = np.vstack([np.zeros((nlead, M)), d_full[:keep, :]])

        # ---------- Geometry/RIR for two stationary noises (order=0) ----------
        L = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
        beta = [float(row["beta"])] * 6
        n_taps = int(row["n"])
        mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)

        noise1_pos = [float(row["noise1_x"]), float(row["noise1_y"]), float(row["noise1_z"])]
        noise2_pos = [float(row["noise2_x"]), float(row["noise2_y"]), float(row["noise2_z"])]

        a_n_1 = np.array(rir_generat(ck,fs, mic_positions, noise1_pos, L, beta, n_taps, order=0)).T  # (M, taps)
        a_n_2 = np.array(rir_generat(ck,fs, mic_positions, noise2_pos, L, beta, n_taps, order=0)).T  # (M, taps)

        # ---------- AR noise drivers -> per-mic filtering ----------
        w = np.random.randn(N, 2)
        AR = [1, -0.7]
        noise_proc = np.column_stack([
            signal.lfilter([1], AR, w[:, 0]),
            signal.lfilter([1], AR, w[:, 1]),
        ])  # (N,2)

        n1 = np.stack([signal.lfilter(a_n_1[m], [1], noise_proc[:, 0]) for m in range(M)], axis=1)
        n2 = np.stack([signal.lfilter(a_n_2[m], [1], noise_proc[:, 1]) for m in range(M)], axis=1)
        # ----- pick two drivers from AudioSet (random), mono, resample if needed, LAST N -----
        p1 = random.choice(self.noise_wavs); x1, sr1 = sf.read(p1, always_2d=False)
        p2 = random.choice(self.noise_wavs); x2, sr2 = sf.read(p2, always_2d=False)

        if x1.ndim > 1: x1 = x1.mean(axis=1)
        if x2.ndim > 1: x2 = x2.mean(axis=1)

        if sr1 != fs: x1 = signal.resample_poly(x1, fs, sr1)
        if sr2 != fs: x2 = signal.resample_poly(x2, fs, sr2)

        if len(x1) < N: x1 = np.tile(x1, int(np.ceil(N/len(x1))))
        if len(x2) < N: x2 = np.tile(x2, int(np.ceil(N/len(x2))))
        x1 = np.asarray(x1[-N:], dtype=np.float64)
        x2 = np.asarray(x2[-N:], dtype=np.float64)

        # ----- spatial filtering -----
        n1 = np.stack([signal.lfilter(a_n_1[m], [1.0], x1) for m in range(M)], axis=1)[:N, :]
        n2 = np.stack([signal.lfilter(a_n_2[m], [1.0], x2) for m in range(M)], axis=1)[:N, :]

        # ---------- SNR scaling (colored noises vs. d at ref mic) ----------
        SNR_noise_db = 3.0
        ref = self.cfg.params.mic_ref
        d_power = np.sum(d[:, ref] ** 2)
        n_power = np.sum((n1[:, ref] ) ** 2) + 1e-10
        scale_n = np.sqrt(d_power * 10 ** (-SNR_noise_db / 10.0) / n_power)
        n1 *= scale_n
        n2 *= scale_n

        # ---------- White noise ----------
        SNR_white_db = 30.0
        v = np.random.randn(*d.shape)
        v_power = np.sum(v[:, ref] ** 2) + 1e-10
        scale_v = np.sqrt(d_power * 10 ** (-SNR_white_db / 10.0) / v_power)
        v *= scale_v

        # ---------- Mix & normalize ----------
        y = d + n1  + v
        peak = np.max(np.abs(y))
        
        y /= peak; d /= peak; n1 /= peak; n2 /= peak; v /= peak


        return y, d, n1, v
