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
from utils import true_rtf_from_all_rirs
#from fft_conv_pytorch import fft_conv
import soundfile as sf
from utils import snr_np
import sys
import math
import os, sys, importlib

SIGGEN_DIR = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_rtf_output"  # folder that holds signal_generator.so
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)   # put it FIRST so it wins

# If something named 'signal_generator' was already imported, purge it
if 'signal_generator' in sys.modules:
    del sys.modules['signal_generator']

import signal_generator as sg
# print("signal_generator loaded from:", getattr(sg, "__file__", "UNKNOWN (namespace pkg?)"))
from signal_generator import SignalGenerator



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




class GeneretedDataTracking_Signal_Generator(Dataset):
    """Online data generation with moving speaker"""

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mic_ref = cfg.params.mic_ref - 1

        csv_path = cfg.dataset.df_path_train if mode == 'train' else cfg.dataset.df_path_test
        self.df = pd.read_csv(csv_path)
        # self.rir = rir_generator

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        M = len(ast.literal_eval(line["mic_positions"]))
        fs, T, hop = self.cfg.params.fs, self.cfg.params.T, 32
        L = [line[f"room_{axis}"] for axis in "xyz"]

        mic_positions = np.array(ast.literal_eval(line["mic_positions"]))
        speaker_start_x = line["speaker_start_x"]
        speaker_start_y = line["speaker_start_y"]
        speaker_start_z = line["speaker_start_z"]
        start = np.array([speaker_start_x, speaker_start_y, speaker_start_z])

        speaker_stop_x = line["speaker_stop_x"]
        speaker_stop_y = line["speaker_stop_y"]
        speaker_stop_z = line["speaker_stop_z"]
        stop = np.array([speaker_stop_x, speaker_stop_y, speaker_stop_z])

        sp_path = np.zeros((T*fs, 3))
        rp_path = np.zeros((T*fs, 3, M))
        for i in range(0, T*fs, hop):
            alpha = i / (T*fs)
            sp = start + alpha * (stop - start)
            sp_path[i:i+hop] = sp
            for m in range(M):
                rp_path[i:i+hop, :, m] = mic_positions[m]

        x, _ = sf.read(line["speaker_path"])
        if x.ndim > 1:
            x = x[:, 0]
        x = x[:T*fs]
 
        # result = gen.generate(list(x), 340.0, fs, rp_path.tolist(), sp_path.tolist(), L, [line["beta"]]*6, 1024, mtype="o", order=2, hp_filter=True)
        gen = SignalGenerator()
        result = gen.generate(
            list(x),               # input_signal
            340.0,                 # c
            fs,                    # fs
            rp_path.tolist(),      # r_path: [T][3][M]
            sp_path.tolist(),      # s_path: [T][3]
            L,                     # room dims [3]
            [line["beta"]]*6,      # beta (6 coeffs) OR [TR]
            1024,                  # nsamples
            "o",                   # mtype
            0,                     # order
            3,                     # dim
            [],                    # orientation
            True,                   # hp_filter
            mode="fast"
        )
        print("passed the signal generator")
        d = np.vstack((np.zeros((int(fs * self.cfg.modelParams.noise_only_time), M)), np.array(result.output).T[:int(fs * (T - self.cfg.modelParams.noise_only_time)), :]))

        w = np.random.randn(int(T * fs), 2)
        AR = [1, -0.7]
        noise_processes = np.apply_along_axis(lambda x: signal.lfilter([1], AR, x), axis=0, arr=w)

        a_n_1 = np.array(rir_generat(self.cfg.params.ck, fs, mic_positions, [line[f"noise1_{axis}"] for axis in "xyz"], L, [line["beta"]]*6, line["n"], order=0)).T
        a_n_2 = np.array(rir_generat(self.cfg.params.ck, fs, mic_positions, [line[f"noise2_{axis}"] for axis in "xyz"], L, [line["beta"]]*6, line["n"], order=0)).T

        n1 = np.stack([signal.lfilter(a_n_1[c], [1], noise_processes[:, 0]) for c in range(M)], axis=1)
        n2 = np.stack([signal.lfilter(a_n_2[c], [1], noise_processes[:, 1]) for c in range(M)], axis=1)

        dVSn = np.sqrt((np.sum(d[:, self.mic_ref] ** 2) * 10 ** (-3 / 10)) / (np.sum((n1[:, self.mic_ref] + n2[:, self.mic_ref]) ** 2) + 1e-10))
        n1 *= dVSn
        n2 *= dVSn

        v = np.random.randn(*d.shape)
        dVSv = np.sqrt((np.sum(d[:, self.mic_ref] ** 2) * 10 ** (-30 / 10)) / (np.sum(v[:, self.mic_ref] ** 2) + 1e-10))
        v *= dVSv

        y = d + n1 + v

        max_y = np.abs(y).max()
        return y/max_y, d/max_y, n1/max_y, v/max_y

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
        # base = "/dsi/gannot-lab1/datasets/AudioSet_noise"
        # sub  = "train" if mode == "train" else "train"
        # self.noise_dir = os.path.join(base, sub, "Air conditioning")
        self.noise_dir = "/dsi/gannot-lab1/datasets/whamr/wav16k/min/tt/noise"
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


        # ---------- Pink noise ----------
        # def generate_pink(N):
        #     # Simple IIR filter approximation of pink noise
        #     b = [0.049922035, 0.050612699, 0.050979644, 0.048882058]
        #     a = [1, -2.494956002, 2.017265875, -0.522189400]
        #     white = np.random.randn(N)
        #     pink = signal.lfilter(b, a, white)
        #     return pink

        # pink = generate_pink(N)

        # # # scale pink noise vs. d at ref mic
        # # SNR_pink_db = 10.0
        # # p_power = np.sum(pink[:, ref] ** 2) + 1e-10
        # # scale_p = np.sqrt(d_power * 10 ** (-SNR_pink_db / 10.0) / p_power)
        # # pink *= scale_p
        # x1 = pink
        # x2 = pink

        # ----- spatial filtering -----
        n1 = np.stack([signal.lfilter(a_n_1[m], [1.0], x1) for m in range(M)], axis=1)[:N, :]
        n2 = np.stack([signal.lfilter(a_n_2[m], [1.0], x2) for m in range(M)], axis=1)[:N, :]

        # ---------- SNR scaling (colored noises vs. d at ref mic) ----------
        SNR_noise_db = 4.0
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
    

# ----------------- main dataset -----------------
class GeneratedDataTrackingFromCleanBabbleNoise(Dataset):
    """
    Loads clean multichannel speech (from .mat) and precomputed multichannel babble (from .wav),
    then does SNR scaling (babble vs. clean @ ref mic), adds white noise, and returns tensors.

    __getitem__ returns:
        y : (N, M)  torch.float32  # mixture
        d : (N, M)  torch.float32  # clean (with noise-only head)
        b : (N, M)  torch.float32  # babble (scaled)
        v : (N, M)  torch.float32  # white noise (scaled)
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        csv_path = cfg.dataset.df_path_train if mode == "train" else cfg.dataset.df_path_test

        import pandas as pd
        self.df = pd.read_csv(csv_path)

        # Clean .mat location/prefix
        self.clean_dir = cfg.paths.train_path if mode == "train" else cfg.paths.test_path
        self.clean_prefix = getattr(cfg, "clean_prefix", "clean_example_")

        # Babble folder (precomputed multichannel wavs)
        self.babble_dir = cfg.paths.babble_noise_train_path if mode == "train" else cfg.paths.babble_noise_test_path

        # Cache frequently used params
        self.fs  = int(getattr(cfg.params, "fs", 16000))
        self.T   = float(getattr(cfg.params, "T", 4.0))
        self.N   = int(self.fs * self.T)
        # mic_ref in your code is 1-based (train loop uses mic_ref-1 for slicing)
        self.mic_ref = int(getattr(cfg.params, "mic_ref", 4))
        self.noise_only_time = float(getattr(cfg.params, "noise_only_time", 0.5))
        self.SNR_babble_db = random.uniform(3.0, 12.0)  if mode == "train" else 6 # random.uniform(5.0, 8.0)
        self.SNR_white_db  = float(getattr(cfg.params, "SNR_white_db", 30.0))

    def __len__(self):
        return len(self.df)

    def _mat_path(self, idx: int) -> str:
        return os.path.join(self.clean_dir, f"{self.clean_prefix}{idx:07d}.mat")

    def _babble_path(self, idx: int) -> str:
        # matches precompute naming: babble_{idx:07d}.wav
        return os.path.join(self.babble_dir, f"babble_{idx:07d}.wav")

    def __getitem__(self, idx: int):
        fs = self.fs
        N  = self.N

        # ----------------- load clean (.mat) -----------------
        mat_path = self._mat_path(idx)
        data = sio.loadmat(mat_path)


        d_full = np.asarray(data["clean"], dtype=np.float32)  # (samples, M) or (M, samples)
        
        
        # time crop/pad to N

        M = d_full.shape[1]
        
        # build clean with noise-only head
        nlead = int(fs * self.noise_only_time)
        
        d = np.vstack([np.zeros((nlead, M), dtype=np.float32), d_full]).astype(np.float32, copy=False)



        all_rirs = np.asarray(data["all_rirs"], dtype=np.float32)
        # all_RTFs =  true_rtf_from_all_rirs(all_rirs, WIN_LENGTH, REF_MIC)
        # ----------------- load babble (multichannel .wav) -----------------
        b_path = self._babble_path(idx)

        babble, sr = sf.read(b_path, always_2d=True)  # (N', M') float32


        babble = babble.astype(np.float32, copy=False)

        # ----------------- SNR scaling (babble vs d @ ref mic) -----------------
        ref_idx = max(0, self.mic_ref - 1)  # 1-based -> 0-based
        d_power = float(np.sum(d[:, ref_idx] ** 2) + 1e-12)
        b_power = float(np.sum(babble[:, ref_idx] ** 2) + 1e-12)
        scale_b = np.sqrt(d_power * 10.0 ** (-self.SNR_babble_db / 10.0) / b_power).astype(np.float32)
        babble *= scale_b

        # ----------------- white noise -----------------
        v = np.random.randn(N, M).astype(np.float32)
        v_power = float(np.sum(v[:, ref_idx] ** 2) + 1e-12)
        scale_v = np.sqrt(d_power * 10.0 ** (-self.SNR_white_db / 10.0) / v_power).astype(np.float32)
        v *= scale_v

        # ----------------- final mix & peak normalize -----------------
        y = d + babble + v
        peak = float(np.max(np.abs(y)) + 1e-12)
        y /= peak; d /= peak; babble /= peak; v /= peak



        return y, d, babble, v, all_rirs
    




class GeneratedDataTrackingFromCleanPinkNoise(Dataset):
    """
    Loads clean multichannel speech (from .mat) and precomputed multichannel babble (from .wav),
    then does SNR scaling (babble vs. clean @ ref mic), adds white noise, and returns tensors.

    __getitem__ returns:
        y : (N, M)  torch.float32  # mixture
        d : (N, M)  torch.float32  # clean (with noise-only head)
        b : (N, M)  torch.float32  # babble (scaled)
        v : (N, M)  torch.float32  # white noise (scaled)
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        csv_path = cfg.dataset.df_path_train if mode == "train" else cfg.dataset.df_path_test

        import pandas as pd
        self.df = pd.read_csv(csv_path)

        # Clean .mat location/prefix
        self.clean_dir = cfg.paths.train_path if mode == "train" else cfg.paths.test_path
        self.clean_prefix = getattr(cfg, "clean_prefix", "clean_example_")

        # Babble folder (precomputed multichannel wavs)
        self.babble_dir = cfg.paths.babble_noise_train_path if mode == "train" else cfg.paths.babble_noise_test_path

        # Cache frequently used params
        self.fs  = int(getattr(cfg.params, "fs", 16000))
        self.T   = float(getattr(cfg.params, "T", 4.0))
        self.N   = int(self.fs * self.T)
        # mic_ref in your code is 1-based (train loop uses mic_ref-1 for slicing)
        self.mic_ref = int(getattr(cfg.params, "mic_ref", 4))
        self.noise_only_time = float(getattr(cfg.params, "noise_only_time", 0.5))

        self.SNR_pink_db = random.uniform(5.0, 12.0) if mode == "train" else 5
        self.SNR_white_db  = float(getattr(cfg.params, "SNR_white_db", 30.0))

    def __len__(self):
        return len(self.df)

    def _mat_path(self, idx: int) -> str:
        return os.path.join(self.clean_dir, f"{self.clean_prefix}{idx:07d}.mat")

    def _babble_path(self, idx: int) -> str:
        # matches precompute naming: babble_{idx:07d}.wav
        return os.path.join(self.babble_dir, f"babble_{idx:07d}.wav")

    def __getitem__(self, idx: int):

        fs = self.fs
        N  = self.N
        row = self.df.iloc[idx]
        # ----------------- load clean (.mat) -----------------
        mat_path = self._mat_path(idx)
        data = sio.loadmat(mat_path)

        ck = self.cfg.params.ck
        d_full = np.asarray(data["clean"], dtype=np.float32)  # (samples, M) or (M, samples)
        
        
        # time crop/pad to N

        M = d_full.shape[1]
        
        # build clean with noise-only head
        nlead = int(fs * self.noise_only_time)
        
        d = np.vstack([np.zeros((nlead, M), dtype=np.float32), d_full]).astype(np.float32, copy=False)



        all_rirs = np.asarray(data["all_rirs"], dtype=np.float32)
        # all_RTFs =  true_rtf_from_all_rirs(all_rirs, WIN_LENGTH, REF_MIC)



        # ----------------- load babble (multichannel .wav) -----------------
                # ---------- Pink noise ----------
        def generate_pink(N):
            # Simple IIR filter approximation of pink noise
            b = [0.049922035, 0.050612699, 0.050979644, 0.048882058]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            white = np.random.randn(N)
            pink = signal.lfilter(b, a, white)
            return pink
                # ---------- Geometry/RIR for two stationary noises (order=0) ----------
        L = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
        beta = [float(row["beta"])] * 6
        n_taps = int(row["n"])
        mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)

        pink = generate_pink(N)

        noise1_pos = [float(row["noise1_x"]), float(row["noise1_y"]), float(row["noise1_z"])]
        # noise2_pos = [float(row["noise2_x"]), float(row["noise2_y"]), float(row["noise2_z"])]

        a_n_1 = np.array(rir_generat(ck,fs, mic_positions, noise1_pos, L, beta, n_taps, order=0)).T  # (M, taps)
        n1 = np.stack([signal.lfilter(a_n_1[m], [1.0], pink) for m in range(M)], axis=1)[:N, :]
        # # scale pink noise vs. d at ref mic
        # SNR_pink_db = 10.0
        # p_power = np.sum(pink[:, ref] ** 2) + 1e-10
        # scale_p = np.sqrt(d_power * 10 ** (-SNR_pink_db / 10.0) / p_power)
        # pink *= scale_p



        # ----------------- SNR scaling (babble vs d @ ref mic) -----------------
        ref_idx = max(0, self.mic_ref - 1)  # 1-based -> 0-based
        d_power = float(np.sum(d[:, ref_idx] ** 2) + 1e-12)
        n_power = float(np.sum(n1[:, ref_idx] ** 2) + 1e-12)
        scale_n = np.sqrt(d_power * 10.0 ** (-self.SNR_pink_db / 10.0) / n_power).astype(np.float32)
        n1 *= scale_n

        # ----------------- white noise -----------------
        v = np.random.randn(N, M).astype(np.float32)
        v_power = float(np.sum(v[:, ref_idx] ** 2) + 1e-12)
        scale_v = np.sqrt(d_power * 10.0 ** (-self.SNR_white_db / 10.0) / v_power).astype(np.float32)
        v *= scale_v

        # ----------------- final mix & peak normalize -----------------
        y = d + n1 + v
        peak = float(np.max(np.abs(y)) + 1e-12)
        y /= peak; d /= peak; n1 /= peak; v /= peak



        return y, d, n1, v, all_rirs
    



# ----------------- main dataset -----------------
class GeneratedData_Two_Static_Speakers_Babble(Dataset):
    """
    Loads clean multichannel speech (from .mat) and precomputed multichannel babble (from .wav),
    then does SNR scaling (babble vs. clean @ ref mic), adds white noise, and returns tensors.

    __getitem__ returns:
        y : (N, M)  torch.float32  # mixture
        d : (N, M)  torch.float32  # clean (with noise-only head)
        b : (N, M)  torch.float32  # babble (scaled)
        v : (N, M)  torch.float32  # white noise (scaled)
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        csv_path = cfg.dataset.df_path_train if mode == "train" else cfg.dataset.df_path_test

        import pandas as pd
        self.df = pd.read_csv(csv_path)

        # Clean .mat location/prefix
        self.clean_dir = cfg.paths.train_path if mode == "train" else cfg.paths.test_path
        self.clean_prefix = getattr(cfg, "clean_prefix", "clean_example_")

        # Babble folder (precomputed multichannel wavs)
        self.babble_dir = cfg.paths.babble_noise_train_path if mode == "train" else cfg.paths.babble_noise_test_path

        # Cache frequently used params
        self.fs  = int(getattr(cfg.params, "fs", 16000))
        self.T   = float(getattr(cfg.params, "T", 4.0))
        self.N   = int(self.fs * self.T)
        # mic_ref in your code is 1-based (train loop uses mic_ref-1 for slicing)
        self.mic_ref = int(getattr(cfg.params, "mic_ref", 4))
        self.noise_only_time = float(getattr(cfg.params, "noise_only_time", 0.5))
        self.SNR_babble_db = random.uniform(3.0, 12.0)  if mode == "train" else 6 # random.uniform(5.0, 8.0)
        self.SNR_white_db  = float(getattr(cfg.params, "SNR_white_db", 30.0))

    def __len__(self):
        return len(self.df)

    def _mat_path(self, idx: int) -> str:
        return os.path.join(self.clean_dir, f"{self.clean_prefix}{idx:07d}.mat")

    def _babble_path(self, idx: int) -> str:
        # matches precompute naming: babble_{idx:07d}.wav
        return os.path.join(self.babble_dir, f"babble_{idx:07d}.wav")

    def __getitem__(self, idx: int):
        C_K = self.cfg.params.ck
        row1 = self.df.iloc[idx]
        row2 = self.df.iloc[(idx + 1) % len(self.df)]  # second speaker uses next row’s wav

        fs = self.fs
        T_sec = self.T
        N = int(fs * T_sec)
        noise_only_samples = int(self.noise_only_time * fs)

        # ==========================================================
        # ---------- ROOM GEOMETRY FROM ROW1 ONLY -------------------
        # ==========================================================
        L = [float(row1["room_x"]), float(row1["room_y"]), float(row1["room_z"])]
        beta = [float(row1["beta"])] * 6
        n_taps = int(row1["n"])
        mic_positions = np.array(ast.literal_eval(row1["mic_positions"]), dtype=np.float64)
        M = mic_positions.shape[0]

        # speaker 1 at its own location
        spk1_pos = [
            float(row1["speaker_start_x"]),
            float(row1["speaker_start_y"]),
            float(row1["speaker_start_z"])
        ]

        # speaker 2 at noise1 location (relative angle)
        spk2_pos = [
            float(row1["noise1_x"]),
            float(row1["noise1_y"]),
            float(row1["noise1_z"])
        ]

        # ==========================================================
        # ---------- LOAD WAV OF SPEAKER 1 -------------------------
        # ==========================================================
        wav1 = row1["speaker_path"]
        OLD_PREFIX = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"
        NEW_PREFIX = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"
        if isinstance(wav1, str) and wav1.startswith(OLD_PREFIX):
            wav1 = wav1.replace(OLD_PREFIX, NEW_PREFIX)

        x1, fs_file = sf.read(wav1)
        # if x1.ndim > 1:
        #     x1 = x1[:, 0]
        # if fs_file != fs:
        #     raise RuntimeError("fs mismatch")

        # ensure long enough
        if len(x1) < N:
            reps = math.ceil(N / len(x1))
            x1 = np.tile(x1, reps)

        # find speech onset
        thr1 = 0.02 * np.max(np.abs(x1) + 1e-12)
        start1 = 0
        for n in range(len(x1)):
            if abs(x1[n]) > thr1:
                start1 = n
                break

        # cut & align speaker 1
        x1_cut = x1[start1 : start1 + (N - noise_only_samples)]
        if len(x1_cut) < (N - noise_only_samples):
            reps = math.ceil((N - noise_only_samples) / len(x1_cut))
            x1_cut = np.tile(x1_cut, reps)[:(N - noise_only_samples)]

        x1_final = np.concatenate([
            np.zeros(noise_only_samples, dtype=np.float32),
            x1_cut[:(N - noise_only_samples)].astype(np.float32)
        ], axis=0)

        # ==========================================================
        # ---------- LOAD WAV OF SPEAKER 2 -------------------------
        # ==========================================================
        wav2 = row2["speaker_path"]
        if isinstance(wav2, str) and wav2.startswith(OLD_PREFIX):
            wav2 = wav2.replace(OLD_PREFIX, NEW_PREFIX)

        x2, fs_file2 = sf.read(wav2)
        if x2.ndim > 1:
            x2 = x2[:, 0]
        if fs_file2 != fs:
            raise RuntimeError("fs mismatch")

        if len(x2) < N:
            reps = math.ceil(N / len(x2))
            x2 = np.tile(x2, reps)

        thr2 = 0.02 * np.max(np.abs(x2) + 1e-12)
        start2 = 0
        for n in range(len(x2)):
            if abs(x2[n]) > thr2:
                start2 = n
                break

        x2_cut = x2[start2 : start2 + (N - noise_only_samples)]
        if len(x2_cut) < (N - noise_only_samples):
            reps = math.ceil((N - noise_only_samples) / len(x2_cut))
            x2_cut = np.tile(x2_cut, reps)[:(N - noise_only_samples)]

        x2_final = np.concatenate([
            np.zeros(noise_only_samples, dtype=np.float32),
            x2_cut[:(N - noise_only_samples)].astype(np.float32)
        ], axis=0)

        # ==========================================================
        # ---------- RIR GENERATION FOR TWO STATIC SPEAKERS --------
        # ==========================================================
        h1 = np.array(
            rir_generat(
                C_K, fs, mic_positions, spk1_pos, L, beta, n_taps
            )
        ).T.astype(np.float32)  # (M, n_taps)

        h2 = np.array(
            rir_generat(
                C_K, fs, mic_positions, spk2_pos, L, beta, n_taps
            )
        ).T.astype(np.float32)  # (M, n_taps)

        # all_rirs shape (2, M, n_taps)
        # all_rirs = np.stack([h1, h2], axis=0)

        # ==========================================================
        # ---------- CONVOLVE SPEAKERS TO MULTICHANNEL ------------
        # ==========================================================
        d1 = np.zeros((N, M), dtype=np.float32)
        d2 = np.zeros((N, M), dtype=np.float32)

        for m in range(M):
            d1[:, m] = signal.lfilter(h1[m], [1.0], x1_final)[:N]
            d2[:, m] = signal.lfilter(h2[m], [1.0], x2_final)[:N]

        d = d1 + d2

        # ==========================================================
        # ---------- LOAD BABBLE MULTICHANNEL WAV ------------------
        # ==========================================================
        babble_path = os.path.join(self.babble_dir, f"babble_{idx:07d}.wav")
        babble, sr_b = sf.read(babble_path, always_2d=True)
        babble = babble.astype(np.float32)

        if babble.shape[0] < N:
            reps = math.ceil(N / babble.shape[0])
            babble = np.tile(babble, (reps, 1))

        babble = babble[:N, :M]

        # ==========================================================
        # ---------- SNR SCALE BABBLE ------------------------------
        # ==========================================================
        ref = self.mic_ref - 1
        d_power = float(np.sum(d[:, ref] ** 2) + 1e-12)
        b_power = float(np.sum(babble[:, ref] ** 2) + 1e-12)
        scale_b = np.sqrt(d_power * 10 ** (-self.SNR_babble_db / 10.0) / b_power).astype(np.float32)
        babble *= scale_b

        # ==========================================================
        # ---------- WHITE NOISE -----------------------------------
        # ==========================================================
        v = np.random.randn(N, M).astype(np.float32)
        v_power = float(np.sum(v[:, ref] ** 2) + 1e-12)
        scale_v = np.sqrt(d_power * 10 ** (-self.SNR_white_db / 10.0) / v_power).astype(np.float32)
        v *= scale_v

        # ==========================================================
        # ---------- FINAL MIX & NORMALIZE -------------------------
        # ==========================================================
        y = d + babble + v
        peak = float(np.max(np.abs(y)) + 1e-12)

        y      = (y      / peak).astype(np.float32)
        d      = (d      / peak).astype(np.float32)
        d1     = (d1     / peak).astype(np.float32)
        d2     = (d2     / peak).astype(np.float32)
        babble = (babble / peak).astype(np.float32)
        v      = (v      / peak).astype(np.float32)


        return y, d, d1, d2, babble, v, h1, h2
    