import sys
# sys.path.insert(0, '../../src')
import sofa
from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.signal import lfilter
import random
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, Optional
import math
from tqdm import tqdm

sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))


class SOFA_HRTF_db:
    def __init__(self,path='',final_fs=16000,nfft=1024):
        self.final_fs = final_fs
        self.nfft=nfft
        self._load_sofa_path(path)
        self._get_positions()
    
    def _load_sofa_path(self,path):
        #loading the HRTF.sofa file
        try:
            self.hrtf_obj = sofa.Database.open(path)
            self.db_fs = int(self.hrtf_obj.Data.SamplingRate.get_values()[0])
        except:
            if not Path(path).exists():
                raise FileNotFoundError(f"The HRTF.sofa path: '{path}' does not exist.")
            return
        
    def get_positions(self):
        return self.H,self.azelevs
    def _get_positions(self):
        # Summarazing all the available hrtf by azimuth and elevation, assuming R is equal for the whole dataset
        pos = self.hrtf_obj.Source.Position.get_values(system="spherical", angle_unit="degree") #az,elev,r
        self.az_list = np.unique(pos[:,0])
        self.elev_list = np.unique(pos[:,1])
        self.R = np.unique(pos[:,2])[0]
        self.lookup_table = pd.DataFrame(index = self.az_list,columns=self.elev_list )
        self.hs = []
        self.azelevs = []
        for idx,p in enumerate(pos):
            self.lookup_table.at[p[0],p[1]] = idx
            h = np.zeros([self.hrtf_obj.Dimensions.N,2])
            h[:,0] = self.hrtf_obj.Data.IR.get_values(indices={"M":idx, "R":0, "E":0})
            h[:,1] = self.hrtf_obj.Data.IR.get_values(indices={"M":idx, "R":1, "E":0})
            h = self.decimate_rir(h.T,self.db_fs,self.final_fs)
            h = torch.tensor(h)
            self.hs.append(h)
            self.azelevs.append(torch.tensor([p[0],p[1]]))
        
        # stack + fft
        h = torch.stack(self.hs)
        max_per_sample = h.abs().amax(dim=(1, 2), keepdim=True)  # shape [865, 1, 1]
        max_per_sample[max_per_sample == 0] = 1e-8
        h_norm = h / max_per_sample
        # h_norm = h/(h.abs().max(dim=-1, keepdim=True).values)
        H = torch.fft.rfft(h_norm,n=self.nfft,dim=-1)
        self.H = torch.concat((H.real,H.imag),dim=1).real
        self.azelevs = torch.stack(self.azelevs)

    def impzest(self,input_signal, output_signal):
        if np.any(np.isnan(input_signal)) or np.any(np.isnan(output_signal)):
            print("Warning: NaN values detected in signals")
        rir_est, _ = signal.deconvolve(output_signal, input_signal)
        return rir_est

    def decimate_rir(self,rir_orig, fs,target_fs=16000):
        rir_orig = np.nan_to_num(rir_orig)
        t = np.linspace(0, 3, int(3 * fs), endpoint=False)
        exc = signal.chirp(t, f0=20, f1=20000, t1=3, method='logarithmic')
        rir_dec = np.zeros_like(rir_orig)
        for i in range(rir_orig.shape[0]):
            exc = np.append(exc, np.zeros(len(rir_orig[i,:]) + 1))

            # Convolve with RIR
            sig = signal.fftconvolve(exc, rir_orig[i,:], mode='full')

            # Decimate the signal
            dec = fs // np.gcd(target_fs, fs)
            exc_dec = signal.decimate(exc, dec)
            exc_dec[np.abs(exc_dec) < 1e-10] = 0

            sig_dec = signal.decimate(sig, dec)
            res = self.impzest(exc_dec, sig_dec)
            rir_dec[i,:len(res)] = res
        return rir_dec
        

class SOFA_HRTF_wrapper:
    def __init__(self,path=''):
        self._load_sofa_path(path)
        self._get_positions()
        
    def _load_sofa_path(self,path):
        #loading the HRTF.sofa file
        try:
            self.hrtf_obj = sofa.Database.open(path)
            self.target_fs = int(self.hrtf_obj.Data.SamplingRate.get_values()[0])
        except:
            if not Path(path).exists():
                raise FileNotFoundError(f"The HRTF.sofa path: '{path}' does not exist.")
    
    def impzest(self,input_signal, output_signal):
        if np.any(np.isnan(input_signal)) or np.any(np.isnan(output_signal)):
            print("Warning: NaN values detected in signals")
        rir_est, _ = signal.deconvolve(output_signal, input_signal)
        return rir_est

    def decimate_rir(self,rir_orig, fs,target_fs=16000):
        rir_orig = np.nan_to_num(rir_orig)
        t = np.linspace(0, 3, int(3 * fs), endpoint=False)
        exc = signal.chirp(t, f0=20, f1=20000, t1=3, method='logarithmic')
        rir_dec = np.zeros_like(rir_orig)
        for i in range(rir_orig.shape[0]):
            exc = np.append(exc, np.zeros(len(rir_orig[i,:]) + 1))

            # Convolve with RIR
            sig = signal.fftconvolve(exc, rir_orig[i,:], mode='full')

            # Decimate the signal
            dec = fs // np.gcd(target_fs, fs)
            exc_dec = signal.decimate(exc, dec)
            exc_dec[np.abs(exc_dec) < 1e-10] = 0

            sig_dec = signal.decimate(sig, dec)
            res = self.impzest(exc_dec, sig_dec)
            rir_dec[i,:len(res)] = res
        return rir_dec
    
    def change_target_fs(self,target_fs):
        if target_fs != self.target_fs:
            self.target_fs = target_fs
        
    def _conv_hrtf_mesurment(self,wav,fs,measurement):
        #convolving a vector 'wav' of time domain audio with an HRTF 
        h = np.zeros([self.hrtf_obj.Dimensions.N,2])
        h[:,0] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":0, "E":0})
        h[:,1] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":1, "E":0})
        if self.target_fs != fs:
            h = self.decimate_rir(h.T,self.target_fs,fs).T
        rend_L = signal.fftconvolve(wav.squeeze(),h[:,0])
        rend_R = signal.fftconvolve(wav.squeeze(),h[:,1])
        stereo_audio = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        stereo_audio = stereo_audio / np.max(np.abs(stereo_audio), axis=0, keepdims=True)
        return stereo_audio,h
    
    def _get_positions(self):
        # Summarazing all the available hrtf by azimuth and elevation, assuming R is equal for the whole dataset
        pos = self.hrtf_obj.Source.Position.get_values(system="spherical", angle_unit="degree") #az,elev,r
        self.az_list = np.unique(pos[:,0])
        self.elev_list = np.unique(pos[:,1])
        self.R = np.unique(pos[:,2])[0]
        self.lookup_table = pd.DataFrame(index = self.az_list,columns=self.elev_list )
        self.azelevs=[]
        for idx,p in enumerate(pos):
            self.lookup_table.at[p[0],p[1]] = idx
            self.azelevs.append(torch.tensor([p[0],p[1]]))
        self.azelevs = torch.stack(self.azelevs)

    def nearest_valid_idx(self, az, elev):
        """
        Find the nearest non-NaN entry in self.lookup_table to (az, elev)
        using Euclidean distance in (az, elev) space.
        Works even if the table contains object or non-numeric entries.
        Returns: (az_idx, elev_idx, az_val, elev_val, measurement)
        """
        az_arr   = np.asarray(self.az_list, dtype=float)
        elev_arr = np.asarray(self.elev_list, dtype=float)

        # Convert table to numeric mask (invalids become NaN)
        vals = self.lookup_table.values
        try:
            vals_float = vals.astype(float)
        except (ValueError, TypeError):
            # Coerce to float, non-numeric become NaN
            vals_float = pd.DataFrame(self.lookup_table).apply(pd.to_numeric, errors='coerce').to_numpy()

        mask = ~np.isnan(vals_float)
        if not mask.any():
            raise ValueError("Lookup table has no valid (non-NaN) entries.")

        # Build coordinate grids
        AZ, EL = np.meshgrid(az_arr, elev_arr, indexing='ij')

        # Compute distances
        d2 = (AZ - az)**2 + (EL - elev)**2
        d2[~mask] = np.inf  # ignore invalid cells

        # Find min distance
        i, j = np.unravel_index(np.argmin(d2), d2.shape)

        az_val = az_arr[i]
        elev_val = elev_arr[j]
        measurement = self.lookup_table.iloc[i, j]  # keep original (could be array or impulse response)

        return i, j, az_val, elev_val, measurement

    
    def conv_hrtf_pos(self,wav,fs,az,elev):
        _,_,az,elev,measurement = self.nearest_valid_idx(az,elev)
        res,h = self._conv_hrtf_mesurment(wav,fs,measurement)
        return torch.tensor(res).T,torch.tensor(h).T,int(az),int(elev)
    
    def conv_file(self,wav_path,az,elev):
        #loads wav file and conv with desierd hrtf, returns stereo audio
        wav,fs = sf.read(wav_path)
        res,hrtf,az,elev =  self.conv_hrtf_pos(wav,fs,az,elev)
        return res,fs,hrtf,az,elev

    def get_hrtf(self,az,elev,fs=16000):
        # measurement = self.lookup_table.iloc[az, elev]      
        _, _, az_val, elev_val, measurement = self.nearest_valid_idx(float(az), float(elev))
        measurement = int(measurement)   
        h = np.zeros([self.hrtf_obj.Dimensions.N,2])
        h[:,0] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":0, "E":0})
        h[:,1] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":1, "E":0})
        if self.target_fs != fs:
            h = self.decimate_rir(h.T,self.target_fs,fs).T
        return h
    
def test_hrtf(path,name):
    hrtf_ir = SOFA_HRTF_wrapper(path)
    azs = {}
    tmp_az = 69
    for i in range(5):
        d = {'az':tmp_az}
        azs[str(i)] = d
        tmp_az -=45
        if tmp_az ==-45:
            tmp_az = 315

    for i,d in azs.items():
        s,fs,hrtf,az,elev = hrtf_ir.conv_file('/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/5105/28240/5105-28240-0021.wav',d['az'],0)
        out_dir = Path(f'/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA/{name}')
        out_dir.mkdir(exist_ok=True)
        sf.write(out_dir/f'hrtf_az_{az}_elev_{elev}.wav',s.T,fs)

    
if __name__=="__main__":
    # # root = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/hrtf_testing/sofa_downloads')
    # # names = root.glob('**/')
    # # for name in tqdm(list(names)):
    # #     try:
    # #         path = list(name.glob('*.sofa'))[0]
    # #         test_hrtf(path,name.stem)
    # #         print(f'{name.stem} Done!')
    # #     except:
    # #         print(f'{name.stem} Failed!')
    # #         continue
    # hrtf_db = SOFA_HRTF_db('/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/riec_full/RIEC_hrir_subject_001.sofa')
    import pickle

    sofas = {
             'riec_full':'/home/dsi/ilaiz/DNN_Based_Beamformer/Code/SOFA/RIEC_hrir_subject_001.sofa'
            }    
    d={}
    for name,p in sofas.items():
        test_hrtf(p,name)
    #     hrtf_ir = SOFA_HRTF_wrapper(p)
    #     azs_elev = {}
    #     for azelev in hrtf_ir.azelevs:        
    #         az = float(azelev[0])
    #         if (0 <= az <= 90) or (270 <= az <= 360):
    #             az = str(az)    
    #             if az in azs_elev.keys():
    #                 azs_elev[az].append(float(azelev[1]))
    #             else:
    #                 azs_elev[az]=[float(azelev[1])]
    #     # print(name,azs_elev.keys())
    #     d[name] = azs_elev
    # with open("sofa_az_elev_lookup.pkl", "wb") as f:
    #     pickle.dump(d, f)

            # azelev['az':1]
    #     # print(hrtf_ir.azelevs)
    #     row_steps = np.diff(hrtf_ir.lookup_table.index)
    #     row_steps = [f"{x:.2f}" for x in row_steps]
    #     col_steps = np.diff(hrtf_ir.lookup_table.columns)
    #     col_steps = [f"{x:.2f}" for x in col_steps]

    #     common_row_step = np.unique(row_steps)
    #     common_col_step = np.unique(col_steps)
    #     d[name] = {'shape':f'{hrtf_ir.lookup_table.shape[0]}X{hrtf_ir.lookup_table.shape[1]}',
    #                'range_az_low':hrtf_ir.lookup_table.index[0],
    #                'range_az_high':hrtf_ir.lookup_table.index[-1],
    #                'common_az_step':common_row_step,
    #                'range_elev_low':hrtf_ir.lookup_table.columns[0],
    #                'range_elev_high':hrtf_ir.lookup_table.columns[-1],
    #                'common_elev_step':common_col_step,
    #                'num_samples':hrtf_ir.lookup_table.count().sum()
    #                }
    # df = pd.DataFrame({name: vals for name, vals in d.items()}) 
    # df.to_csv("/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/sofa_db_stats.csv", index=True,float_format="%.2f")
    # print(df.to_string())