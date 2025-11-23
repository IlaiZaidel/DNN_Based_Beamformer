import os
import random
import soundfile as sf
import numpy as np

# Folder with your babble files
BABBLE_DIR = "/dsi/gannot-lab1/datasets/Ilai_data/Babble_Noise/Train"
OUT_DIR  = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python"
# Pick one random babble file
all_files = [f for f in os.listdir(BABBLE_DIR) if f.endswith(".wav")]
chosen = random.choice(all_files)
wav_path = os.path.join(BABBLE_DIR, chosen)

# Load full multichannel babble
audio, sr = sf.read(wav_path)

# If it's already mono, just duplicate
if audio.ndim == 1:
    stereo = np.stack([audio, audio], axis=1)
else:
    first = audio[:, 0]
    last  = audio[:, -1]
    stereo = np.stack([first, last], axis=1)

# Save 10 seconds as stereo preview
out_path = os.path.join(OUT_DIR, "babble_preview_stereo.wav")
sf.write(out_path, stereo[:10*sr], sr, subtype="PCM_16")

print(f"Saved stereo preview: {out_path} (L=ch0, R=ch{audio.shape[1]-1}, source={chosen})")
