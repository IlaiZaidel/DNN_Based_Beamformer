import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from glob import glob
import soundfile as sf
from tqdm import tqdm

# ==== CONFIG ====
CSV_IN = Path("/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_train.csv")
CSV_OUT = CSV_IN.with_name("room_parameters_tracking_train_speechFromTrain.csv")

# root of *train* speech (adjust to your actual LibriSpeech layout)
TRAIN_ROOT = Path("/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Train/")
FS = 16000
T = 4  # sec
MIN_FRAMES = T * FS

random.seed(0)  # for reproducibility
np.random.seed(0)

print("Loading CSV...")
df = pd.read_csv(CSV_IN)

print("Collecting training wav files...")
# You can adjust the pattern depending on your folder structure
train_wavs = glob(str(TRAIN_ROOT / "**/*.wav"), recursive=True)
if not train_wavs:
    raise RuntimeError(f"No wav files found under {TRAIN_ROOT}")

print(f"Found {len(train_wavs)} wav files in train set.")

def get_random_long_enough_wav():
    while True:
        wav_path = random.choice(train_wavs)
        try:
            info = sf.info(wav_path)
        except RuntimeError:
            continue
        if info.frames >= MIN_FRAMES:
            return wav_path

new_paths = []
print("Assigning new train wavs to each row...")
for _ in tqdm(range(len(df))):
    new_paths.append(get_random_long_enough_wav())

df["speaker_path"] = new_paths

df.to_csv(CSV_OUT, index=False)
print(f"Saved updated CSV with train speech to: {CSV_OUT}")
