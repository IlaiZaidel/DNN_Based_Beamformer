import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from pathlib import Path
from glob import glob
import soundfile as sf
import rir_generator

# === Configuration ===
OUTPUT_CSV = Path("/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_train.csv")  # Train or Test
os.makedirs(OUTPUT_CSV.parent, exist_ok=True)

C_K = 343
FS = 16000
MICS = 8
ROOM_HEIGHT = 3
MAX_SAMPLES = 20000
T = 4  # duration in seconds

# === CSV Columns ===
columns = [
    "speaker_path", "beta", "n", "room_x", "room_y", "room_z",
    "mic_x", "mic_y", "mic_z",
    "speaker_start_x", "speaker_start_y", "speaker_start_z",
    "speaker_stop_x", "speaker_stop_y", "speaker_stop_z",
    "noise1_x", "noise1_y", "noise1_z",
    "noise2_x", "noise2_y", "noise2_z",
    "angleOrientation", "angle_x", "angle_n1", "angle_n2", "radius", "mic_positions", "dynamic"
]

df = pd.DataFrame(columns=columns)

# === Helper ===


def randomize_noise_angles(source_angle, min_angle=10):
    valid_angles = [a for a in range(181) if abs(a - source_angle) >= min_angle]
    noise_angle_1 = random.choice(valid_angles)
    valid_angles = [a for a in valid_angles if abs(a - noise_angle_1) >= min_angle]
    noise_angle_2 = random.choice(valid_angles)
    return noise_angle_1, noise_angle_2

# === Main Room Creation ===
def create_room(dynamic=True):
    beta = random.choice(np.arange(0.3, 0.55, 0.05))
    n = int(FS * beta)
    x_lim, y_lim = random.uniform(6, 9), random.uniform(6, 9)
    room_dim = [x_lim, y_lim, ROOM_HEIGHT]

    angleOrientation = random.choice(np.arange(-45, 46))
    mic_height = random.uniform(1.0, 1.5)
    mic_x = random.uniform(2, x_lim - 2)
    mic_y = random.uniform(2, y_lim - 2)
    center = np.array([mic_x, mic_y])

    mic_offsets = np.array([
        [-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
        [0.04, 0], [0.07, 0], [0.12, 0], [0.17, 0]
    ])
    theta = np.radians(angleOrientation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mic_rotated = mic_offsets @ rotation_matrix.T + center
    mic_positions = [[x, y, mic_height] for x, y in mic_rotated]

    radius = random.uniform(1, 1.5)
    angle_x = random.randint(0, 180)
    speaker_start_x = mic_x + radius * np.cos(np.radians(angle_x + angleOrientation))
    speaker_start_y = mic_y + radius * np.sin(np.radians(angle_x + angleOrientation))
    speaker_start_z = random.uniform(1.2, 1.8)

    if dynamic:
        while True:
            delta_angle = random.randint(-150, 150)
            if abs(delta_angle) >= 45:  # Ensure non-trivial motion
                break
    #delta_angle = random.choice([-180, 180])
    speaker_stop_x = mic_x + radius * np.cos(np.radians(angle_x + delta_angle + angleOrientation))
    speaker_stop_y = mic_y + radius * np.sin(np.radians(angle_x + delta_angle + angleOrientation))
    speaker_stop_z = speaker_start_z


    angle_n1, angle_n2 = randomize_noise_angles(angle_x)
    noise1_x = mic_x + radius * np.cos(np.radians(angle_n1 + angleOrientation))
    noise1_y = mic_y + radius * np.sin(np.radians(angle_n1 + angleOrientation))
    noise1_z = random.uniform(1.2, 1.9)
    noise2_x = mic_x + radius * np.cos(np.radians(angle_n2 + angleOrientation))
    noise2_y = mic_y + radius * np.sin(np.radians(angle_n2 + angleOrientation))
    noise2_z = random.uniform(1.2, 1.9)

    return {
        "beta": beta, "n": n,
        "room_x": x_lim, "room_y": y_lim, "room_z": ROOM_HEIGHT,
        "mic_x": mic_x, "mic_y": mic_y, "mic_z": mic_height,
        "speaker_start_x": speaker_start_x, "speaker_start_y": speaker_start_y, "speaker_start_z": speaker_start_z,
        "speaker_stop_x": speaker_stop_x, "speaker_stop_y": speaker_stop_y, "speaker_stop_z": speaker_stop_z,
        "noise1_x": noise1_x, "noise1_y": noise1_y, "noise1_z": noise1_z,
        "noise2_x": noise2_x, "noise2_y": noise2_y, "noise2_z": noise2_z,
        "angleOrientation": angleOrientation,
        "angle_x": angle_x, "angle_n1": angle_n1, "angle_n2": angle_n2,
        "radius": radius,
        "mic_positions": mic_positions,
        "dynamic": 1 if dynamic else 0
    }

# === Data Creation Loop ===
root_path = Path('/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/')
speakers_list = [p for p in glob(str(root_path/'**')) if os.path.isdir(p)]

num_generated = 0
with tqdm(total=MAX_SAMPLES, desc="Generating Room Data") as pbar:
    while num_generated < MAX_SAMPLES:
        while True:
            random_speaker = random.choice(speakers_list)
            wav_files = glob(random_speaker + '/**/*.wav', recursive=True)
            if wav_files:
                break

        while True:
            wav_path = random.choice(wav_files)
            try:
                num_frames = sf.info(wav_path).frames
                if num_frames > T * FS:
                    break
            except RuntimeError:
                continue

        room_data = create_room(dynamic=True)  # Enable moving speaker
        room_data["speaker_path"] = wav_path
        df = pd.concat([df, pd.DataFrame([room_data])], ignore_index=True)
        num_generated += 1
        pbar.update(1)

df = df.iloc[:MAX_SAMPLES]
df.to_csv(OUTPUT_CSV, index=False)
print(f"Room parameters saved to: {OUTPUT_CSV}")
