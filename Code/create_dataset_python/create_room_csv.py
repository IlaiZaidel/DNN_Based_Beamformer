import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from pathlib import Path
from glob import glob
from random import shuffle
import wave
import rir_generator

# Define output paths
#OUTPUT_CSV = Path("/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters.csv") # Train
OUTPUT_CSV = Path("/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_test.csv") # Test
# Ensure the output directory exists
os.makedirs(OUTPUT_CSV.parent, exist_ok=True)

# Constants
C_K = 343  # Speed of sound (m/s)
FS = 16000  # Sample rate
MICS = 8  # Number of microphones
ROOM_HEIGHT = 3  # Fixed room height
MIC_REF = 4  # Reference microphone index
MAX_SAMPLES = 100 # Number of generated samples
T = 4
# Columns for CSV
columns = ["speaker_path",
    "beta","n", "room_x", "room_y", "room_z",
    "mic_x", "mic_y", "mic_z",
    "speaker_x", "speaker_y", "speaker_z",
    "noise1_x", "noise1_y", "noise1_z",
    "noise2_x", "noise2_y", "noise2_z",
    "angleOrientation", "angle_x", "angle_n1", "angle_n2", "radius", "mic_positions"
]
df = pd.DataFrame(columns=columns)

# Randomized Room Creation Function
def create_room():
    """
    Generates a random room with a microphone array, speaker, and two noise sources.
    Returns a dictionary of room parameters.
    """
    # Reverberation time
    beta = random.choice(np.arange(0.3, 0.55, 0.05))  # Random reverberation time
    n = int(FS * beta)  # Length of impulse response

    # Room dimensions
    x_lim = random.uniform(6, 9)  # [6,9] meters
    y_lim = random.uniform(6, 9)  # [6,9] meters
    room_dim = [x_lim, y_lim, ROOM_HEIGHT]

    # Random orientation angle
    angleOrientation = random.choice(np.arange(-45, 46, 1))


    # Microphone array position before rotation
    mic_height = random.uniform(1, 1.5)
    mic_x = random.uniform(2, x_lim - 2)
    mic_y = random.uniform(2, y_lim - 2)
    center = np.array([mic_x, mic_y])

    # Define mic positions relative to center
    mic_offsets = np.array([
        [-0.17, 0], [-0.12, 0], [-0.07, 0], [-0.04, 0],
        [0.04, 0], [0.07, 0], [0.12, 0], [0.17, 0]
    ])

    # Rotation matrix
    theta = np.radians(angleOrientation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotate mic positions
    mic_rotated = mic_offsets @ rotation_matrix.T + center

    # Convert to final mic positions
    mic_positions = [[x, y, mic_height] for x, y in mic_rotated]





    # Speaker position
    # lim_radius = min(mic_x - 0.5, mic_y - 0.5, x_lim - mic_x - 0.5, y_lim - mic_y - 0.5)
    # lim_radius = min(max(lim_radius, 1), 1.5)  # Clamped between 1m and 2.2m
    radius = random.uniform(1, 1.5) # 1.5 is because length of array is 0.34 and mic center is x_lim - 2 tops!
    angle_x = random.randint(0, 180)
    speaker_x = mic_x + radius * np.cos(np.radians(angle_x + angleOrientation))
    speaker_y = mic_y + radius * np.sin(np.radians(angle_x + angleOrientation))
    speaker_z = random.uniform(1, 1.5)  # Fixed speaker height

    # Noise source positions
    angle_n1, angle_n2 = randomize_noise_angles(angle_x)
    noise1_x = mic_x + radius * np.cos(np.radians(angle_n1 + angleOrientation))
    noise1_y = mic_y + radius * np.sin(np.radians(angle_n1 + angleOrientation))
    noise1_z = random.uniform(1.2, 1.9)

    noise2_x = mic_x + radius * np.cos(np.radians(angle_n2 + angleOrientation))
    noise2_y = mic_y + radius * np.sin(np.radians(angle_n2 + angleOrientation))
    noise2_z = random.uniform(1.2, 1.9)



# Save parameters
    room_data = {"speaker_path": 'Hello',
        "beta": beta,"n":n, "room_x": x_lim, "room_y": y_lim, "room_z": ROOM_HEIGHT,
        "mic_x": mic_x, "mic_y": mic_y, "mic_z": mic_height,
        "speaker_x": speaker_x, "speaker_y": speaker_y, "speaker_z": speaker_z,
        "noise1_x": noise1_x, "noise1_y": noise1_y, "noise1_z": noise1_z,
        "noise2_x": noise2_x, "noise2_y": noise2_y, "noise2_z": noise2_z,
        "angleOrientation": angleOrientation, "angle_x": angle_x, "angle_n1": angle_n1, "angle_n2": angle_n2, "radius": radius,
        "mic_positions": mic_positions
    }
    return room_data





import random

def randomize_noise_angles(source_angle, min_angle=10):
    """
    Ensures that two noise sources are at least `min_angle` degrees apart 
    from each other and from the main source (speaker).

    Parameters:
        source_angle (int): The angle of the main source (speaker).
        min_angle (int): The minimum angular separation between all sources.
        
    Returns:
        tuple: Two noise source angles (noise_angle_1, noise_angle_2).
    """
    # Create a list of valid angles that are at least `min_angle` degrees away from the source
    valid_angles = [angle for angle in range(181) if abs(angle - source_angle) >= min_angle]

    # Select the first noise angle
    noise_angle_1 = random.choice(valid_angles)

    # Create a new valid list excluding angles too close to both the source and noise_angle_1
    valid_angles = [angle for angle in valid_angles if abs(angle - noise_angle_1) >= min_angle]

    # Select the second noise angle
    noise_angle_2 = random.choice(valid_angles)

    return noise_angle_1, noise_angle_2





### Constructing the CSV:

speakers ={}
root_path = Path('/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/') # Train
speakers_list = glob(str(root_path/'**'))




num_generated = 0  # Counter for successful samples

with tqdm(total=MAX_SAMPLES, desc="Generating Room Data") as pbar:
    while num_generated < MAX_SAMPLES:
        while True:  # Keep trying until a valid speaker with WAV files is found
            random_speaker = random.choice(speakers_list)
            l = glob(random_speaker + '/**/*.wav')  # List of wav files
            if l:  # Ensure the list is not empty
                break  

        while True:  # Keep trying until a valid WAV file is found
            random_wav_path = random.choice(l)
            import soundfile as sf

            try:
                num_frames = sf.info(random_wav_path).frames
                if num_frames > T * FS:
                    break  # Found a valid file, exit the loop
            except RuntimeError:
                print(f"Skipping incompatible WAV file: {random_wav_path}")

        # Create room and store data
        room_data = create_room()
        room_data["speaker_path"] = random_wav_path  # Add wav_path
        df = pd.concat([df, pd.DataFrame([room_data])], ignore_index=True)

        num_generated += 1  # Increment only after successful generation
        pbar.update(1)  # Update progress bar

# Ensure we have exactly MAX_SAMPLES rows
df = df.iloc[:MAX_SAMPLES]  # Trim extra rows if any

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Room parameters saved to: {OUTPUT_CSV}")
