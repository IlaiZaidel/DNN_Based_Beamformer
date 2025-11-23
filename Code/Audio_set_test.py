import os, shutil, random

# src_dir = "/dsi/gannot-lab1/datasets/AudioSet_noise/train/Mechanical fan"
src_dir = "/dsi/gannot-lab1/datasets/whamr/wav16k/min/tt/noise"
dst_dir = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/AudioSet_copies"
os.makedirs(dst_dir, exist_ok=True)

# list all wav files (case-insensitive)
files = [f for f in os.listdir(src_dir) if f.lower().endswith(".wav")]
if not files:
    raise RuntimeError(f"No .wav files found in {src_dir}")

# pick one at random
src_file = random.choice(files)
src_path = os.path.join(src_dir, src_file)

# make sure destination filename won’t overwrite
dst_file = f"copy_{src_file}"
dst_path = os.path.join(dst_dir, dst_file)

# copy
shutil.copy(src_path, dst_path)   # copy() keeps permissions/metadata; copyfile() also works
print(f"Copied: {src_path} → {dst_path}")
