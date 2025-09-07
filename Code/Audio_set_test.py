import os, shutil

src_dir = "/dsi/gannot-lab1/datasets/AudioSet_noise/test/Air conditioning"
src_file = "h2XTOVCno2M.wav"  # pick any file from your ls
src_path = os.path.join(src_dir, src_file)

dst_dir = "/home/dsi/ilaiz/AudioSet_copies"
os.makedirs(dst_dir, exist_ok=True)
dst_path = os.path.join(dst_dir, f"copy_{src_file}")

shutil.copyfile(src_path, dst_path)
print(f"Saved: {dst_path}")
