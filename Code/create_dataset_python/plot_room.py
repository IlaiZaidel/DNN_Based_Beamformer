import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Load the CSV
csv_path = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters.csv"
df = pd.read_csv(csv_path)

# Choose a row to plot
index = 3019 # Change this to plot a different sample
row = df.iloc[index]

# Room dimensions
room_x = row['room_x']
room_y = row['room_y']

# Microphone positions (X, Y only)
mic_positions = ast.literal_eval(row['mic_positions'])
mic_positions_xy = [(x, y) for x, y, _ in mic_positions]

# Speaker and noise sources (X, Y only)
speaker_xy = [row['speaker_x'], row['speaker_y']]
noise1_xy = [row['noise1_x'], row['noise1_y']]
noise2_xy = [row['noise2_x'], row['noise2_y']]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f"Room Setup (Top-Down): Sample #{index}")

# Plot room boundary
ax.set_xlim(0, room_x)
ax.set_ylim(0, room_y)
ax.set_aspect('equal')
ax.grid(True)

# Plot microphones
mic_x, mic_y = zip(*mic_positions_xy)
ax.scatter(mic_x, mic_y, c='blue', label='Microphones', s=60)

# Plot speaker
ax.scatter(*speaker_xy, c='red', label='Speaker', s=80, marker='^')

# Plot noise sources
ax.scatter(*noise1_xy, c='green', label='Noise Source 1', s=70, marker='x')
ax.scatter(*noise2_xy, c='orange', label='Noise Source 2', s=70, marker='x')

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()

# Save plot to file
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"room_topdown_sample_{index}.png")
plt.tight_layout()
plt.savefig(out_path)
print(f"2D plot saved to: {out_path}")
