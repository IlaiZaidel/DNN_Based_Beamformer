import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from signal_generator import SignalGenerator

# --- Settings ---
c = 340.0                      # Speed of sound [m/s]
L = [4.0, 5.0, 6.0]            # Room dimensions [m]
beta = [0.2]                   # Reverberation time [s]
nsamples = 1024               # RIR length [samples]
order = 2                     # Reflection order
hop = 32                      # Motion step size
fs = 16000                    # Will be updated from WAV

# --- Load source signal ---
in_signal, fs = sf.read('/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator/female_speech.wav')
if in_signal.ndim > 1:
    in_signal = in_signal[:, 0]  # Use first channel

T = len(in_signal)                      # Total time steps
# T = fs*10
# in_signal = 0.9*np.sin(2*np.pi*880*np.linspace(0,5,T))
# --- Receiver positions [M x 3] ---
rp = np.array([
    [1.5, 2.4, 3.0],
    [1.5, 2.6, 3.0]
])
M = rp.shape[0]
cp = np.mean(rp, axis=0)

# --- Source movement: linear path ---
start = np.array([2.0, 1.0, 3.0])
stop  = np.array([2.0, 3.0, 3.0])
sp_path = np.zeros((T, 3))
rp_path = np.zeros((T, 3, M))

for i in range(0, T, hop):
    alpha = i / T
    sp = start + alpha * (stop - start)
    sp_path[i:i + hop] = sp

    for m in range(M):
        rp_path[i:i + hop, :, m] = rp[m] 
# --- Run signal generator ---
gen = SignalGenerator()
result = gen.generate(
    input_signal=list(in_signal),
    c=c,
    fs=fs,
    r_path=rp_path.tolist(),
    s_path=sp_path.tolist(),
    L=L,
    beta_or_tr=beta,
    nsamples=nsamples,
    mtype="o",
    order=order,
    hp_filter=True
)

# --- Plotting source & mic positions ---
time = np.arange(T) / fs

plt.figure(figsize=(10, 4))
ax = plt.axes(projection='3d')
ax.plot3D(sp_path[:, 0], sp_path[:, 1], sp_path[:, 2], 'r.', label='Source path')
ax.plot3D(rp[:, 0], rp[:, 1], rp[:, 2], 'bx', label='Receivers')
ax.set_xlim(0, L[0])
ax.set_ylim(0, L[1])
ax.set_zlim(0, L[2])
ax.set_title('Source and Receiver Positions')
ax.legend()
plt.tight_layout()

# --- Plot waveforms ---
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, in_signal)
plt.title("Input signal")
plt.xlabel("Time [s]")

plt.subplot(2, 1, 2)
for m in range(len(result.output)):
    plt.plot(time, result.output[m], label=f"Mic {m+1}")
plt.title("Output signal(s)")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig('output.png')

# --- Normalize and write WAV file ---
output = np.array(result.output).T  # [T, M]
peak = np.max(np.abs(output))
if peak > 0:
    output = output / (peak + 1e-9)*0.99  # Normalize + clip-safe
sf.write('output.wav', output, fs,subtype='FLOAT')
