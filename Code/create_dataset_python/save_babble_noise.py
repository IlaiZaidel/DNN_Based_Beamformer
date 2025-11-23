#!/usr/bin/env python3
# precompute_babble_fast_mp.py
import os, ast
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import rir_generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from time import perf_counter
from math import radians
from tqdm import tqdm

# ---------- CONFIG ----------
CSV_PATH     = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
OUT_DIR      = "/dsi/gannot-lab1/datasets/Ilai_data/Babble_Noise/Test"
FS           = 16000
T            = 4.0
N            = int(FS * T)
K_TALKERS    = 20
RING_RADIUS  = 2.5
RIR_ORDER    = 2        # 0 or 1 for speed
MAX_TAPS     = 2048      # cap length
MARGIN       = 0.30
POOL_SIZE    = 3000      # driver segments in RAM
WORKERS      = 16        # CPU workers
SKIP_IF_EXISTS = True    # resume
SAVE_SUBTYPE = "FLOAT"   # "FLOAT" fast write; "PCM_16" smaller files
PRINT_EVERY  = 200

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- GLOBALS for workers (set in initializer) ----------
DRIVER_POOL = None  # np.ndarray [POOL_SIZE, N] float32
DRIVER_POOL_SZ = 0

def _init_worker(pool):
    """Initializer to set globals in each worker process."""
    global DRIVER_POOL, DRIVER_POOL_SZ
    # Limit BLAS threads per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    DRIVER_POOL = pool
    DRIVER_POOL_SZ = 0 if pool is None else pool.shape[0]

# ---------- helpers ----------
def _clip_radius_to_room(cx, cy, theta_deg, Lx, Ly, margin=0.30):
    th = radians(theta_deg % 360.0)
    dx, dy = np.cos(th), np.sin(th)
    cand = []
    for xw in (margin, Lx - margin):
        if abs(dx) > 1e-12:
            t = (xw - cx) / dx
            if t > 0:
                y_at = cy + t * dy
                if (margin - 1e-9) <= y_at <= (Ly - margin + 1e-9):
                    cand.append(t)
    for yw in (margin, Ly - margin):
        if abs(dy) > 1e-12:
            t = (yw - cy) / dy
            if t > 0:
                x_at = cx + t * dx
                if (margin - 1e-9) <= x_at <= (Lx - margin + 1e-9):
                    cand.append(t)
    return 0.0 if not cand else max(0.0, min(cand) - 1e-3)

def _ring_positions(center_xyz, room_dim, K, radius, margin=0.30, jitter_deg=4.0):
    cx, cy, cz = center_xyz
    Lx, Ly, Lz = room_dim
    base = np.linspace(0, 360, K, endpoint=False)
    if jitter_deg > 0:
        base = base + np.random.uniform(-jitter_deg, +jitter_deg, size=K)
    pos = []
    for a in base:
        r_max = _clip_radius_to_room(cx, cy, a, Lx, Ly, margin)
        r_use = min(radius, r_max)
        x = cx + r_use * np.cos(np.radians(a))
        y = cy + r_use * np.sin(np.radians(a))
        z = np.random.uniform(1.2, 1.9)
        pos.append([float(x), float(y), float(z)])
    return pos

def _take_random_segment(x, sr, fs, N, rng):
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != fs: x = signal.resample_poly(x, fs, sr)
    if len(x) < N:
        reps = int(np.ceil(N/len(x))); x = np.tile(x, reps)
    start = rng.integers(0, len(x)-N+1)
    seg = x[start:start+N].astype(np.float32, copy=False)
    seg /= (np.sqrt(np.mean(seg**2) + 1e-12)).astype(np.float32)
    return seg

def _preload_driver_pool(paths, pool_size, fs, N):
    """Load drivers once into RAM (float32)."""
    print(f"Preloading {pool_size} drivers into RAM...")
    rng = np.random.default_rng(12345)

    # if pool_size > len(paths), sample with replacement
    replace = pool_size > len(paths)
    idxs = rng.choice(len(paths), size=pool_size, replace=replace)

    drivers = []
    for idx in tqdm(idxs, total=pool_size, desc="Loading drivers"):
        p = paths[idx]
        x, sr = sf.read(p, always_2d=False)
        drivers.append(_take_random_segment(x, sr, fs, N, rng))
    return np.stack(drivers, axis=0)  # [pool, N]
# ---------- worker ----------
def _worker_row(idx_row, row_dict):
    """Render one babble file for CSV row = idx_row."""
    # local rng per row
    rng = np.random.default_rng(idx_row)

    # parse geometry
    L = [float(row_dict["room_x"]), float(row_dict["room_y"]), float(row_dict["room_z"])]
    beta = [float(row_dict["beta"])] * 6
    n_taps_csv = int(row_dict["n"])
    n_taps = int(min(MAX_TAPS, n_taps_csv))
    mic_positions = np.array(ast.literal_eval(row_dict["mic_positions"]), dtype=np.float64)
    M = mic_positions.shape[0]
    mic_center = (float(row_dict["mic_x"]), float(row_dict["mic_y"]), float(row_dict["mic_z"]))

    # positions
    pos = _ring_positions(mic_center, L, K_TALKERS, RING_RADIUS, margin=MARGIN)

    # RIRs
    a_b = []
    for s in pos:
        h_full = rir_generator.generate(
            c=343.0, fs=FS, r=mic_positions, s=np.array(s),
            L=L, beta=beta, nsample=n_taps,
            mtype=rir_generator.mtype.omnidirectional,
            order=RIR_ORDER, dim=3, hp_filter=True
        )  # (nsample, M)
        h = np.asarray(h_full[:n_taps, :], dtype=np.float32, order="C").T  # (M, taps)
        a_b.append(h)

    # pick K drivers from global pool
    assert DRIVER_POOL is not None and DRIVER_POOL.shape[1] == N
    pool_idx = rng.choice(DRIVER_POOL.shape[0], size=K_TALKERS, replace=False)
    gains = (10.0**(rng.uniform(-6.0, +3.0, size=K_TALKERS)/20.0)).astype(np.float32)

    # render babble
    babble = np.zeros((N, M), dtype=np.float32)
    for j in range(K_TALKERS):
        drv = (DRIVER_POOL[pool_idx[j]] * gains[j]).astype(np.float32, copy=False)
        hkm = a_b[j]
        n_k = np.stack([signal.lfilter(hkm[m], [1.0], drv, axis=0)[:N] for m in range(M)], axis=1)
        babble += n_k

    # save
    out_path = os.path.join(OUT_DIR, f"babble_{idx_row:07d}.wav")
    sf.write(out_path, babble, FS, subtype=SAVE_SUBTYPE)
    return out_path

# ---------- main ----------
def main():
    df = pd.read_csv(CSV_PATH)

    # limit test to 100 samples
    TEST_COUNT = 100
    df = df.iloc[:TEST_COUNT].reset_index(drop=True)

    paths = df["speaker_path"].tolist()

    # preload pool in parent  (cap and allow replace if needed)
    pool_size = min(POOL_SIZE, len(paths))
    driver_pool = _preload_driver_pool(paths, pool_size, FS, N)  # [pool_size, N], float32

    # rows to do (resume)
    indices = list(range(len(df)))
    if SKIP_IF_EXISTS:
        todo = []
        for i in indices:
            if not os.path.exists(os.path.join(OUT_DIR, f"babble_{i:07d}.wav")):
                todo.append(i)
    else:
        todo = indices

    if not todo:
        print("Nothing to do — all files exist.")
        return

    print(f"Total rows: {len(df)} | To generate: {len(todo)} | Workers: {WORKERS}")

    worker_fn = partial(_worker_row)

    t0 = perf_counter()
    done = 0
    fails = 0

    # IMPORTANT: pass driver_pool via initializer so workers see it as a global (COW on Linux)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(driver_pool,)) as ex:
        futures = {ex.submit(worker_fn, i, df.iloc[i].to_dict()): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as e:
                fails += 1
                print(f"[fail] row {i}: {repr(e)}")
            if (done + fails) % PRINT_EVERY == 0:
                dt = perf_counter() - t0
                rate = (done + fails) / max(dt, 1e-9)
                print(f"[{done+fails}/{len(todo)}] ok={done} fail={fails} | {rate:.2f}/s | {dt/60:.1f} min")

    dt = perf_counter() - t0
    rate = (done + fails) / max(dt, 1e-9)
    print(f"Done. ok={done}, fail={fails}, time={dt/60:.1f} min, avg={dt/max(1, done):.3f} s/file, rate={rate:.2f}/s")

if __name__ == "__main__":
    main()
