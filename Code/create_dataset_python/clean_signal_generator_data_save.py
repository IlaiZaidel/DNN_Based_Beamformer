#!/usr/bin/env python3
import os
import sys
import math
import ast
import re
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import savemat
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from time import perf_counter

# ===================== USER SETTINGS =====================
CSV_PATH = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
OUT_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Tracking_Signal_Gen_Data/Test_Rev_Signal_Gen_with_rir"
OLD_PREFIX = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/"
NEW_PREFIX = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/"
COUNT = 100                # max to process this run
START_IDX =  0             # if None -> auto-resume from last saved + 1
PREFIX = "clean_example_"    # filename prefix
WORKERS = 16                 # CPU workers
NSAMPLES = 1024              # RIR length in samples
ORDER = 2                    # 0 = direct only; -1 = full order
HOP = 32                     # AIR refresh hop in samples
M_TYPE = "o"                 # mic type
SPEED_OF_SOUND = 343.0       # m/s
HP_FILTER = True
DIM = 3
NOISE_ONLY_TIME=0.5
eps_ratio = 0.05
SIGGEN_SO = "/home/dsi/ilaiz/DNN_Based_Beamformer/Code/signal_generator_rtf_output/signal_generator.cpython-310-x86_64-linux-gnu.so"
SIGGEN_DIR = os.path.dirname(SIGGEN_SO)

# Timelapse printing cadence
PRINT_EVERY_N = 100          # print every N completions
PRINT_EVERY_SEC = 30         # or at least every 30s
# ========================================================

# make the compiled module importable
if SIGGEN_DIR not in sys.path:
    sys.path.insert(0, SIGGEN_DIR)
from signal_generator import SignalGenerator  # compiled .so

IDX_RE = re.compile(rf"^{re.escape(PREFIX)}(\d+)\.mat$")

def list_empty_indices(out_dir: str, prefix: str):
    """Return indices of files that exist but are 0 bytes."""
    empty = []
    try:
        for n in os.listdir(out_dir):
            if n.startswith(prefix) and n.endswith(".mat"):
                path = os.path.join(out_dir, n)
                try:
                    if os.path.getsize(path) == 0:
                        idx = int(n[len(prefix):-4])
                        empty.append(idx)
                except OSError:
                    pass
    except FileNotFoundError:
        pass
    return sorted(set(empty))

def detect_resume_start(out_dir: str, prefix: str) -> int:
    """Return next index to generate based on existing files."""
    try:
        names = os.listdir(out_dir)
    except FileNotFoundError:
        return 0
    found = []
    for n in names:
        m = IDX_RE.match(n)
        if m:
            found.append(int(m.group(1)))
    return (max(found) + 1) if found else 0


def list_missing_indices(out_dir: str, prefix: str, total: int, start_from: int = 0):
    existing = set()
    try:
        for n in os.listdir(out_dir):
            if n.startswith(prefix) and n.endswith(".mat"):
                try:
                    existing.add(int(n[len(prefix):-4]))
                except:
                    pass
    except FileNotFoundError:
        pass
    return [i for i in range(start_from, total) if i not in existing]


def _worker_process(
    row_tuple,
    out_dir,
    prefix,
    start_idx,
    nsamples,
    order,
    hop,
    speed_of_sound,
    mtype,
    dim,
    hp_filter,
):
    # keep each worker single-threaded for BLAS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    csv_row_idx, row = row_tuple  # absolute CSV row index (0-based)
    # --- Load CSV ---
    wav_path = row["speaker_path"]
    OLD_PREFIX = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/"
    NEW_PREFIX = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/"
    if isinstance(wav_path, str) and wav_path.startswith(OLD_PREFIX):
        wav_path = wav_path.replace(OLD_PREFIX, NEW_PREFIX)
    # --- parse row ---
    
    fs = int(row.get("fs", 16000))
    T_sec = float(row.get("T", 4.0))
    L = [float(row["room_x"]), float(row["room_y"]), float(row["room_z"])]
    beta = [float(row["beta"])/3] * 6
    mic_positions = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
    M = mic_positions.shape[0]

    # --- load and prepare speech ---
    x, fs_file = sf.read(wav_path)
    if x.ndim > 1:
        x = x[:, 0]
    if fs_file != fs:
        raise RuntimeError(f"fs mismatch (wav:{fs_file}, csv:{fs}) @ row {csv_row_idx}")

    N = int(T_sec * fs)
    noise_only_samples = int(NOISE_ONLY_TIME * fs)
    if len(x) < N:
        reps = math.ceil(N / max(1, len(x)))
        x = np.tile(x, reps)
    thr = eps_ratio * np.max(np.abs(x))
    start_idx = 0
    for n in range(len(x)):
        if abs(x[n]) > thr:
            start_idx = n
            break

    # enforce minimum drop if start_idx too small
    start_idx = min(start_idx, int(0.1 * fs)) #max
    
    x = x[start_idx:N-noise_only_samples+start_idx].astype(np.float64)

    # --- build source/receiver paths ---
    start = np.array([float(row["speaker_start_x"]),
                      float(row["speaker_start_y"]),
                      float(row["speaker_start_z"])], dtype=np.float64)
    stop  = np.array([float(row["speaker_stop_x"]),
                      float(row["speaker_stop_y"]),
                      float(row["speaker_stop_z"])], dtype=np.float64)
    
    # ONLY FOR NOW
    
    # stop = start.copy()



    sp_path = np.zeros((N-noise_only_samples, 3), dtype=np.float64)
    rp_path = np.zeros((N-noise_only_samples, 3, M), dtype=np.float64)

    center = mic_positions[:, :2].mean(axis=0)          # (cx, cy)

    r0   = start[:2] - center
    r1   = stop[:2]  - center
    theta0 = np.arctan2(r0[1], r0[0])                   # start angle
    theta1 = np.arctan2(r1[1], r1[0])                   # end angle (from STOP)

    # choose radius: either CSV value or the start radius to keep a perfect circle
    r = float(row["radius"])

    # (optional) go along the shortest arc:
    dtheta = ((theta1 - theta0 + np.pi) % (2*np.pi)) - np.pi

    for i in range(0, N-noise_only_samples, HOP):
        a = i / max(1, N-noise_only_samples-1)                             # 0..1
        theta = theta0 + a * dtheta                     # interpolate angle
        x_pos = center[0] + r * np.cos(theta)
        y_pos = center[1] + r * np.sin(theta)
        z = start[2]                                    # keep Z fixed (or lerp if needed)
        end = min(i + HOP, N-noise_only_samples)
        sp_path[i:end] = (x_pos, y_pos, z)
        for m in range(M):
            rp_path[i:end, :, m] = mic_positions[m]

   
    # --- run signal generator ---
    gen = SignalGenerator()
    result = gen.generate(
        list(x),                # input_signal
        speed_of_sound,         # c
        fs,                     # fs
        rp_path.tolist(),       # r_path: [T][3][M]
        sp_path.tolist(),       # s_path: [T][3]
        L,                      # room dims
        beta,                   # beta or [TR]
        nsamples,               # nsamples
        mtype,                  # mtype
        order,                  # order
        dim,                    # dimension
        [],                     # orientation
        hp_filter               # hp_filter
    )

    clean = np.array(result.output, dtype=np.float64).T  # (N, M)
    all_rirs = np.array(result.all_rirs, dtype=np.float64)
    # print(all_rirs.shape)
    # --- save .mat (index == CSV row index) ---
    out_name = f"{prefix}{csv_row_idx:07d}.mat"
    out_path = os.path.join(out_dir, out_name)
    savemat(out_path, {"clean": clean, "all_rirs": all_rirs})
    return (csv_row_idx, out_name, out_path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    csv_total = len(df)


    
    # Auto-resume if START_IDX is None
    resume_from = detect_resume_start(OUT_DIR, PREFIX) if START_IDX is None else START_IDX
    if resume_from >= csv_total:
        print(f"Nothing to do: resume_from={resume_from} >= csv_total={csv_total}")
        return

    # Compute how many to process this run
    remaining = csv_total - resume_from
    total = min(COUNT, remaining)

    # Build absolute CSV rows slice [resume_from, resume_from+total)
    target_rows = list(range(resume_from, resume_from + total))

    # Skip any rows whose output already exists (idempotent)
    rows_to_do = []
    preexisting = 0
    for i in target_rows:
        out_name = f"{PREFIX}{i:07d}.mat"
        if os.path.exists(os.path.join(OUT_DIR, out_name)):
            preexisting += 1
            continue
        rows_to_do.append((i, df.iloc[i].to_dict()))



    # MANUAL_RANGE = range(6600, 6900)

    # # Only regenerate ones that are empty files
    # rows_to_do = []
    # for i in MANUAL_RANGE:
    #     f = os.path.join(OUT_DIR, f"{PREFIX}{i:07d}.mat")
    #     if os.path.exists(f) and os.path.getsize(f) == 0:
    #         rows_to_do.append((i, df.iloc[i].to_dict()))

    # === custom override: manually fix missing .mat files ===
    # missing_indices = [
    #     481, 485, 760, 1250, 1665, 1847, 1945, 2185, 2697, 3308, 3357, 3814,
    #     4096, 5110, 12184, 12604, 12898, 13023, 13300, 14900, 15449, 15508,
    #     15601, 15772, 16068, 16383, 16391, 16397, 16858, 17587, 17870, 17987,
    #     19155, 19393, 19636, 19752
    # ]

    # rows_to_do = [(i, df.iloc[i].to_dict()) for i in missing_indices
    #             if not os.path.exists(os.path.join(OUT_DIR, f"{PREFIX}{i:07d}.mat"))]
    print(f"Will regenerate {len(rows_to_do)} missing .mat files: {', '.join(map(str, [r[0] for r in rows_to_do]))}")


    if not rows_to_do:
        print("All target outputs already exist. Nothing to do.")
        return

    print(f"CSV rows total: {csv_total}")
    print(f"Resuming from: {resume_from}")
    print(f"Will process  : {len(rows_to_do)} rows (skipped {preexisting} existing files)")
    print(f"Using {WORKERS} CPU workers")

    worker_fn = partial(
        _worker_process,
        out_dir=OUT_DIR,
        prefix=PREFIX,
        start_idx=resume_from,   # not used for naming now; kept for API symmetry
        nsamples=NSAMPLES,
        order=ORDER,
        hop=HOP,
        speed_of_sound=SPEED_OF_SOUND,
        mtype=M_TYPE,
        dim=DIM,
        hp_filter=HP_FILTER,
    )

    t0 = perf_counter()
    last_print = t0
    ok_cnt = 0
    fails = []
    manifest_rows = []   # {"csv_row": i, "filename": name, "path": path}

    # For global progress: already-done before this run
    # Count existing files with indices < csv_total
    existing_all = 0
    for n in os.listdir(OUT_DIR):
        m = IDX_RE.match(n)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < csv_total:
                existing_all += 1

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(worker_fn, row): row[0] for row in rows_to_do}
        for fut in as_completed(futures):
            csv_row_idx = futures[fut]
            try:
                r_idx, out_name, out_path = fut.result()
                ok_cnt += 1
                manifest_rows.append({"csv_row": r_idx, "filename": out_name, "path": out_path})
            except Exception as e:
                err = repr(e)
                fails.append({"csv_row": csv_row_idx, "error": repr(e)})
                if len(fails) <= 10:  # show first 10
                    print(f"[FAIL {csv_row_idx}] {err}", flush=True)
            # Timelapse / progress
            now = perf_counter()
            elapsed = now - t0
            need_print = (ok_cnt % PRINT_EVERY_N == 0) or ((now - last_print) >= PRINT_EVERY_SEC) or (ok_cnt == len(rows_to_do))
            if need_print:
                global_done = min(csv_total, existing_all + ok_cnt)  # conservative
                rate = ok_cnt / elapsed if elapsed > 0 else 0.0
                remaining_this_run = len(rows_to_do) - ok_cnt
                eta = remaining_this_run / rate if rate > 0 else float('inf')
                pct = 100.0 * global_done / csv_total
                print(f"[{global_done:6d}/{csv_total}] {pct:6.2f}%  |  elapsed {elapsed:7.1f}s  |  rate {rate:6.2f}/s  |  ETA {eta:7.1f}s  |  fails {len(fails)}")
                last_print = now

    # Append/Write manifest
    manifest_path = os.path.join(OUT_DIR, f"{PREFIX}manifest.csv")
    new_df = pd.DataFrame(manifest_rows).sort_values("csv_row")
    if os.path.exists(manifest_path):
        old_df = pd.read_csv(manifest_path)
        merged = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["csv_row"], keep="last")
        merged.sort_values("csv_row").to_csv(manifest_path, index=False)
    else:
        new_df.to_csv(manifest_path, index=False)

    # Failures (if any)
    if fails:
        fail_path = os.path.join(OUT_DIR, f"{PREFIX}failures.csv")
        pd.DataFrame(fails).sort_values("csv_row").to_csv(fail_path, index=False)
        print(f"Completed with {len(fails)} failures. See: {fail_path}")

    dt = perf_counter() - t0
    rate = ok_cnt / dt if dt > 0 else 0.0
    print(f"Done. Generated: {ok_cnt} new files | Time: {dt:.1f}s | Avg: {dt/max(1, ok_cnt):.3f}s/file | Rate: {rate:.2f}/s")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
