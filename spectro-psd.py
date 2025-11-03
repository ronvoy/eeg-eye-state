import pandas as pd
import numpy as np
from scipy.signal import welch, spectrogram
import matplotlib.pyplot as plt
import os

# --- Data Loading and Setup ---
try:
    df = pd.read_csv('dataset/eeg_data.csv')
except FileNotFoundError:
    print("Error: eeg_data.csv not found. Please ensure the file is in the current directory.")
    exit()

sf = 128  # Sampling frequency (Hz)
nperseg = 256  # Segment size for analysis

feature_columns = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# --- Directory Setup ---
psd_dir = 'plots/psd'
spectrogram_dir = 'plots/spectrograms'
eyes_open_dir = os.path.join(spectrogram_dir, 'eyes_open')
eyes_closed_dir = os.path.join(spectrogram_dir, 'eyes_close')

os.makedirs(psd_dir, exist_ok=True)
os.makedirs(eyes_open_dir, exist_ok=True)
os.makedirs(eyes_closed_dir, exist_ok=True)
print(f"Directories ready: {psd_dir}, {eyes_open_dir}, {eyes_closed_dir}")

# --- Loop through all channels ---
for channel in feature_columns:
    print(f"--- Analyzing Channel: {channel} ---")

    # --- 1. PSD Analysis and Saving ---
    data_open = df[df['eyeDetection'] == 1][channel]
    data_closed = df[df['eyeDetection'] == 0][channel]

    freqs_open, psd_open = welch(data_open, sf, nperseg=nperseg)
    freqs_closed, psd_closed = welch(data_closed, sf, nperseg=nperseg)

    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs_open, psd_open, label='Eyes Open', color='blue')
    plt.semilogy(freqs_closed, psd_closed, label='Eyes Closed', color='red')
    plt.xlim(0, 30)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (µV²/Hz)')
    plt.title(f'PSD Analysis for Channel {channel}')
    plt.legend()
    plt.grid(True)
    psd_filename = os.path.join(psd_dir, f'psd_{channel}.png')
    plt.savefig(psd_filename)
    plt.close()
    print(f"Saved PSD plot: {psd_filename}")

    # --- 2. Spectrogram Analysis and Saving (separate for eyes open/closed) ---
    for state, data, out_dir in zip(['eyes_open', 'eyes_closed'],
                                     [data_open, data_closed],
                                     [eyes_open_dir, eyes_closed_dir]):
        freqs, times, Sxx = spectrogram(data, fs=sf, noverlap=nperseg // 2)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylim(0, 30)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectrogram ({state}) for Channel {channel}')
        plt.colorbar(label='Power/Frequency (dB/Hz)')

        # Brain wave lines
        plt.axhline(y=12, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=8, color='blue', linestyle='--', linewidth=1)
        plt.axhline(y=4, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0.5, color='white', linestyle='--', linewidth=1)

        spec_filename = os.path.join(out_dir, f'{state}_{channel}.png')
        plt.savefig(spec_filename)
        plt.close()
        print(f"Saved Spectrogram: {spec_filename}")

    print(f"--- Finished analyzing Channel: {channel} ---")

print("All analyses complete. Spectrograms stored separately for eyes open/closed.")
