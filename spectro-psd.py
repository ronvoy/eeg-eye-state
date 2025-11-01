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

# List of all EEG feature channels
feature_columns = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# --- Directory Setup ---
psd_dir = 'plots/psd'
spectrogram_dir = 'plots/spectrograms'

# Create directories if they do not exist
os.makedirs(psd_dir, exist_ok=True)
os.makedirs(spectrogram_dir, exist_ok=True)
print(f"Directories '{psd_dir}' and '{spectrogram_dir}' are ready.")

# --- Loop through all channels for analysis and saving ---
for channel in feature_columns:
    print(f"--- Analyzing Channel: {channel} ---")

    # --- 1. PSD Analysis and Saving ---
    # Separate data into eye states
    data_open = df[df['eyeDetection'] == 1][channel]
    data_closed = df[df['eyeDetection'] == 0][channel]

    # Calculate PSD using Welch's method
    freqs_open, psd_open = welch(data_open, sf, nperseg=nperseg)
    freqs_closed, psd_closed = welch(data_closed, sf, nperseg=nperseg)

    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs_open, psd_open, label='Eyes Open', color='blue')
    plt.semilogy(freqs_closed, psd_closed, label='Eyes Closed', color='red')
    plt.xlim(0, 30)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (µV²/Hz)')
    plt.title(f'PSD Analysis for Channel {channel}')
    plt.legend()
    plt.grid(True)

    # Save the PSD plot to the directory
    psd_filename = os.path.join(psd_dir, f'psd_{channel}.png')
    plt.savefig(psd_filename)
    plt.close()  # Close the plot to free memory
    print(f"Saved PSD plot: {psd_filename}")

    # --- 2. Spectrogram Analysis and Saving ---
    data = df[channel].values

    # Calculate the spectrogram
    freqs, times, Sxx = spectrogram(data, fs=sf, noverlap=nperseg // 2)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylim(0, 30)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram for Channel {channel}')
    plt.colorbar(label='Power/Frequency (dB/Hz)')

    # Add horizontal lines for brain wave frequency ranges
    plt.axhline(y=12, color='red', linestyle='--', linewidth=1, label='Beta (12-30 Hz)')
    plt.axhline(y=8, color='blue', linestyle='--', linewidth=1, label='Alpha (8-12 Hz)')
    plt.axhline(y=4, color='black', linestyle='--', linewidth=1, label='Theta (4-8 Hz)')
    plt.axhline(y=0.5, color='white', linestyle='--', linewidth=1, label='Delta (0.5-4 Hz)')
    plt.legend(loc='upper right')

    # Save the spectrogram plot to the directory
    spectrogram_filename = os.path.join(spectrogram_dir, f'spectrogram_{channel}.png')
    plt.savefig(spectrogram_filename)
    plt.close()  # Close the plot to free memory
    print(f"Saved Spectrogram plot: {spectrogram_filename}")

    print(f"--- Finished analyzing Channel: {channel} ---")

print("All analyses complete. Plots saved in the 'plots' directory.")
