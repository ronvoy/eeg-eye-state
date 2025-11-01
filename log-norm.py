import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('dataset/eeg_data.csv')

# Sampling rate (Hz)
fs = 128
time = [i / fs for i in range(len(df))]

# Select the AF3 channel
af3 = df['AF3']

# Normalize amplitude using logarithmic scale
# Add a small constant to avoid log(0)
log_af3 = np.log10(af3 - af3.min() + 1)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(time, log_af3, color='darkorange', linewidth=1)
plt.title('EEG Channel AF3 (Log-Normalized Amplitude)')
plt.xlabel('Time (seconds)')
plt.ylabel('log₁₀(Normalized Amplitude)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
