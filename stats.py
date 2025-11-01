import pandas as pd
import os
import numpy as np
from scipy import stats

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Load EEG dataset
input_path = "dataset/eeg_data.csv"
df = pd.read_csv(input_path)

# Drop the target variable
data = df.drop(columns=["eyeDetection"], errors="ignore")

# Compute descriptive statistics
stats_data = {
    "mean": data.mean(),
    "median": data.median(),
    "mode": data.mode().iloc[0],
    "min": data.min(),
    "max": data.max(),
    "std_dev": data.std(),
    "25%_quantile": data.quantile(0.25),
    "50%_quantile": data.quantile(0.5),
    "75%_quantile": data.quantile(0.75),
}

# Combine all into one DataFrame
stats_df = pd.DataFrame(stats_data)

# Save to CSV
output_path = "output/stats.csv"
stats_df.to_csv(output_path)

print(f"Descriptive statistics saved to: {output_path}")
