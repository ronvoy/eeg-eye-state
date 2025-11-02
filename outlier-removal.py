import pandas as pd
import numpy as np
import os

# Paths
input_path = "dataset/eeg_data_og.csv"
output_path = "dataset/eeg_data.csv"

# Ensure output folder exists
os.makedirs("dataset", exist_ok=True)

# Load EEG dataset
df = pd.read_csv(input_path)

# Exclude target variable if exists
feature_cols = [col for col in df.columns if col != "eyeDetection"]
cleaned_df = df.copy()

for col in feature_cols:
    # Compute IQR thresholds
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - 1.5 * IQR
    upper_bound_iqr = Q3 + 1.5 * IQR

    # Compute sigma-based thresholds
    mean = df[col].mean()
    std = df[col].std()
    lower_bound_sigma = mean - 5 * std  # 5σ rule
    upper_bound_sigma = mean + 5 * std

    # Combine both filters
    lower_bound = max(lower_bound_iqr, lower_bound_sigma)
    upper_bound = min(upper_bound_iqr, upper_bound_sigma)

    # Filter outliers
    cleaned_df = cleaned_df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Save the cleaned data
cleaned_df.to_csv(output_path, index=False)

print(f"✅ Cleaned EEG data saved to: {output_path}")
print(f"Original samples: {len(df)}, After cleaning: {len(cleaned_df)}")
