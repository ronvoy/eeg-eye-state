import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# --- Data Loading and Setup ---
try:
    df = pd.read_csv('dataset/eeg_data.csv')
except FileNotFoundError:
    print("Error: eeg_data.csv not found. Please ensure the file is in the current directory.")
    exit()

# List of all EEG feature channels
feature_columns = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# Map eyeDetection to meaningful labels
df['eyeState'] = df['eyeDetection'].map({0: 'Closed', 1: 'Open'})

# --- Directory Setup and Creation ---
directories = {
    'class_balance': 'plots/classbased',
    'histogram_raw': 'plots/histogram/raw',
    'histogram_normalized': 'plots/histogram/normalized',
    'boxplot_with_outliers': 'plots/boxplot/with_outliers',
    'boxplot_without_outliers': 'plots/boxplot/without_outliers',
    'violinplot': 'plots/violinplot',
    'scatterchart': 'plots/scatterchart',
    'correlationheatmap': 'plots/correlationheatmap'
}

for name, dir_path in directories.items():
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory '{dir_path}' is ready.")

# --- Analysis and Plotting ---

# --- Class Balance Diagram / Count Plot ---
plt.figure(figsize=(7, 5))
sns.countplot(x='eyeState', data=df)
plt.title('Class Balance of Eye States')
plt.xlabel('Eye State')
plt.ylabel('Count')
balance_path = os.path.join(directories['class_balance'], 'class_balance_diagram.png')
plt.savefig(balance_path)
plt.close()
print(f"Saved Class Balance Diagram: {balance_path}")

# --- Correlation Heat Map (Bivariate for all features) ---
plt.figure(figsize=(12, 10))
corr_matrix = df[feature_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heat Map of EEG Channels')
heatmap_path = os.path.join(directories['correlationheatmap'], 'correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.close()
print(f"Saved Correlation Heat Map: {heatmap_path}")

# --- Univariate and Grouped Plots for Each Feature (Loop) ---
for channel in feature_columns:
    print(f"--- Analyzing Channel: {channel} ---")

    # --- Histograms (Raw - Without Normalization) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=channel, hue='eyeState', kde=True, bins=50)
    plt.title(f'Raw Amplitude Distribution for Channel {channel}')
    hist_raw_path = os.path.join(directories['histogram_raw'], f'histogram_raw_{channel}.png')
    plt.savefig(hist_raw_path)
    plt.close()
    print(f"Saved Raw Histogram: {hist_raw_path}")
    
    # --- Histograms (Min-Max Normalized) ---
    # Temporarily normalize the current channel data for visualization
    scaler = MinMaxScaler()
    df[f'{channel}_normalized'] = scaler.fit_transform(df[[channel]])
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=f'{channel}_normalized', hue='eyeState', kde=True, bins=50)
    plt.title(f'Min-Max Normalized Amplitude Distribution for Channel {channel}')
    plt.xlabel('Normalized Amplitude (0 to 1)')
    hist_norm_path = os.path.join(directories['histogram_normalized'], f'histogram_normalized_{channel}.png')
    plt.savefig(hist_norm_path)
    plt.close()
    # Remove the temporary normalized column
    df = df.drop(columns=[f'{channel}_normalized'])
    print(f"Saved Normalized Histogram: {hist_norm_path}")

    # --- Box Plots (With Outliers) ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='eyeState', y=channel, data=df)
    plt.title(f'Box Plot (With Outliers) for Channel {channel}')
    boxplot_with_path = os.path.join(directories['boxplot_with_outliers'], f'boxplot_with_outliers_{channel}.png')
    plt.savefig(boxplot_with_path)
    plt.close()
    print(f"Saved Box Plot (With Outliers): {boxplot_with_path}")

    # --- Box Plots (Without Outliers for visualization only) ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='eyeState', y=channel, data=df, showfliers=False) # showfliers=False hides the outlier points
    plt.title(f'Box Plot (Without Outliers) for Channel {channel}')
    boxplot_without_path = os.path.join(directories['boxplot_without_outliers'], f'boxplot_without_outliers_{channel}.png')
    plt.savefig(boxplot_without_path)
    plt.close()
    print(f"Saved Box Plot (Without Outliers): {boxplot_without_path}")
    
    # --- Violin Plots ---
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='eyeState', y=channel, data=df)
    plt.title(f'Violin Plot of Amplitude by Eye State for Channel {channel}')
    violinplot_path = os.path.join(directories['violinplot'], f'violinplot_{channel}.png')
    plt.savefig(violinplot_path)
    plt.close()
    print(f"Saved Violin Plot: {violinplot_path}")

    # --- Scatter Plot (Strip Plot) ---
    plt.figure(figsize=(8, 5))
    sns.stripplot(x='eyeState', y=channel, data=df, jitter=True, alpha=0.1)
    plt.title(f'Scatter (Strip) Plot by Eye State for Channel {channel}')
    scatter_path = os.path.join(directories['scatterchart'], f'scatterplot_{channel}.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved Scatter Plot: {scatter_path}")
    
    print(f"--- Finished analyzing Channel: {channel} ---")

print("All analyses complete. Plots saved in the 'plots' directory.")
