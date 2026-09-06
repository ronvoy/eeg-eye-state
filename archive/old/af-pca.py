import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os # Import os for file path checks

# Define file names
INPUT_FILE = 'dataset/eeg_data.csv'
OUTPUT_FILE = 'dataset/eeg_data_pca.csv'
FEATURES_TO_REDUCE = ['AF3', 'AF4']

print(f"Starting PCA process: Loading {INPUT_FILE}...")

# --- 1. LOAD THE DATA FROM CSV ---
try:
    # Assuming 'eeg_data.csv' is in the current working directory
    eeg_df = pd.read_csv(INPUT_FILE)
    print(f"✅ Data loaded successfully. Shape: {eeg_df.shape}")
except FileNotFoundError:
    print(f"❌ Error: '{INPUT_FILE}' not found. Please ensure the file is in the correct directory.")
    exit() # Exit the script if the file is not found

# --- 2. VALIDATE FEATURES AND PREPARE DATA ---
if not all(col in eeg_df.columns for col in FEATURES_TO_REDUCE):
    print(f"❌ Error: Data does not contain both '{FEATURES_TO_REDUCE[0]}' and '{FEATURES_TO_REDUCE[1]}' columns. Please check your CSV file headers.")
    exit()

X_redundant = eeg_df[FEATURES_TO_REDUCE]

# --- 3. STANDARDIZE AND APPLY PCA ---

# Standardize the data (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_redundant)

# Apply PCA, keeping only the first component (PC1) which captures the shared variance
# Since we know r=0.86, PC1 is sufficient.
pca = PCA(n_components=1)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the new PCA feature
af_pca_df = pd.DataFrame(data = principal_components,
                         columns = ['AF_PCA_Component'])

# --- 4. CREATE THE FINAL ML-READY DATASET ---

# Drop the original redundant features from the main DataFrame
eeg_df_non_redundant = eeg_df.drop(columns=FEATURES_TO_REDUCE)

# Concatenate the new PCA feature with the non-redundant original features
# Use reset_index() to ensure proper alignment during concatenation
final_ml_df = pd.concat([eeg_df_non_redundant.reset_index(drop=True),
                         af_pca_df],
                        axis=1)

# --- 5. SAVE THE NEW CSV FILE ---
final_ml_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- PCA Summary ---")
print(f"Explained Variance Ratio of AF_PCA_Component (PC1): {pca.explained_variance_ratio_[0]:.4f}")
print(f"Original shape: {eeg_df.shape}")
print(f"New shape: {final_ml_df.shape}")
print(f"New feature columns (excluding target): {final_ml_df.shape[1] - 1} (12 original + 1 PCA)")
print(f"\n✨ Successfully created and saved the new dataset to: **{OUTPUT_FILE}**")