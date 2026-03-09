#!/usr/bin/env python3
"""
EEG Eye State Classification - Complete Analysis Pipeline
Generates a Markdown report with analysis plots.
Usage: python script.py > report.md
Dataset: https://archive.ics.uci.edu/dataset/264/eeg+eye+state
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram as scipy_spectrogram

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False

# =============================================================================
# Configuration
# =============================================================================
PLOT_DIR = "analysis-plots"
DATA_FILE = "dataset/eeg_data_og.csv"
SAMPLING_RATE = 128
FEATURE_COLUMNS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
TARGET = "eyeDetection"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Per UCI: 0 = eye-open, 1 = eye-closed
EYE_MAP = {0: "Open", 1: "Closed"}

HEMI_PAIRS = [
    ("AF3", "AF4"), ("F7", "F8"), ("F3", "F4"),
    ("FC5", "FC6"), ("T7", "T8"), ("P7", "P8"), ("O1", "O2"),
]

FREQ_BANDS = {
    "Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12),
    "Beta": (12, 30), "Gamma": (30, 64),
}
BAND_COLORS = ["#8B0000", "#FF4500", "#FFD700", "#00CED1", "#9370DB"]

ELECTRODE_INFO = [
    ("AF3", "Anterior Frontal Left", "Prefrontal Cortex", "Executive function, attention"),
    ("F7",  "Frontal Left Lateral", "Left Temporal-Frontal", "Language processing"),
    ("F3",  "Frontal Left", "Left Frontal Lobe", "Motor planning, positive affect"),
    ("FC5", "Fronto-Central Left", "Left Motor-Frontal", "Motor preparation"),
    ("T7",  "Temporal Left", "Left Temporal Lobe", "Auditory processing, memory"),
    ("P7",  "Parietal Left", "Left Parietal-Temporal", "Visual-spatial processing"),
    ("O1",  "Occipital Left", "Left Visual Cortex", "Visual processing"),
    ("O2",  "Occipital Right", "Right Visual Cortex", "Visual processing"),
    ("P8",  "Parietal Right", "Right Parietal-Temporal", "Spatial attention"),
    ("T8",  "Temporal Right", "Right Temporal Lobe", "Face / emotion recognition"),
    ("FC6", "Fronto-Central Right", "Right Motor-Frontal", "Motor preparation"),
    ("F4",  "Frontal Right", "Right Frontal Lobe", "Motor planning, negative affect"),
    ("F8",  "Frontal Right Lateral", "Right Temporal-Frontal", "Emotion, social cognition"),
    ("AF4", "Anterior Frontal Right", "Prefrontal Cortex", "Executive function, attention"),
]

os.makedirs(PLOT_DIR, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Markdown helpers
# =============================================================================

def title(text):
    print(f"\n# {text}\n")

def subtitle(text):
    print(f"\n## {text}\n")

def subsubtitle(text):
    print(f"\n### {text}\n")

def md_table(headers, rows):
    print("| " + " | ".join(str(h) for h in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()

def md_text(text):
    print(text)
    print()

def md_image(path, caption=""):
    print(f"![{caption}]({path})")
    print()

def progress(msg):
    print(msg, file=sys.stderr, flush=True)

def save_fig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    return path

# =============================================================================
# Table of Contents
# =============================================================================

def print_toc():
    title("EEG Eye State Classification — Complete Analysis Report")
    md_text("---")
    md_text(
        "**Dataset Source:** [UCI Machine Learning Repository — EEG Eye State]"
        "(https://archive.ics.uci.edu/dataset/264/eeg+eye+state)"
    )
    md_text("---")
    subtitle("Table of Contents")
    toc = """\
1. [Data Description Overview](#1-data-description-overview)
   - 1.1 [Dataset Citation & Source](#11-dataset-citation--source)
   - 1.2 [Dataset Loading](#12-dataset-loading)
   - 1.3 [Variable Classification](#13-variable-classification)
   - 1.4 [Electrode Positions & Significance](#14-electrode-positions--significance)
   - 1.5 [Basic Statistics](#15-basic-statistics)
   - 1.6 [Class Distribution](#16-class-distribution)
2. [Data Imputation](#2-data-imputation)
3. [Data Visualization (Raw Data)](#3-data-visualization-raw-data)
   - 3.1 [Class Balance](#31-class-balance)
   - 3.2 [Correlation Heatmap](#32-correlation-heatmap)
   - 3.3 [Box Plots](#33-box-plots)
   - 3.4 [Histograms](#34-histograms)
   - 3.5 [Violin Plots](#35-violin-plots)
4. [Outlier Removal](#4-outlier-removal)
5. [Data Visualization (After Outlier Removal)](#5-data-visualization-after-outlier-removal)
   - 5.1 [Box Plots Comparison](#51-box-plots-comparison)
   - 5.2 [Histograms After Cleaning](#52-histograms-after-cleaning)
6. [Log-Normalization](#6-log-normalization)
7. [Feature Engineering](#7-feature-engineering)
   - 7.1 [Hemispheric Asymmetry](#71-hemispheric-asymmetry)
   - 7.2 [Global Channel Statistics](#72-global-channel-statistics)
   - 7.3 [Feature Summary](#73-feature-summary)
8. [FFT, Spectrogram and PSD Analysis](#8-fft-spectrogram-and-psd-analysis)
   - 8.1 [FFT Frequency Spectrum](#81-fft-frequency-spectrum)
   - 8.2 [Power Spectral Density (PSD)](#82-power-spectral-density-psd)
   - 8.3 [Spectrogram Analysis](#83-spectrogram-analysis)
9. [Dimensionality Reduction](#9-dimensionality-reduction)
   - 9.1 [PCA](#91-pca)
   - 9.2 [LDA](#92-lda)
   - 9.3 [t-SNE](#93-t-sne)
   - 9.4 [UMAP](#94-umap)
   - 9.5 [Clustering Evaluation](#95-clustering-evaluation)
10. [Machine Learning Classification](#10-machine-learning-classification)
    - 10.1 [Train/Test Split & Class Balance](#101-traintest-split--class-balance)
    - 10.2 [Cross-Validation Results](#102-cross-validation-results)
    - 10.3 [Logistic Regression](#103-logistic-regression)
    - 10.4 [K-Nearest Neighbors](#104-k-nearest-neighbors)
    - 10.5 [Support Vector Machine](#105-support-vector-machine)
    - 10.6 [Random Forest](#106-random-forest)
    - 10.7 [Gradient Boosting](#107-gradient-boosting)
    - 10.8 [Feature Importance](#108-feature-importance)
    - 10.9 [ROC Curves](#109-roc-curves)
    - 10.10 [ML Model Comparison](#1010-ml-model-comparison)
11. [Neural Network Classification](#11-neural-network-classification)
    - 11.1 [1D CNN on Raw EEG](#111-1d-cnn-on-raw-eeg)
    - 11.2 [CNN on Spectrograms](#112-cnn-on-spectrograms)
    - 11.3 [LSTM / RNN](#113-lstm--rnn)
    - 11.4 [Neural Network Comparison](#114-neural-network-comparison)
12. [Final Comparison and Inference](#12-final-comparison-and-inference)
    - 12.1 [Unified Comparison Table](#121-unified-comparison-table)
    - 12.2 [Inference and Recommendation](#122-inference-and-recommendation)"""
    md_text(toc)
    md_text("---")

# =============================================================================
# 1. Data Description Overview
# =============================================================================

def section_data_description(df):
    title("1. Data Description Overview")

    # 1.1 Citation
    subtitle("1.1 Dataset Citation & Source")
    md_text(
        "**Source:** [UCI Machine Learning Repository — EEG Eye State]"
        "(https://archive.ics.uci.edu/dataset/264/eeg+eye+state)"
    )
    md_text(
        "> All data is from one continuous EEG measurement with the Emotiv EEG "
        "Neuroheadset. The duration of the measurement was 117 seconds. The eye "
        "state was detected via a camera during the EEG measurement and added "
        "later manually to the file after analysing the video frames. '1' "
        "indicates the eye-closed and '0' the eye-open state. All values are in "
        "chronological order with the first measured value at the top of the data."
    )

    # 1.2 Dataset Loading
    subtitle("1.2 Dataset Loading")
    md_text(f"The dataset is loaded from `{DATA_FILE}`.")
    md_table(
        ["Property", "Value"],
        [
            ["Samples", df.shape[0]],
            ["Features", df.shape[1] - 1],
            ["Target Column", TARGET],
            ["Sampling Rate", f"{SAMPLING_RATE} Hz"],
            ["Recording Duration", f"{df.shape[0] / SAMPLING_RATE:.1f} seconds"],
        ],
    )

    # 1.3 Variable Classification
    subtitle("1.3 Variable Classification")
    md_text("**Numerical Variables (Continuous):** 14 EEG electrode channels recording voltage in micro-volts (uV).")
    md_table(
        ["Variable", "Type", "Description"],
        [[ch, "Continuous (float64)", f"EEG voltage at {ch} electrode (uV)"]
         for ch in FEATURE_COLUMNS],
    )
    md_text("**Categorical Variable (Target):**")
    md_table(
        ["Variable", "Type", "Values", "Description"],
        [[TARGET, "Binary (int)", "0 = Open, 1 = Closed",
          "Eye state detected via camera during recording"]],
    )

    # 1.4 Electrode Positions
    subtitle("1.4 Electrode Positions & Significance")
    md_text(
        "The Emotiv EPOC headset uses a modified 10-20 international system for electrode "
        "placement. Each electrode captures electrical activity from a specific cortical region."
    )
    md_table(
        ["Electrode", "10-20 Position", "Brain Region", "Functional Significance"],
        [[e[0], e[1], e[2], e[3]] for e in ELECTRODE_INFO],
    )

    # 1.5 Basic Statistics
    subtitle("1.5 Basic Statistics")
    md_text("Descriptive statistics for all 14 EEG channels (uV).")
    desc = df[FEATURE_COLUMNS].describe().T
    headers = ["Channel", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    rows = []
    for ch in FEATURE_COLUMNS:
        r = desc.loc[ch]
        rows.append([
            ch, int(r["count"]),
            f"{r['mean']:.2f}", f"{r['std']:.2f}",
            f"{r['min']:.2f}", f"{r['25%']:.2f}", f"{r['50%']:.2f}",
            f"{r['75%']:.2f}", f"{r['max']:.2f}",
        ])
    md_table(headers, rows)

    # 1.6 Class Distribution
    subtitle("1.6 Class Distribution")
    md_text(f"Distribution of the target variable `{TARGET}` (per UCI: 0 = open, 1 = closed).")
    vc = df[TARGET].value_counts()
    md_table(
        ["Eye State", "Count", "Percentage"],
        [
            ["Open (0)", vc.get(0, 0), f"{vc.get(0, 0) / len(df) * 100:.1f}%"],
            ["Closed (1)", vc.get(1, 0), f"{vc.get(1, 0) / len(df) * 100:.1f}%"],
        ],
    )

# =============================================================================
# 2. Data Imputation
# =============================================================================

def section_data_imputation(df):
    title("2. Data Imputation")
    md_text(
        "Missing values are detected and filled using column-wise **median imputation** "
        "to preserve the statistical properties of each EEG channel."
    )
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        md_text(
            f"**Result:** No missing values detected across any of the "
            f"{len(FEATURE_COLUMNS)} EEG channels. The dataset is complete."
        )
    else:
        md_text(f"**Missing values detected:** {total_missing}")
        rows = [[ch, int(missing[ch])] for ch in FEATURE_COLUMNS if missing[ch] > 0]
        md_table(["Channel", "Missing Count"], rows)
        for col in FEATURE_COLUMNS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        md_text("Missing values filled using **median imputation**.")
    return df

# =============================================================================
# 3. Data Visualization (Raw Data) — full suite
# =============================================================================

def section_data_viz_raw(df):
    title("3. Data Visualization (Raw Data)")
    md_text("Visualizations of the raw EEG data before any preprocessing.")

    # 3.1 Class Balance
    subtitle("3.1 Class Balance")
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[TARGET].value_counts()
    bars = ax.bar(
        ["Open (0)", "Closed (1)"],
        [vc.get(0, 0), vc.get(1, 0)],
        color=["#3498db", "#e74c3c"], edgecolor="black",
    )
    ax.set_title("Class Balance of Eye States")
    ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                str(int(b.get_height())), ha="center", fontweight="bold")
    path = save_fig("class_balance_raw.png")
    md_image(path, "Class Balance")

    # 3.2 Correlation Heatmap
    subtitle("3.2 Correlation Heatmap")
    md_text(
        "The correlation heatmap reveals linear relationships between EEG channels. "
        "Highly correlated channels may carry redundant information."
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap of EEG Channels")
    path = save_fig("correlation_heatmap_raw.png")
    md_image(path, "Correlation Heatmap")

    # 3.3 Box Plots
    subtitle("3.3 Box Plots")
    md_text("Box plots highlight potential outliers beyond the 1.5x IQR whiskers.")
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df[ch], ax=ax, color="#3498db")
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle("Box Plots — All EEG Channels (Raw)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("boxplots_raw.png")
    md_image(path, "Box Plots")

    # 3.4 Histograms
    subtitle("3.4 Histograms")
    md_text("Amplitude distributions per channel split by eye state.")
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        for state, color in [(0, "#3498db"), (1, "#e74c3c")]:
            ax.hist(df.loc[df[TARGET] == state, ch], bins=40, alpha=0.5,
                    color=color, label=EYE_MAP[state])
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle("Histograms — All Channels by Eye State (Raw)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("histograms_raw.png")
    md_image(path, "Histograms")

    # 3.5 Violin Plots
    subtitle("3.5 Violin Plots")
    md_text("Violin plots combine box-plot summaries with kernel density estimates.")
    df_tmp = df.copy()
    df_tmp["eyeState"] = df_tmp[TARGET].map(EYE_MAP)
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.violinplot(x="eyeState", y=ch, data=df_tmp, ax=ax,
                       palette=["#3498db", "#e74c3c"], inner="quartile",
                       order=["Open", "Closed"])
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
    plt.suptitle("Violin Plots — All Channels by Eye State (Raw)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("violinplots_raw.png")
    md_image(path, "Violin Plots")

# =============================================================================
# 4. Outlier Removal — multi-pass IQR
# =============================================================================

def section_outlier_removal(df):
    title("4. Outlier Removal")
    md_text(
        "Outliers in EEG data arise from muscle artifacts, electrode displacement, or "
        "external interference. An **iterative IQR method** (1.5x interquartile range) is "
        "applied in multiple passes until no further outliers remain, ensuring that new "
        "outliers exposed by earlier passes are also removed."
    )

    original_count = len(df)
    cleaned = df.copy()
    pass_num = 0
    first_pass_bounds = []

    while True:
        pass_num += 1
        before = len(cleaned)
        for col in FEATURE_COLUMNS:
            Q1, Q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            cleaned = cleaned[(cleaned[col] >= lo) & (cleaned[col] <= hi)]
            if pass_num == 1:
                first_pass_bounds.append([col, f"{lo:.2f}", f"{hi:.2f}"])
        after = len(cleaned)
        if before - after == 0 or pass_num > 10:
            break

    removed = original_count - len(cleaned)

    md_text(f"**Passes required:** {pass_num} (converged when no further outliers found).")
    md_table(["Channel", "Lower Bound (Pass 1)", "Upper Bound (Pass 1)"], first_pass_bounds)
    md_table(
        ["Metric", "Value"],
        [
            ["Original samples", original_count],
            ["Cleaned samples", len(cleaned)],
            ["Removed samples", removed],
            ["Removal percentage", f"{removed / original_count * 100:.1f}%"],
            ["Passes", pass_num],
        ],
    )
    return cleaned.reset_index(drop=True)

# =============================================================================
# 5. Data Visualization (After Outlier Removal) — condensed
# =============================================================================

def section_data_viz_cleaned(df_raw, df_clean):
    title("5. Data Visualization (After Outlier Removal)")
    md_text("Comparison of distributions before and after outlier removal.")

    # 5.1 Box Plots Comparison
    subtitle("5.1 Box Plots Comparison")
    md_text("Side-by-side box plots confirm outlier removal effectiveness.")
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df_clean[ch], ax=ax, color="#2ecc71")
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle("Box Plots — After Outlier Removal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("boxplots_cleaned.png")
    md_image(path, "Box Plots After Cleaning")

    # 5.2 Histograms After Cleaning
    subtitle("5.2 Histograms After Cleaning")
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        for state, color in [(0, "#3498db"), (1, "#e74c3c")]:
            ax.hist(df_clean.loc[df_clean[TARGET] == state, ch], bins=40,
                    alpha=0.5, color=color, label=EYE_MAP[state])
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle("Histograms — After Outlier Removal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("histograms_cleaned.png")
    md_image(path, "Histograms After Cleaning")

# =============================================================================
# 6. Log-Normalization
# =============================================================================

def section_log_normalization(df):
    title("6. Log-Normalization")
    md_text(
        "Logarithmic normalization compresses the dynamic range of EEG amplitudes, "
        "reducing the impact of extreme values and making distributions more symmetric. "
        "We apply `log10(x - min + 1)` to each channel."
    )

    df_norm = df.copy()
    for col in FEATURE_COLUMNS:
        df_norm[col] = np.log10(df[col] - df[col].min() + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["AF3"], bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    axes[0].set_title("AF3 — Before Log-Normalization")
    axes[0].set_xlabel("Amplitude (uV)")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(df_norm["AF3"], bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
    axes[1].set_title("AF3 — After Log-Normalization")
    axes[1].set_xlabel("log10(Normalized Amplitude)")
    axes[1].set_ylabel("Frequency")
    plt.suptitle("Effect of Log-Normalization on Channel AF3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("log_normalization_comparison.png")
    md_image(path, "Log-Normalization Effect")

    md_table(
        ["Channel", "Orig Mean", "Orig Std", "Norm Mean", "Norm Std"],
        [[ch, f"{df[ch].mean():.2f}", f"{df[ch].std():.2f}",
          f"{df_norm[ch].mean():.4f}", f"{df_norm[ch].std():.4f}"]
         for ch in FEATURE_COLUMNS],
    )
    return df_norm

# =============================================================================
# 7. Feature Engineering
# =============================================================================

def section_feature_engineering(df):
    title("7. Feature Engineering")
    md_text(
        "Feature engineering derives new variables from raw EEG channels to capture "
        "domain-specific patterns that may improve classification performance."
    )

    df_eng = df.copy()
    new_features = []

    # 7.1 Hemispheric Asymmetry
    subtitle("7.1 Hemispheric Asymmetry")
    md_text(
        "The asymmetry index $(Left - Right)$ for paired electrodes captures "
        "lateralisation differences linked to cognitive and emotional states. "
        "Research shows that hemispheric imbalance correlates with attentional shifts "
        "associated with eye opening and closing."
    )
    asym_rows = []
    for left, right in HEMI_PAIRS:
        fname = f"{left}_{right}_asym"
        df_eng[fname] = df_eng[left] - df_eng[right]
        new_features.append(fname)
        asym_rows.append([
            fname, left, right,
            f"{df_eng[fname].mean():.4f}", f"{df_eng[fname].std():.4f}",
        ])
    md_table(["Feature", "Left", "Right", "Mean", "Std"], asym_rows)

    # 7.2 Global Channel Statistics
    subtitle("7.2 Global Channel Statistics")
    md_text(
        "Per-sample summary statistics across all 14 channels capture overall "
        "brain activity levels at each time point."
    )
    df_eng["ch_mean"] = df_eng[FEATURE_COLUMNS].mean(axis=1)
    df_eng["ch_std"] = df_eng[FEATURE_COLUMNS].std(axis=1)
    new_features += ["ch_mean", "ch_std"]
    md_table(
        ["Feature", "Description", "Mean", "Std"],
        [
            ["ch_mean", "Mean across 14 channels",
             f"{df_eng['ch_mean'].mean():.2f}", f"{df_eng['ch_mean'].std():.2f}"],
            ["ch_std", "Std across 14 channels",
             f"{df_eng['ch_std'].mean():.4f}", f"{df_eng['ch_std'].std():.4f}"],
        ],
    )

    # 7.3 Feature Summary
    subtitle("7.3 Feature Summary")
    all_features = FEATURE_COLUMNS + new_features
    md_text(
        f"Total features for classification: **{len(all_features)}** "
        f"(14 original + {len(new_features)} engineered)."
    )
    md_table(
        ["#", "Feature", "Type"],
        [[i + 1, f, "Original EEG" if f in FEATURE_COLUMNS else "Engineered"]
         for i, f in enumerate(all_features)],
    )
    return df_eng, all_features

# =============================================================================
# 8. FFT, Spectrogram and PSD — all 14 channels
# =============================================================================

def section_fft_psd_spectro(df):
    title("8. FFT, Spectrogram and PSD Analysis")
    md_text(
        "Frequency-domain analysis reveals the power distribution across brain wave "
        "bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** (8-12 Hz), "
        "**Beta** (12-30 Hz), and **Gamma** (30-64 Hz). Alpha power increases when "
        "eyes are closed (the **Berger effect**)."
    )

    # 8.1 FFT — all 14 channels in a grid
    subtitle("8.1 FFT Frequency Spectrum")
    md_text("The FFT decomposes each EEG channel into constituent frequencies.")

    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    axes_flat = axes.flatten()
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes_flat[idx]
        signal = df[ch].values
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, 1 / SAMPLING_RATE)
        pos = freqs > 0
        power = np.abs(fft_vals[pos]) ** 2 / n
        ax.semilogy(freqs[pos], power, linewidth=0.4, color="#1f77b4")
        ax.set_xlim(0, 64)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        for i, (_, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            ax.axvspan(lo, hi, alpha=0.1, color=BAND_COLORS[i])
    plt.suptitle("FFT Frequency Spectrum — All Channels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("fft_frequency_spectrum.png")
    md_image(path, "FFT Frequency Spectrum")

    # 8.2 PSD — all 14 channels with band labels
    subtitle("8.2 Power Spectral Density (PSD)")
    md_text(
        "Welch's method estimates the PSD for each channel. Shaded regions and labels "
        "indicate standard EEG frequency bands."
    )

    df_open = df[df[TARGET] == 0]
    df_closed = df[df[TARGET] == 1]
    nperseg = min(256, len(df_open), len(df_closed))

    fig, axes = plt.subplots(2, 7, figsize=(28, 10))
    axes_flat = axes.flatten()
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes_flat[idx]
        fo, po = welch(df_open[ch].values, SAMPLING_RATE, nperseg=nperseg)
        fc, pc = welch(df_closed[ch].values, SAMPLING_RATE, nperseg=nperseg)
        ax.semilogy(fo, po, label="Open", color="blue", linewidth=1, alpha=0.8)
        ax.semilogy(fc, pc, label="Closed", color="red", linewidth=1, alpha=0.8)
        ax.set_xlim(0, 35)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        # band shading and labels
        ylims = ax.get_ylim()
        for i, (bname, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            bhi = min(hi, 35)
            ax.axvspan(lo, bhi, alpha=0.08, color=BAND_COLORS[i])
            if bhi <= 35:
                mid = (lo + bhi) / 2
                ax.text(mid, ylims[1] * 0.3, bname, fontsize=5,
                        ha="center", va="top", color=BAND_COLORS[i],
                        fontweight="bold", rotation=90)
        if idx == 0:
            ax.legend(fontsize=6)
    plt.suptitle("PSD — All Channels (Open vs Closed)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("psd_analysis.png")
    md_image(path, "PSD Analysis")

    # 8.3 Spectrograms — all 14 channels, one grid per state
    subtitle("8.3 Spectrogram Analysis")
    md_text(
        "Spectrograms show the time-frequency power distribution. Horizontal dashed "
        "lines mark band boundaries (4, 8, 12, 30 Hz)."
    )

    for state_name, state_val in [("Open", 0), ("Closed", 1)]:
        fig, axes = plt.subplots(2, 7, figsize=(28, 8))
        axes_flat = axes.flatten()
        data_state = df[df[TARGET] == state_val]
        for idx, ch in enumerate(FEATURE_COLUMNS):
            ax = axes_flat[idx]
            data = data_state[ch].values
            seg = min(128, len(data) // 4)
            if seg < 4:
                seg = 4
            f, t, Sxx = scipy_spectrogram(
                data, fs=SAMPLING_RATE, nperseg=seg, noverlap=seg // 2)
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                               shading="gouraud", cmap="viridis")
            ax.set_ylim(0, 30)
            ax.set_title(ch, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)
            for freq in [4, 8, 12, 30]:
                ax.axhline(y=freq, color="white", linestyle="--",
                           linewidth=0.4, alpha=0.5)
        plt.suptitle(f"Spectrograms — Eyes {state_name} (All Channels)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = save_fig(f"spectrograms_{state_name.lower()}.png")
        md_image(path, f"Spectrograms Eyes {state_name}")

# =============================================================================
# 9. Dimensionality Reduction — PCA, LDA, t-SNE, UMAP
# =============================================================================

def section_dim_reduction(df, all_features):
    title("9. Dimensionality Reduction")
    md_text(
        "Projecting high-dimensional EEG data into lower-dimensional spaces reveals "
        "clustering structure. **PCA** maximises variance; **LDA** maximises class "
        "separability; **t-SNE** and **UMAP** capture non-linear manifold structure."
    )

    X = df[all_features].values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9.1 PCA
    subtitle("9.1 PCA")
    md_text("PCA identifies orthogonal directions of maximum variance.")
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_comp = len(cum_var)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_ * 100,
                color="#3498db", edgecolor="black")
    axes[0].set_title("Explained Variance per Component")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance (%)")
    axes[0].set_xticks(range(1, n_comp + 1))
    axes[1].plot(range(1, n_comp + 1), cum_var, "o-", color="#e74c3c", linewidth=2)
    axes[1].axhline(y=95, color="gray", linestyle="--", alpha=0.7, label="95% threshold")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_xticks(range(1, n_comp + 1))
    axes[1].legend()
    plt.tight_layout()
    path = save_fig("pca_variance.png")
    md_image(path, "PCA Variance")

    md_table(
        ["Component", "Variance (%)", "Cumulative (%)"],
        [[f"PC{i+1}", f"{pca_full.explained_variance_ratio_[i]*100:.2f}",
          f"{cum_var[i]:.2f}"] for i in range(n_comp)],
    )

    n_95 = int(np.argmax(cum_var >= 95) + 1)
    md_text(f"**{n_95} components** capture >= 95% of variance.")

    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y, cmap="coolwarm",
                         alpha=0.4, s=10, edgecolors="none")
    ax.set_title("PCA — 2D Projection")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(scatter, label="Eye State (0=Open, 1=Closed)")
    path = save_fig("pca_2d_projection.png")
    md_image(path, "PCA 2D Projection")

    # 9.2 LDA
    subtitle("9.2 LDA")
    md_text(
        "LDA maximises the ratio of between-class to within-class variance, yielding "
        "a single discriminant for binary classification."
    )
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#3498db", "Open"), (1, "#e74c3c", "Closed")]:
        ax.hist(X_lda[y == label], bins=50, alpha=0.6, color=color,
                label=name, edgecolor="black")
    ax.set_title("LDA — 1D Projection")
    ax.set_xlabel("LD1")
    ax.set_ylabel("Frequency")
    ax.legend()
    path = save_fig("lda_1d_projection.png")
    md_image(path, "LDA 1D Projection")

    # 9.3 t-SNE
    subtitle("9.3 t-SNE")
    md_text(
        "t-Distributed Stochastic Neighbor Embedding is a non-linear technique that "
        "preserves local neighbourhood structure. A subsample of 5000 points is used "
        "for computational efficiency."
    )
    n_tsne = min(5000, len(X_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    idx_sub = rng.choice(len(X_scaled), n_tsne, replace=False)
    X_sub = X_scaled[idx_sub]
    y_sub = y[idx_sub]

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap="coolwarm",
                         alpha=0.4, s=10, edgecolors="none")
    ax.set_title("t-SNE — 2D Projection")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Eye State (0=Open, 1=Closed)")
    path = save_fig("tsne_2d_projection.png")
    md_image(path, "t-SNE 2D Projection")

    # 9.4 UMAP
    subtitle("9.4 UMAP")
    X_umap = None
    if HAS_UMAP:
        md_text(
            "UMAP preserves both local and global structure, often producing cleaner "
            "clusters than t-SNE with faster computation."
        )
        umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=RANDOM_STATE)
        X_umap = umap_model.fit_transform(X_sub)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sub, cmap="coolwarm",
                             alpha=0.4, s=10, edgecolors="none")
        ax.set_title("UMAP — 2D Projection")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter, label="Eye State (0=Open, 1=Closed)")
        path = save_fig("umap_2d_projection.png")
        md_image(path, "UMAP 2D Projection")
    else:
        md_text("> **Note:** `umap-learn` is not installed. Skipping UMAP.")

    # 9.5 Clustering Evaluation
    subtitle("9.5 Clustering Evaluation")
    md_text("Clustering metrics quantify separation quality in reduced spaces.")
    metrics_rows = []
    for name, Xr, y_eval in [
        ("PCA (2D)", X_pca2, y),
        ("LDA (1D)", np.column_stack([X_lda, np.zeros_like(X_lda)]), y),
        ("t-SNE (2D)", X_tsne, y_sub),
    ]:
        sil = silhouette_score(Xr, y_eval)
        db = davies_bouldin_score(Xr, y_eval)
        ch_s = calinski_harabasz_score(Xr, y_eval)
        metrics_rows.append([name, f"{sil:.4f}", f"{db:.4f}", f"{ch_s:.2f}"])

    if X_umap is not None:
        sil = silhouette_score(X_umap, y_sub)
        db = davies_bouldin_score(X_umap, y_sub)
        ch_s = calinski_harabasz_score(X_umap, y_sub)
        metrics_rows.append(["UMAP (2D)", f"{sil:.4f}", f"{db:.4f}", f"{ch_s:.2f}"])

    md_table(
        ["Method", "Silhouette (higher better)",
         "Davies-Bouldin (lower better)",
         "Calinski-Harabasz (higher better)"],
        metrics_rows,
    )

# =============================================================================
# 10. Machine Learning Classification
# =============================================================================

def section_ml(df, all_features):
    title("10. Machine Learning Classification")
    md_text(
        "Five classical ML algorithms are evaluated using a 70/30 stratified "
        "train-test split. `StandardScaler` is fit **exclusively on training data** "
        "to prevent data leakage."
    )

    X = df[all_features].values
    y = df[TARGET].values

    # Split BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 10.1 Train/Test Split & Class Balance
    subtitle("10.1 Train/Test Split & Class Balance")
    md_text(
        f"Stratified split: {int((1 - TEST_SIZE) * 100)}% train / "
        f"{int(TEST_SIZE * 100)}% test, preserving class proportions."
    )
    train_vc = pd.Series(y_train).value_counts()
    test_vc = pd.Series(y_test).value_counts()
    md_table(
        ["Split", "Open (0)", "Closed (1)", "Total", "Closed %"],
        [
            ["Train", train_vc.get(0, 0), train_vc.get(1, 0), len(y_train),
             f"{train_vc.get(1, 0) / len(y_train) * 100:.1f}%"],
            ["Test", test_vc.get(0, 0), test_vc.get(1, 0), len(y_test),
             f"{test_vc.get(1, 0) / len(y_test) * 100:.1f}%"],
        ],
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
    }

    descriptions = {
        "Logistic Regression": (
            "Logistic Regression models the posterior probability using the sigmoid function:\n\n"
            "$$P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = "
            "\\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$\n\n"
            "The model minimises binary cross-entropy loss with L2 regularisation:\n\n"
            "$$\\mathcal{L} = -\\frac{1}{N}\\sum_{i=1}^{N}"
            "[y_i \\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)] "
            "+ \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$$\n\n"
            "It serves as an interpretable linear baseline for binary classification."
        ),
        "K-Nearest Neighbors": (
            "KNN classifies each sample by majority vote among its $k$ nearest neighbours "
            "using the Euclidean distance metric:\n\n"
            "$$d(\\mathbf{x}_i, \\mathbf{x}_j) = "
            "\\sqrt{\\sum_{m=1}^{M}(x_{im} - x_{jm})^2}$$\n\n"
            "The predicted class is:\n\n"
            "$$\\hat{y} = \\arg\\max_c \\sum_{i \\in N_k(\\mathbf{x})} "
            "\\mathbb{1}(y_i = c)$$\n\n"
            "KNN is non-parametric, making no distributional assumptions. With $k=5$ and "
            "standardised features, it captures local EEG decision boundaries."
        ),
        "Support Vector Machine": (
            "SVM finds the hyperplane that maximises the margin between classes. "
            "The RBF kernel maps features into higher-dimensional space:\n\n"
            "$$K(\\mathbf{x}_i, \\mathbf{x}_j) = "
            "\\exp(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2)$$\n\n"
            "The optimisation objective with soft margin is:\n\n"
            "$$\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 "
            "+ C \\sum_{i=1}^{N} \\max(0, 1 - y_i(\\mathbf{w}^T"
            "\\phi(\\mathbf{x}_i) + b))$$\n\n"
            "The RBF kernel captures non-linear decision boundaries between eye states."
        ),
        "Random Forest": (
            "Random Forest builds an ensemble of $B$ decision trees, each trained on a "
            "bootstrapped subset with random feature selection:\n\n"
            "$$\\hat{y} = \\text{mode}\\{h_b(\\mathbf{x})\\}_{b=1}^{B}$$\n\n"
            "Each tree splits nodes using the Gini impurity criterion:\n\n"
            "$$G = 1 - \\sum_{c=1}^{C} p_c^2$$\n\n"
            "Bagging reduces variance and random subspace selection decorrelates trees. "
            "200 estimators are used."
        ),
        "Gradient Boosting": (
            "Gradient Boosting builds an additive ensemble where each tree corrects "
            "residual errors of the previous ensemble:\n\n"
            "$$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\eta \\cdot h_m(\\mathbf{x})$$\n\n"
            "Each tree $h_m$ is fit to the negative gradient of the loss function. "
            "The learning rate $\\eta$ controls the contribution of each tree. "
            "200 boosting rounds are used with default depth and $\\eta = 0.1$."
        ),
    }

    # 10.2 Cross-Validation
    subtitle("10.2 Cross-Validation Results (5-Fold Stratified)")
    md_text("5-fold stratified cross-validation on the training set.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="f1")
        cv_rows.append([name, f"{scores.mean():.4f}", f"{scores.std():.4f}"])
    md_table(["Model", "CV F1 Mean", "CV F1 Std"], cv_rows)

    # Train and evaluate each model
    all_results = {}
    all_y_probs = {}
    sec_idx = 3
    for name, model in models.items():
        subtitle(f"10.{sec_idx} {name}")
        md_text(descriptions[name])
        progress(f"  Training {name} ...")

        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test_s)
        y_prob = (model.predict_proba(X_test_s)[:, 1]
                  if hasattr(model, "predict_proba") else None)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        all_results[name] = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            **{"F1-Score": f1, "AUC-ROC": auc, "Train Time (s)": train_time})
        if y_prob is not None:
            all_y_probs[name] = y_prob

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{acc:.4f}"], ["Precision", f"{prec:.4f}"],
            ["Recall", f"{rec:.4f}"], ["F1-Score", f"{f1:.4f}"],
            ["AUC-ROC", f"{auc:.4f}"], ["Training Time", f"{train_time:.3f}s"],
        ])
        sec_idx += 1

    # 10.8 Feature Importance
    subtitle("10.8 Feature Importance")
    md_text("Feature importance from Random Forest and Gradient Boosting.")
    fig, axes_fi = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, mname in enumerate(["Random Forest", "Gradient Boosting"]):
        model_fi = models[mname]
        importances = model_fi.feature_importances_
        sorted_idx = np.argsort(importances)
        top_n = min(15, len(all_features))
        top_idx = sorted_idx[-top_n:]
        axes_fi[ax_idx].barh(
            [all_features[i] for i in top_idx],
            importances[top_idx], color="#3498db", edgecolor="black")
        axes_fi[ax_idx].set_title(f"{mname} — Top Features", fontweight="bold")
        axes_fi[ax_idx].set_xlabel("Importance")
    plt.tight_layout()
    path = save_fig("feature_importance.png")
    md_image(path, "Feature Importance")

    # 10.9 ROC Curves
    subtitle("10.9 ROC Curves")
    md_text("ROC curves plot True Positive Rate vs False Positive Rate.")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, yp in all_y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc_val = all_results[name]["AUC-ROC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ML Models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = save_fig("ml_roc_curves.png")
    md_image(path, "ML ROC Curves")

    # 10.10 Comparison
    subtitle("10.10 ML Model Comparison")
    headers = ["Model", "Accuracy", "Precision", "Recall",
               "F1-Score", "AUC-ROC", "Time (s)"]
    rows = [[n, f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
             f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
             f"{r['AUC-ROC']:.4f}", f"{r['Train Time (s)']:.3f}"]
            for n, r in all_results.items()]
    md_table(headers, rows)

    # Confusion matrices
    fig, axes_cm = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[idx],
                    xticklabels=["Open", "Closed"],
                    yticklabels=["Open", "Closed"])
        short = name.split()[-1] if len(name) > 15 else name
        axes_cm[idx].set_title(short, fontsize=10)
        axes_cm[idx].set_xlabel("Predicted")
        axes_cm[idx].set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — ML Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("ml_confusion_matrices.png")
    md_image(path, "ML Confusion Matrices")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(all_results.keys())
    metrics_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(model_names))
    w = 0.18
    for i, m in enumerate(metrics_plot):
        vals = [all_results[n][m] for n in model_names]
        ax.bar(x + i * w, vals, w, label=m, edgecolor="black")
    ax.set_ylabel("Score")
    ax.set_title("ML Model Performance Comparison")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = save_fig("ml_comparison_chart.png")
    md_image(path, "ML Comparison Chart")

    return all_results

# =============================================================================
# 11. Neural Network Classification
# =============================================================================

def _eval_nn(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        **{"F1-Score": f1_score(y_true, y_pred, zero_division=0),
           "AUC-ROC": roc_auc_score(y_true, y_prob)},
    )


def _plot_history(history, model_name, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    return save_fig(filename)


def section_neural_network(df):
    title("11. Neural Network Classification")
    md_text(
        "Deep-learning models learn hierarchical feature representations from raw "
        "EEG signals. This section evaluates a **1D CNN**, a **2D CNN on spectrograms**, "
        "and an **LSTM** network."
    )

    nn_results = {}
    WINDOW = 64
    STEP = 16

    # ---- Fallback: no TensorFlow -------------------------------------------
    if not HAS_TF:
        md_text(
            "> **Note:** TensorFlow not installed. Using sklearn MLPClassifier."
        )
        subtitle("11.1 MLP Neural Network (sklearn fallback)")
        md_text(
            "A multi-layer perceptron (128-64-32 hidden units) serves as the "
            "deep-learning proxy.\n\n"
            "The MLP computes:\n\n"
            "$$\\mathbf{h}^{(l)} = \\text{ReLU}(\\mathbf{W}^{(l)} \\mathbf{h}^{(l-1)} "
            "+ \\mathbf{b}^{(l)})$$\n\n"
            "with output $\\hat{y} = \\sigma(\\mathbf{w}^T \\mathbf{h}^{(L)} + b)$."
        )

        X = df[FEATURE_COLUMNS].values
        y = df[TARGET].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                            max_iter=500, random_state=RANDOM_STATE)
        t0 = time.time()
        mlp.fit(Xtr, ytr)
        tt = time.time() - t0
        yp = mlp.predict_proba(Xte)[:, 1]
        res = _eval_nn(yte, yp)
        res["Train Time (s)"] = tt
        nn_results["MLP (sklearn)"] = res

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res['Accuracy']:.4f}"],
            ["Precision", f"{res['Precision']:.4f}"],
            ["Recall", f"{res['Recall']:.4f}"],
            ["F1-Score", f"{res['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res['AUC-ROC']:.4f}"],
            ["Training Time", f"{tt:.3f}s"],
        ])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mlp.loss_curve_, color="#3498db")
        ax.set_title("MLP — Training Loss Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        path = save_fig("mlp_loss_curve.png")
        md_image(path, "MLP Training Loss")

        subtitle("11.4 Neural Network Comparison")
        md_table(
            ["Model", "Accuracy", "Precision", "Recall",
             "F1-Score", "AUC-ROC", "Train Time (s)"],
            [[n, f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
              f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
              f"{r['AUC-ROC']:.4f}", f"{r['Train Time (s)']:.3f}"]
             for n, r in nn_results.items()],
        )
        return nn_results

    # ---- TensorFlow path ---------------------------------------------------
    progress("  Preparing EEG windows for deep learning ...")

    # Build windows
    X_data = df[FEATURE_COLUMNS].values
    y_data = df[TARGET].values
    Xw, yw = [], []
    for i in range(0, len(X_data) - WINDOW, STEP):
        Xw.append(X_data[i:i + WINDOW])
        yw.append(int(np.round(np.mean(y_data[i:i + WINDOW]))))
    X_win = np.array(Xw)
    y_win = np.array(yw)

    # Stratified split BEFORE scaling
    Xtr, Xte, ytr, yte = train_test_split(
        X_win, y_win, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_win)

    # Scale: fit on train only
    tr_flat = Xtr.reshape(-1, Xtr.shape[-1])
    sc = StandardScaler().fit(tr_flat)
    Xtr = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xte = sc.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)

    md_text(
        f"Window size = {WINDOW} samples, step = {STEP}. "
        f"Total windows: {len(X_win)} (train {len(Xtr)}, test {len(Xte)})."
    )

    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                 monitor="val_loss")

    # 11.1 1D CNN
    subtitle("11.1 1D CNN on Raw EEG")
    md_text(
        "A 1D Convolutional Neural Network applies learnable filters across the "
        f"temporal dimension of multi-channel EEG windows ({WINDOW} samples x 14 channels). "
        "The convolution for filter $f$ at position $t$ is:\n\n"
        "$$y_t^{(f)} = \\text{ReLU}\\left(\\sum_{k=0}^{K-1} \\sum_{c=1}^{C} "
        "w_{k,c}^{(f)} \\cdot x_{t+k,c} + b^{(f)}\\right)$$\n\n"
        "where $K$ is the kernel size and $C$ the number of channels. Max-pooling "
        "reduces dimensionality and global average pooling aggregates features."
    )
    progress("  Training 1D CNN ...")

    m1 = keras.Sequential([
        layers.Input(shape=(WINDOW, len(FEATURE_COLUMNS))),
        layers.Conv1D(64, 7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h1 = m1.fit(Xtr, ytr, epochs=30, batch_size=64,
                validation_split=0.2, callbacks=[es], verbose=0)
    t1 = time.time() - t0

    yp1 = m1.predict(Xte, verbose=0).flatten()
    res1 = _eval_nn(yte, yp1)
    res1["Train Time (s)"] = t1
    nn_results["1D CNN"] = res1

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res1['Accuracy']:.4f}"],
        ["Precision", f"{res1['Precision']:.4f}"],
        ["Recall", f"{res1['Recall']:.4f}"],
        ["F1-Score", f"{res1['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res1['AUC-ROC']:.4f}"],
        ["Training Time", f"{t1:.3f}s"],
    ])
    path = _plot_history(h1, "1D CNN", "cnn1d_training.png")
    md_image(path, "1D CNN Training History")

    # 11.2 CNN on Spectrograms (improved)
    subtitle("11.2 CNN on Spectrograms")
    md_text(
        "A 2D CNN processes spectrogram representations of EEG windows as "
        "multi-channel images (frequency x time x EEG channels). The 2D convolution "
        "learns frequency-time patterns:\n\n"
        "$$Y_{i,j}^{(f)} = \\text{ReLU}\\left(\\sum_{m,n,c} W_{m,n,c}^{(f)} "
        "\\cdot X_{i+m,j+n,c} + b^{(f)}\\right)$$\n\n"
        "**Improvements:** Smaller windows with more overlap generate more training "
        "samples. Class weights address label imbalance. A reduced learning rate "
        "and increased patience improve convergence."
    )
    progress("  Building spectrogram windows ...")

    SPEC_WIN = 64
    SPEC_STEP = 8
    seg = 32
    ovlp = seg - 8

    specs, labels = [], []
    for i in range(0, len(X_data) - SPEC_WIN, SPEC_STEP):
        window = X_data[i:i + SPEC_WIN]
        label = int(np.round(np.mean(y_data[i:i + SPEC_WIN])))
        ch_specs = []
        for c in range(window.shape[1]):
            _, _, Sxx = scipy_spectrogram(
                window[:, c], fs=SAMPLING_RATE, nperseg=seg, noverlap=ovlp)
            ch_specs.append(10 * np.log10(Sxx + 1e-10))
        specs.append(np.stack(ch_specs, axis=-1))
        labels.append(label)

    X_spec = np.array(specs)
    y_spec = np.array(labels)

    md_text(
        f"Spectrogram window = {SPEC_WIN}, step = {SPEC_STEP}. "
        f"Shape per sample: {X_spec.shape[1:]} (freq x time x channels). "
        f"Total samples: {len(X_spec)}."
    )

    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_spec, y_spec, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_spec)

    # Normalise: fit stats on train only
    tr_mean = Xtr2.mean()
    tr_std = Xtr2.std()
    Xtr2 = (Xtr2 - tr_mean) / (tr_std + 1e-8)
    Xte2 = (Xte2 - tr_mean) / (tr_std + 1e-8)

    # Class weights
    n0 = int(np.sum(ytr2 == 0))
    n1 = int(np.sum(ytr2 == 1))
    cw = {0: len(ytr2) / (2.0 * max(n0, 1)), 1: len(ytr2) / (2.0 * max(n1, 1))}

    progress("  Training 2D CNN on spectrograms ...")

    es2 = callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                  monitor="val_loss")
    m2 = keras.Sequential([
        layers.Input(shape=X_spec.shape[1:]),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    m2.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h2 = m2.fit(Xtr2, ytr2, epochs=50, batch_size=32,
                validation_split=0.2, callbacks=[es2],
                class_weight=cw, verbose=0)
    t2 = time.time() - t0

    yp2 = m2.predict(Xte2, verbose=0).flatten()
    res2 = _eval_nn(yte2, yp2)
    res2["Train Time (s)"] = t2
    nn_results["CNN (Spectrogram)"] = res2

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res2['Accuracy']:.4f}"],
        ["Precision", f"{res2['Precision']:.4f}"],
        ["Recall", f"{res2['Recall']:.4f}"],
        ["F1-Score", f"{res2['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res2['AUC-ROC']:.4f}"],
        ["Training Time", f"{t2:.3f}s"],
    ])
    path = _plot_history(h2, "2D CNN (Spectrogram)", "cnn2d_spectrogram_training.png")
    md_image(path, "CNN Spectrogram Training History")

    # 11.3 LSTM
    subtitle("11.3 LSTM / RNN")
    md_text(
        "Long Short-Term Memory networks capture temporal dependencies through "
        "gating mechanisms:\n\n"
        "$$\\mathbf{f}_t = \\sigma(\\mathbf{W}_f [\\mathbf{h}_{t-1}, "
        "\\mathbf{x}_t] + \\mathbf{b}_f) \\quad \\text{(forget gate)}$$\n"
        "$$\\mathbf{i}_t = \\sigma(\\mathbf{W}_i [\\mathbf{h}_{t-1}, "
        "\\mathbf{x}_t] + \\mathbf{b}_i) \\quad \\text{(input gate)}$$\n"
        "$$\\tilde{\\mathbf{c}}_t = \\tanh(\\mathbf{W}_c [\\mathbf{h}_{t-1}, "
        "\\mathbf{x}_t] + \\mathbf{b}_c) \\quad \\text{(candidate)}$$\n"
        "$$\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} "
        "+ \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t \\quad \\text{(cell state)}$$\n"
        "$$\\mathbf{o}_t = \\sigma(\\mathbf{W}_o [\\mathbf{h}_{t-1}, "
        "\\mathbf{x}_t] + \\mathbf{b}_o) \\quad \\text{(output gate)}$$\n"
        "$$\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t) "
        "\\quad \\text{(hidden state)}$$\n\n"
        "The forget gate controls what to discard, the input gate what to store, "
        "and the output gate what to expose as the hidden state."
    )
    progress("  Training LSTM ...")

    es3 = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                  monitor="val_loss")
    m3 = keras.Sequential([
        layers.Input(shape=(WINDOW, len(FEATURE_COLUMNS))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h3 = m3.fit(Xtr, ytr, epochs=30, batch_size=64,
                validation_split=0.2, callbacks=[es3], verbose=0)
    t3 = time.time() - t0

    yp3 = m3.predict(Xte, verbose=0).flatten()
    res3 = _eval_nn(yte, yp3)
    res3["Train Time (s)"] = t3
    nn_results["LSTM"] = res3

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res3['Accuracy']:.4f}"],
        ["Precision", f"{res3['Precision']:.4f}"],
        ["Recall", f"{res3['Recall']:.4f}"],
        ["F1-Score", f"{res3['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res3['AUC-ROC']:.4f}"],
        ["Training Time", f"{t3:.3f}s"],
    ])
    path = _plot_history(h3, "LSTM", "lstm_training.png")
    md_image(path, "LSTM Training History")

    # 11.4 Comparison
    subtitle("11.4 Neural Network Comparison")
    md_text("Side-by-side comparison of all neural-network architectures.")

    headers = ["Model", "Accuracy", "Precision", "Recall",
               "F1-Score", "AUC-ROC", "Train Time (s)"]
    rows = [[n, f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
             f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
             f"{r['AUC-ROC']:.4f}", f"{r['Train Time (s)']:.3f}"]
            for n, r in nn_results.items()]
    md_table(headers, rows)

    # comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    nn_names = list(nn_results.keys())
    mets = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(nn_names))
    w = 0.18
    for i, mt in enumerate(mets):
        vals = [nn_results[n][mt] for n in nn_names]
        ax.bar(x + i * w, vals, w, label=mt, edgecolor="black")
    ax.set_ylabel("Score")
    ax.set_title("Neural Network Performance Comparison")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(nn_names, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = save_fig("nn_comparison_chart.png")
    md_image(path, "Neural Network Comparison")

    # Confusion matrices for NN
    nn_preds_map = {}
    if "1D CNN" in nn_results:
        nn_preds_map["1D CNN"] = (yte, yp1)
    if "CNN (Spectrogram)" in nn_results:
        nn_preds_map["CNN (Spectrogram)"] = (yte2, yp2)
    if "LSTM" in nn_results:
        nn_preds_map["LSTM"] = (yte, yp3)

    if nn_preds_map:
        n_nn = len(nn_preds_map)
        fig, axes_cm = plt.subplots(1, n_nn, figsize=(5 * n_nn, 4))
        if n_nn == 1:
            axes_cm = [axes_cm]
        for idx, (nm, (yt, yp)) in enumerate(nn_preds_map.items()):
            cm = confusion_matrix(yt, (yp > 0.5).astype(int))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes_cm[idx],
                        xticklabels=["Open", "Closed"],
                        yticklabels=["Open", "Closed"])
            axes_cm[idx].set_title(nm, fontsize=10)
            axes_cm[idx].set_xlabel("Predicted")
            axes_cm[idx].set_ylabel("Actual")
        plt.suptitle("Confusion Matrices — Neural Networks",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = save_fig("nn_confusion_matrices.png")
        md_image(path, "NN Confusion Matrices")

    return nn_results

# =============================================================================
# 12. Final Comparison and Inference
# =============================================================================

def section_final_comparison(ml_results, nn_results):
    title("12. Final Comparison and Inference")
    md_text(
        "This section unifies all models — classical ML and deep learning — "
        "ranked by F1-Score."
    )

    all_res = {}
    all_res.update(ml_results)
    all_res.update(nn_results)

    subtitle("12.1 Unified Comparison Table")

    sorted_models = sorted(all_res.items(),
                           key=lambda x: x[1]["F1-Score"], reverse=True)

    headers = ["Rank", "Model", "Accuracy", "Precision", "Recall",
               "F1-Score", "AUC-ROC"]
    rows = []
    for rank, (name, r) in enumerate(sorted_models, 1):
        rows.append([rank, name,
                     f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
                     f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
                     f"{r['AUC-ROC']:.4f}"])
    md_table(headers, rows)

    # Final bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [m[0] for m in sorted_models]
    acc = [m[1]["Accuracy"] for m in sorted_models]
    f1s = [m[1]["F1-Score"] for m in sorted_models]
    x = np.arange(len(names))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#3498db", edgecolor="black")
    ax.bar(x + 0.2, f1s, 0.4, label="F1-Score", color="#e74c3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("All Models — Accuracy vs F1-Score (Ranked by F1)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = save_fig("final_comparison.png")
    md_image(path, "Final Model Comparison")

    # Inference
    subtitle("12.2 Inference and Recommendation")

    best_name, best_r = sorted_models[0]
    second_name, second_r = sorted_models[1]

    md_text(f"### Best Overall Model: **{best_name}**")
    md_text(
        f"Based on comprehensive evaluation, **{best_name}** achieves the "
        f"highest F1-Score of **{best_r['F1-Score']:.4f}** with accuracy "
        f"**{best_r['Accuracy']:.4f}** and AUC-ROC **{best_r['AUC-ROC']:.4f}**."
    )
    md_text(
        f"Runner-up: **{second_name}** (F1 = {second_r['F1-Score']:.4f})."
    )

    md_text("**Key Observations:**")

    best_ml = max(ml_results.items(), key=lambda x: x[1]["F1-Score"])
    best_nn = max(nn_results.items(), key=lambda x: x[1]["F1-Score"])

    if best_nn[1]["F1-Score"] > best_ml[1]["F1-Score"]:
        diff = (best_nn[1]["F1-Score"] - best_ml[1]["F1-Score"]) * 100
        md_text(
            f"- Deep learning (**{best_nn[0]}**) outperforms the best "
            f"classical ML model (**{best_ml[0]}**) by **{diff:.2f}** "
            f"percentage points in F1-Score."
        )
    else:
        diff = (best_ml[1]["F1-Score"] - best_nn[1]["F1-Score"]) * 100
        md_text(
            f"- The classical ML model (**{best_ml[0]}**) outperforms "
            f"deep learning (**{best_nn[0]}**) by **{diff:.2f}** "
            f"percentage points in F1-Score."
        )

    md_text(
        f"- **For production deployment**, **{best_name}** is recommended."
    )

    fastest = min(all_res.items(), key=lambda x: x[1]["Train Time (s)"])
    md_text(
        f"- **For low-latency applications**, **{fastest[0]}** offers the "
        f"fastest training ({fastest[1]['Train Time (s)']:.3f}s) with "
        f"F1 = {fastest[1]['F1-Score']:.4f}."
    )

    md_text("")
    md_text("---")

# =============================================================================
# Main pipeline
# =============================================================================

def main():
    progress("=" * 60)
    progress("EEG Eye State Classification — Pipeline Started")
    progress("=" * 60)

    print_toc()

    # 1. Load data
    progress("[1/12] Loading data ...")
    df = pd.read_csv(DATA_FILE)
    section_data_description(df)

    # 2. Imputation
    progress("[2/12] Data imputation ...")
    df = section_data_imputation(df)

    # 3. Visualization (raw)
    progress("[3/12] Visualising raw data ...")
    section_data_viz_raw(df)

    # 4. Outlier removal (multi-pass)
    progress("[4/12] Removing outliers (iterative) ...")
    df_raw_copy = df.copy()
    df_clean = section_outlier_removal(df)

    # 5. Visualization (after outlier removal) — condensed
    progress("[5/12] Visualising cleaned data ...")
    section_data_viz_cleaned(df_raw_copy, df_clean)

    # 6. Log-normalisation
    progress("[6/12] Log-normalising ...")
    df_norm = section_log_normalization(df_clean)

    # 7. Feature Engineering
    progress("[7/12] Feature engineering ...")
    df_eng, all_features = section_feature_engineering(df_clean)

    # 8. FFT / PSD / Spectrogram
    progress("[8/12] Frequency-domain analysis ...")
    section_fft_psd_spectro(df_clean)

    # 9. Dimensionality Reduction
    progress("[9/12] Dimensionality reduction (PCA, LDA, t-SNE, UMAP) ...")
    section_dim_reduction(df_eng, all_features)

    # 10. ML classifiers
    progress("[10/12] Training ML models ...")
    ml_results = section_ml(df_eng, all_features)

    # 11. Neural networks
    progress("[11/12] Training neural networks ...")
    nn_results = section_neural_network(df_clean)

    # 12. Final comparison
    progress("[12/12] Generating final comparison ...")
    section_final_comparison(ml_results, nn_results)

    progress("=" * 60)
    progress("Pipeline complete. Run:  python script.py > report.md")
    progress("=" * 60)


if __name__ == "__main__":
    main()
