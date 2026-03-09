#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Eye State Classification — Complete Analysis Pipeline
==========================================================
Usage:  python script.py > report.md

All plots are saved to the analysis-plots/ directory.
Progress messages are printed to stderr so they don't appear in the report.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------- try importing TensorFlow -----------------------------------------
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

os.makedirs(PLOT_DIR, exist_ok=True)

# Force UTF-8 on stdout when redirected (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Markdown helper functions
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
    """Print progress to stderr — visible in terminal, absent from report."""
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
    subtitle("Table of Contents")
    toc = """\
1. [Data Description Overview](#1-data-description-overview)
   - 1.1 [Dataset Loading](#11-dataset-loading)
   - 1.2 [Basic Statistics](#12-basic-statistics)
   - 1.3 [Class Distribution](#13-class-distribution)
2. [Data Imputation](#2-data-imputation)
3. [Data Visualization (Raw Data)](#3-data-visualization-raw-data)
   - 3.1 [Class Balance](#31-class-balance)
   - 3.2 [Correlation Heatmap](#32-correlation-heatmap)
   - 3.3 [Box Plots](#33-box-plots)
   - 3.4 [Histograms](#34-histograms)
   - 3.5 [Violin Plots](#35-violin-plots)
4. [Outlier Removal](#4-outlier-removal)
5. [Data Visualization (After Outlier Removal)](#5-data-visualization-after-outlier-removal)
   - 5.1 [Class Balance](#51-class-balance)
   - 5.2 [Correlation Heatmap](#52-correlation-heatmap)
   - 5.3 [Box Plots](#53-box-plots)
   - 5.4 [Histograms](#54-histograms)
   - 5.5 [Violin Plots](#55-violin-plots)
6. [Log-Normalization](#6-log-normalization)
7. [Data Visualization (After Normalization)](#7-data-visualization-after-normalization)
   - 7.1 [Class Balance](#71-class-balance)
   - 7.2 [Correlation Heatmap](#72-correlation-heatmap)
   - 7.3 [Box Plots](#73-box-plots)
   - 7.4 [Histograms](#74-histograms)
   - 7.5 [Violin Plots](#75-violin-plots)
8. [FFT, Spectrogram and PSD Analysis](#8-fft-spectrogram-and-psd-analysis)
   - 8.1 [FFT Frequency Spectrum](#81-fft-frequency-spectrum)
   - 8.2 [Power Spectral Density (PSD)](#82-power-spectral-density-psd)
   - 8.3 [Spectrogram Analysis](#83-spectrogram-analysis)
9. [PCA and LDA Analysis](#9-pca-and-lda-analysis)
   - 9.1 [PCA](#91-pca)
   - 9.2 [LDA](#92-lda)
   - 9.3 [Clustering Evaluation](#93-clustering-evaluation)
10. [Machine Learning Classification](#10-machine-learning-classification)
    - 10.1 [Logistic Regression](#101-logistic-regression)
    - 10.2 [K-Nearest Neighbors](#102-k-nearest-neighbors)
    - 10.3 [Support Vector Machine](#103-support-vector-machine)
    - 10.4 [Random Forest](#104-random-forest)
    - 10.5 [Gradient Boosting](#105-gradient-boosting)
    - 10.6 [ML Model Comparison](#106-ml-model-comparison)
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
    md_text(
        "This section provides an overview of the EEG eye-state dataset collected "
        "using an Emotiv EPOC neuroheadset with 14 electrodes at a sampling rate of "
        "128 Hz. The binary target variable `eyeDetection` indicates whether the "
        "subject's eyes were open (1) or closed (0)."
    )

    subtitle("1.1 Dataset Loading")
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

    subtitle("1.2 Basic Statistics")
    md_text("Descriptive statistics for all 14 EEG channels (micro-volts).")
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

    subtitle("1.3 Class Distribution")
    md_text("Distribution of the target variable `eyeDetection`.")
    vc = df[TARGET].value_counts()
    md_table(
        ["Eye State", "Count", "Percentage"],
        [
            ["Closed (0)", vc.get(0, 0), f"{vc.get(0, 0) / len(df) * 100:.1f}%"],
            ["Open (1)", vc.get(1, 0), f"{vc.get(1, 0) / len(df) * 100:.1f}%"],
        ],
    )

# =============================================================================
# 2. Data Imputation
# =============================================================================

def section_data_imputation(df):
    title("2. Data Imputation")
    md_text(
        "Data imputation handles missing or invalid values. Missing values are "
        "detected and filled using column-wise **median imputation** to preserve "
        "the statistical properties of each EEG channel."
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
        md_text("Missing values have been filled using **median imputation**.")
    return df

# =============================================================================
# 3 / 5 / 7 — Reusable data-visualization section
# =============================================================================

def section_data_viz(df, suffix, sec, sec_title):
    title(f"{sec}. {sec_title}")
    md_text(
        "Visualizations help identify data distributions, correlations, and "
        "potential anomalies in the EEG signals."
    )

    # --- class balance ---
    subtitle(f"{sec}.1 Class Balance")
    md_text(
        "The class-balance diagram shows the distribution of samples across eye "
        "states — essential for identifying potential class imbalance that may "
        "bias downstream classifiers."
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[TARGET].value_counts()
    bars = ax.bar(
        ["Closed (0)", "Open (1)"],
        [vc.get(0, 0), vc.get(1, 0)],
        color=["#3498db", "#e74c3c"], edgecolor="black",
    )
    ax.set_title("Class Balance of Eye States")
    ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                str(int(b.get_height())), ha="center", fontweight="bold")
    path = save_fig(f"class_balance_{suffix}.png")
    md_image(path, "Class Balance")

    # --- correlation heatmap ---
    subtitle(f"{sec}.2 Correlation Heatmap")
    md_text(
        "The correlation heatmap reveals linear relationships between EEG "
        "channels. Highly correlated channels may carry redundant information, "
        "motivating dimensionality reduction (e.g., PCA)."
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap of EEG Channels")
    path = save_fig(f"correlation_heatmap_{suffix}.png")
    md_image(path, "Correlation Heatmap")

    # --- box plots ---
    subtitle(f"{sec}.3 Box Plots")
    md_text(
        "Box plots summarize the distribution of each channel and highlight "
        "potential outliers as points beyond the whiskers (1.5x IQR rule)."
    )
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes_flat = axes.flatten()
    for i, ch in enumerate(FEATURE_COLUMNS):
        sns.boxplot(y=df[ch], ax=axes_flat[i], color="#3498db")
        axes_flat[i].set_title(ch, fontsize=10)
        axes_flat[i].tick_params(labelsize=8)
    plt.suptitle("Box Plots — All EEG Channels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig(f"boxplots_{suffix}.png")
    md_image(path, "Box Plots")

    # --- histograms ---
    subtitle(f"{sec}.4 Histograms")
    md_text(
        "Histograms show the amplitude distribution for each EEG channel split "
        "by eye state. Deviations from normality or bimodal patterns indicate "
        "differences between open- and closed-eye signals."
    )
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes_flat = axes.flatten()
    colors = {"Closed": "#3498db", "Open": "#e74c3c"}
    eye_map = {0: "Closed", 1: "Open"}
    for i, ch in enumerate(FEATURE_COLUMNS):
        for state, color in colors.items():
            mask = df[TARGET].map(eye_map) == state
            axes_flat[i].hist(df.loc[mask, ch], bins=40, alpha=0.5,
                              color=color, label=state)
        axes_flat[i].set_title(ch, fontsize=10)
        axes_flat[i].tick_params(labelsize=8)
        if i == 0:
            axes_flat[i].legend(fontsize=7)
    plt.suptitle("Histograms — All EEG Channels by Eye State",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig(f"histograms_{suffix}.png")
    md_image(path, "Histograms")

    # --- violin plots ---
    subtitle(f"{sec}.5 Violin Plots")
    md_text(
        "Violin plots combine box-plot summaries with kernel density estimates, "
        "providing a richer view of the distribution shape for each channel "
        "across eye states."
    )
    df_tmp = df.copy()
    df_tmp["eyeState"] = df_tmp[TARGET].map({0: "Closed", 1: "Open"})
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes_flat = axes.flatten()
    for i, ch in enumerate(FEATURE_COLUMNS):
        sns.violinplot(x="eyeState", y=ch, data=df_tmp, ax=axes_flat[i],
                       palette=["#3498db", "#e74c3c"], inner="quartile")
        axes_flat[i].set_title(ch, fontsize=10)
        axes_flat[i].set_xlabel("")
        axes_flat[i].tick_params(labelsize=8)
    plt.suptitle("Violin Plots — All EEG Channels by Eye State",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig(f"violinplots_{suffix}.png")
    md_image(path, "Violin Plots")

# =============================================================================
# 4. Outlier Removal
# =============================================================================

def section_outlier_removal(df):
    title("4. Outlier Removal")
    md_text(
        "Outliers in EEG data often arise from muscle artifacts, electrode "
        "displacement, or external interference. This pipeline combines the "
        "**IQR method** (1.5x interquartile range) with the **5-sigma rule** "
        "(five standard deviations from the mean). The more conservative bound "
        "is applied to retain legitimate EEG variability while removing extreme "
        "values."
    )

    original_count = len(df)
    cleaned = df.copy()
    bounds_rows = []

    for col in FEATURE_COLUMNS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo_iqr, hi_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        mean, std = df[col].mean(), df[col].std()
        lo_sig, hi_sig = mean - 5 * std, mean + 5 * std

        lo = max(lo_iqr, lo_sig)
        hi = min(hi_iqr, hi_sig)
        cleaned = cleaned[(cleaned[col] >= lo) & (cleaned[col] <= hi)]
        bounds_rows.append([col, f"{lo:.2f}", f"{hi:.2f}"])

    removed = original_count - len(cleaned)

    md_table(["Channel", "Lower Bound", "Upper Bound"], bounds_rows)
    md_table(
        ["Metric", "Value"],
        [
            ["Original samples", original_count],
            ["Cleaned samples", len(cleaned)],
            ["Removed samples", removed],
            ["Removal percentage", f"{removed / original_count * 100:.1f}%"],
        ],
    )
    return cleaned.reset_index(drop=True)

# =============================================================================
# 6. Log-Normalization
# =============================================================================

def section_log_normalization(df):
    title("6. Log-Normalization")
    md_text(
        "Logarithmic normalization compresses the dynamic range of EEG "
        "amplitudes, reducing the impact of extreme values and making "
        "distributions more symmetric. We apply `log10(x - min + 1)` to each "
        "channel to ensure all values are positive before transformation."
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

    plt.suptitle("Effect of Log-Normalization on Channel AF3",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("log_normalization_comparison.png")
    md_image(path, "Log-Normalization Effect")

    md_table(
        ["Channel", "Orig Mean", "Orig Std", "Norm Mean", "Norm Std"],
        [
            [ch,
             f"{df[ch].mean():.2f}", f"{df[ch].std():.2f}",
             f"{df_norm[ch].mean():.4f}", f"{df_norm[ch].std():.4f}"]
            for ch in FEATURE_COLUMNS
        ],
    )
    return df_norm

# =============================================================================
# 8. FFT, Spectrogram and PSD
# =============================================================================

FREQ_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-12 Hz)": (8, 12),
    "Beta (12-30 Hz)": (12, 30),
    "Gamma (30-64 Hz)": (30, 64),
}
BAND_COLORS = ["#8B0000", "#FF4500", "#FFD700", "#00CED1", "#9370DB"]


def section_fft_psd_spectro(df):
    title("8. FFT, Spectrogram and PSD Analysis")
    md_text(
        "Frequency-domain analysis transforms EEG signals from the time domain "
        "to the frequency domain, revealing the power distribution across brain "
        "wave bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** "
        "(8-12 Hz), **Beta** (12-30 Hz), and **Gamma** (30-64 Hz). This is "
        "critical for EEG eye-state classification since alpha power "
        "characteristically increases when eyes are closed (the **Berger effect**)."
    )

    rep = ["AF3", "O1", "T7"]

    # 8.1 FFT -----------------------------------------------------------------
    subtitle("8.1 FFT Frequency Spectrum")
    md_text(
        "The Fast Fourier Transform decomposes EEG signals into constituent "
        "frequencies. The frequency spectrum shows the power at each frequency, "
        "highlighting dominant brain-wave activity."
    )

    fig, axes = plt.subplots(len(rep), 1, figsize=(14, 4 * len(rep)))
    for idx, ch in enumerate(rep):
        signal = df[ch].values
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, 1 / SAMPLING_RATE)
        pos = freqs > 0
        power = np.abs(fft_vals[pos]) ** 2 / n

        axes[idx].semilogy(freqs[pos], power, linewidth=0.6, color="#1f77b4")
        axes[idx].set_xlim(0, 64)
        axes[idx].set_title(f"Frequency Spectrum — {ch}", fontweight="bold")
        axes[idx].set_xlabel("Frequency (Hz)")
        axes[idx].set_ylabel("Power (uV^2)")
        axes[idx].grid(True, alpha=0.3)
        for i, (_, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            axes[idx].axvspan(lo, hi, alpha=0.1, color=BAND_COLORS[i])
    plt.tight_layout()
    path = save_fig("fft_frequency_spectrum.png")
    md_image(path, "FFT Frequency Spectrum")

    # 8.2 PSD -----------------------------------------------------------------
    subtitle("8.2 Power Spectral Density (PSD)")
    md_text(
        "Welch's method estimates the PSD with reduced variance compared to raw "
        "periodograms. Comparing PSD between eyes-open and eyes-closed states "
        "reveals characteristic changes — notably increased alpha power (8-12 Hz) "
        "during eye closure."
    )

    df_open = df[df[TARGET] == 1]
    df_closed = df[df[TARGET] == 0]
    nperseg = min(256, len(df_open), len(df_closed))

    fig, axes = plt.subplots(len(rep), 1, figsize=(14, 4 * len(rep)))
    for idx, ch in enumerate(rep):
        fo, po = welch(df_open[ch].values, SAMPLING_RATE, nperseg=nperseg)
        fc, pc = welch(df_closed[ch].values, SAMPLING_RATE, nperseg=nperseg)
        axes[idx].semilogy(fo, po, label="Eyes Open", color="blue", linewidth=1.5)
        axes[idx].semilogy(fc, pc, label="Eyes Closed", color="red", linewidth=1.5)
        axes[idx].set_xlim(0, 30)
        axes[idx].set_title(f"PSD — {ch}", fontweight="bold")
        axes[idx].set_xlabel("Frequency (Hz)")
        axes[idx].set_ylabel("Power / Freq (uV^2/Hz)")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_fig("psd_analysis.png")
    md_image(path, "PSD Analysis")

    # 8.3 Spectrogram ---------------------------------------------------------
    subtitle("8.3 Spectrogram Analysis")
    md_text(
        "Spectrograms provide a time-frequency representation of EEG signals, "
        "showing how the power at different frequencies evolves over time. This "
        "is particularly useful for identifying transient events and state "
        "transitions — and serves as the input representation for the CNN-based "
        "deep-learning model in Section 11."
    )

    for ch in rep:
        fig, axes_s = plt.subplots(1, 2, figsize=(16, 5))
        for ax, (sname, sval) in zip(axes_s,
                                      [("Eyes Open", 1), ("Eyes Closed", 0)]):
            data = df[df[TARGET] == sval][ch].values
            seg = min(128, len(data) // 4)
            if seg < 4:
                seg = 4
            f, t, Sxx = scipy_spectrogram(
                data, fs=SAMPLING_RATE, nperseg=seg, noverlap=seg // 2)
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                               shading="gouraud", cmap="viridis")
            ax.set_ylim(0, 30)
            ax.set_title(f"{ch} — {sname}", fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(im, ax=ax, label="Power (dB/Hz)")
            for freq in [4, 8, 12, 30]:
                ax.axhline(y=freq, color="white", linestyle="--",
                           linewidth=0.5, alpha=0.5)
        plt.suptitle(f"Spectrogram — {ch}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = save_fig(f"spectrogram_{ch}.png")
        md_image(path, f"Spectrogram {ch}")

# =============================================================================
# 9. PCA and LDA
# =============================================================================

def section_pca_lda(df):
    title("9. PCA and LDA Analysis")
    md_text(
        "Dimensionality-reduction techniques project high-dimensional EEG data "
        "into lower-dimensional spaces while preserving meaningful structure. "
        "**PCA** (unsupervised) maximises variance; **LDA** (supervised) "
        "maximises class separability — both are fundamental tools for "
        "understanding the data geometry before classification."
    )

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9.1 PCA -----------------------------------------------------------------
    subtitle("9.1 PCA")
    md_text(
        "Principal Component Analysis identifies orthogonal directions of "
        "maximum variance. Applied to the 14 EEG channels, PCA reveals how "
        "much of the total signal variance can be captured in fewer dimensions."
    )

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, 15), pca_full.explained_variance_ratio_ * 100,
                color="#3498db", edgecolor="black")
    axes[0].set_title("Explained Variance per Component")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_xticks(range(1, 15))

    axes[1].plot(range(1, 15), cum_var, "o-", color="#e74c3c", linewidth=2)
    axes[1].axhline(y=95, color="gray", linestyle="--", alpha=0.7, label="95% threshold")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_xticks(range(1, 15))
    axes[1].legend()
    plt.tight_layout()
    path = save_fig("pca_variance.png")
    md_image(path, "PCA Variance")

    md_table(
        ["Component", "Variance Explained (%)", "Cumulative (%)"],
        [[f"PC{i+1}",
          f"{pca_full.explained_variance_ratio_[i]*100:.2f}",
          f"{cum_var[i]:.2f}"] for i in range(14)],
    )

    n_95 = int(np.argmax(cum_var >= 95) + 1)
    md_text(f"**{n_95} components** are required to capture >= 95% of total variance.")

    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y, cmap="coolwarm",
                         alpha=0.4, s=10, edgecolors="none")
    ax.set_title("PCA — 2D Projection")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(scatter, label="Eye State")
    path = save_fig("pca_2d_projection.png")
    md_image(path, "PCA 2D Projection")

    # 9.2 LDA -----------------------------------------------------------------
    subtitle("9.2 LDA")
    md_text(
        "Linear Discriminant Analysis finds the projection that maximises the "
        "ratio of between-class to within-class variance. For binary eye-state "
        "classification, LDA yields a single discriminant dimension that "
        "optimally separates the two classes."
    )

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#3498db", "Closed"), (1, "#e74c3c", "Open")]:
        ax.hist(X_lda[y == label], bins=50, alpha=0.6, color=color,
                label=name, edgecolor="black")
    ax.set_title("LDA — 1D Projection")
    ax.set_xlabel("LD1")
    ax.set_ylabel("Frequency")
    ax.legend()
    path = save_fig("lda_1d_projection.png")
    md_image(path, "LDA 1D Projection")

    # 9.3 Clustering Evaluation -----------------------------------------------
    subtitle("9.3 Clustering Evaluation")
    md_text(
        "Clustering metrics quantify how well the reduced representations "
        "separate eye states, independent of the downstream classifier."
    )
    metrics_rows = []
    for name, Xr in [("PCA (2D)", X_pca2),
                      ("LDA (1D)", np.column_stack([X_lda, np.zeros_like(X_lda)]))]:
        sil = silhouette_score(Xr, y)
        db = davies_bouldin_score(Xr, y)
        ch_s = calinski_harabasz_score(Xr, y)
        metrics_rows.append([name, f"{sil:.4f}", f"{db:.4f}", f"{ch_s:.2f}"])
    md_table(
        ["Method", "Silhouette Score (higher better)",
         "Davies-Bouldin Index (lower better)",
         "Calinski-Harabasz Score (higher better)"],
        metrics_rows,
    )

    return X_scaled, y

# =============================================================================
# 10. Machine Learning Classification
# =============================================================================

def section_ml(X_scaled, y):
    title("10. Machine Learning Classification")
    md_text(
        "This section evaluates five classical machine-learning algorithms on "
        "the standardised EEG features using a 70/30 stratified train-test "
        "split. Each model is assessed on accuracy, precision, recall, "
        "F1-score, and AUC-ROC."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

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

    intros = {
        "Logistic Regression":
            "Logistic Regression models the probability of eye state as a "
            "logistic function of the EEG features. It serves as a simple, "
            "interpretable baseline for binary classification.",
        "K-Nearest Neighbors":
            "KNN classifies each sample by majority vote among its k nearest "
            "neighbours in feature space. It makes no assumptions about the "
            "data distribution.",
        "Support Vector Machine":
            "SVM finds the optimal hyperplane that maximises the margin "
            "between classes. With an RBF kernel, it can capture non-linear "
            "decision boundaries.",
        "Random Forest":
            "Random Forest builds an ensemble of decision trees trained on "
            "bootstrapped subsets, reducing overfitting through bagging and "
            "random feature selection.",
        "Gradient Boosting":
            "Gradient Boosting sequentially builds weak learners (trees), "
            "each correcting errors of the previous ensemble. It often "
            "achieves top accuracy on structured/tabular data.",
    }

    all_results = {}
    sec_idx = 1
    for name, model in models.items():
        subtitle(f"10.{sec_idx} {name}")
        md_text(intros[name])
        progress(f"  Training {name} ...")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba") else None)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        all_results[name] = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            **{"F1-Score": f1, "AUC-ROC": auc, "Train Time (s)": train_time})

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{acc:.4f}"],
            ["Precision", f"{prec:.4f}"],
            ["Recall", f"{rec:.4f}"],
            ["F1-Score", f"{f1:.4f}"],
            ["AUC-ROC", f"{auc:.4f}"],
            ["Training Time", f"{train_time:.3f}s"],
        ])
        sec_idx += 1

    # 10.6 Comparison ---------------------------------------------------------
    subtitle("10.6 ML Model Comparison")
    md_text("Summary comparison of all classical ML models.")

    headers = ["Model", "Accuracy", "Precision", "Recall",
               "F1-Score", "AUC-ROC", "Train Time (s)"]
    rows = []
    for n, r in all_results.items():
        rows.append([n, f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
                      f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
                      f"{r['AUC-ROC']:.4f}", f"{r['Train Time (s)']:.3f}"])
    md_table(headers, rows)

    # confusion matrices
    fig, axes_cm = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[idx],
                    xticklabels=["Closed", "Open"],
                    yticklabels=["Closed", "Open"])
        short = name.split()[-1] if len(name) > 15 else name
        axes_cm[idx].set_title(short, fontsize=10)
        axes_cm[idx].set_xlabel("Predicted")
        axes_cm[idx].set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — ML Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("ml_confusion_matrices.png")
    md_image(path, "ML Confusion Matrices")

    # bar chart
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

def _create_windows(df, window_size=64, step=32):
    """Slide a window over sequential EEG rows; label by majority vote."""
    X_data = df[FEATURE_COLUMNS].values
    y_data = df[TARGET].values
    Xw, yw = [], []
    for i in range(0, len(X_data) - window_size, step):
        Xw.append(X_data[i:i + window_size])
        yw.append(int(np.round(np.mean(y_data[i:i + window_size]))))
    return np.array(Xw), np.array(yw)


def _create_spectrogram_windows(df, window_size=128, step=64):
    """Build spectrogram tensors (freq x time x channels) from EEG windows."""
    X_data = df[FEATURE_COLUMNS].values
    y_data = df[TARGET].values
    specs, labels = [], []
    seg = min(32, window_size // 2)
    if seg < 4:
        seg = 4
    ovlp = seg - max(1, seg // 4)

    for i in range(0, len(X_data) - window_size, step):
        window = X_data[i:i + window_size]
        label = int(np.round(np.mean(y_data[i:i + window_size])))
        ch_specs = []
        for c in range(window.shape[1]):
            _, _, Sxx = scipy_spectrogram(
                window[:, c], fs=SAMPLING_RATE, nperseg=seg, noverlap=ovlp)
            ch_specs.append(10 * np.log10(Sxx + 1e-10))
        specs.append(np.stack(ch_specs, axis=-1))
        labels.append(label)
    return np.array(specs), np.array(labels)


def _eval_nn(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred),
        Recall=recall_score(y_true, y_pred),
        **{"F1-Score": f1_score(y_true, y_pred),
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
        "Deep-learning models learn hierarchical feature representations from "
        "raw EEG signals. This section evaluates three architectures: a **1D "
        "CNN** on raw multi-channel EEG windows, a **2D CNN on spectrograms**, "
        "and an **LSTM** (recurrent) network — all trained to predict eye state "
        "from temporal EEG patterns."
    )

    nn_results = {}
    WINDOW = 64
    STEP = 16

    # --- Fallback when TensorFlow is not available ---------------------------
    if not HAS_TF:
        md_text(
            "> **Note:** TensorFlow is not installed. Falling back to "
            "sklearn `MLPClassifier` as the neural-network proxy."
        )

        subtitle("11.1 MLP Neural Network (sklearn fallback)")
        md_text(
            "A multi-layer perceptron with three hidden layers (128-64-32) "
            "serves as the deep-learning baseline when TensorFlow/Keras is "
            "unavailable."
        )

        X = df[FEATURE_COLUMNS].values
        y = df[TARGET].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

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

        # plot MLP loss curve
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

    # --- TensorFlow path -----------------------------------------------------
    progress("  Preparing EEG windows for deep learning ...")

    X_win, y_win = _create_windows(df, window_size=WINDOW, step=STEP)
    # standardise per-channel
    flat = X_win.reshape(-1, X_win.shape[-1])
    sc = StandardScaler().fit(flat)
    X_win = sc.transform(flat).reshape(X_win.shape)

    Xtr, Xte, ytr, yte = train_test_split(
        X_win, y_win, test_size=0.3, random_state=RANDOM_STATE, stratify=y_win)

    md_text(f"Window size = {WINDOW} samples, step = {STEP}. "
            f"Total windows: {len(X_win)} "
            f"(train {len(Xtr)}, test {len(Xte)}).")

    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                  monitor="val_loss")

    # 11.1 1D CNN -------------------------------------------------------------
    subtitle("11.1 1D CNN on Raw EEG")
    md_text(
        "A 1D Convolutional Neural Network processes windows of multi-channel "
        f"EEG data ({WINDOW} samples x 14 channels), learning local temporal "
        "patterns through convolutional filters before classifying eye state."
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
        ["Accuracy",  f"{res1['Accuracy']:.4f}"],
        ["Precision", f"{res1['Precision']:.4f}"],
        ["Recall",    f"{res1['Recall']:.4f}"],
        ["F1-Score",  f"{res1['F1-Score']:.4f}"],
        ["AUC-ROC",   f"{res1['AUC-ROC']:.4f}"],
        ["Training Time", f"{t1:.3f}s"],
    ])
    path = _plot_history(h1, "1D CNN", "cnn1d_training.png")
    md_image(path, "1D CNN Training History")

    # 11.2 CNN on Spectrograms ------------------------------------------------
    subtitle("11.2 CNN on Spectrograms")
    md_text(
        "A 2D CNN processes spectrogram representations of EEG windows — "
        "treating the time-frequency image with channel depth as input. This "
        "leverages the CNN's ability to detect spatial patterns in "
        "frequency-time maps, similar to image classification."
    )
    progress("  Building spectrogram windows ...")

    SPEC_WIN = 128
    SPEC_STEP = 64
    X_spec, y_spec = _create_spectrogram_windows(
        df, window_size=SPEC_WIN, step=SPEC_STEP)

    md_text(f"Spectrogram window = {SPEC_WIN} samples, step = {SPEC_STEP}. "
            f"Shape per sample: {X_spec.shape[1:]} (freq x time x channels). "
            f"Total: {len(X_spec)}.")

    # normalise
    X_spec = (X_spec - X_spec.mean()) / (X_spec.std() + 1e-8)

    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_spec, y_spec, test_size=0.3, random_state=RANDOM_STATE, stratify=y_spec)

    progress("  Training 2D CNN on spectrograms ...")

    es2 = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                   monitor="val_loss")
    m2 = keras.Sequential([
        layers.Input(shape=X_spec.shape[1:]),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h2 = m2.fit(Xtr2, ytr2, epochs=30, batch_size=64,
                validation_split=0.2, callbacks=[es2], verbose=0)
    t2 = time.time() - t0

    yp2 = m2.predict(Xte2, verbose=0).flatten()
    res2 = _eval_nn(yte2, yp2)
    res2["Train Time (s)"] = t2
    nn_results["CNN (Spectrogram)"] = res2

    md_table(["Metric", "Value"], [
        ["Accuracy",  f"{res2['Accuracy']:.4f}"],
        ["Precision", f"{res2['Precision']:.4f}"],
        ["Recall",    f"{res2['Recall']:.4f}"],
        ["F1-Score",  f"{res2['F1-Score']:.4f}"],
        ["AUC-ROC",   f"{res2['AUC-ROC']:.4f}"],
        ["Training Time", f"{t2:.3f}s"],
    ])
    path = _plot_history(h2, "2D CNN (Spectrogram)", "cnn2d_spectrogram_training.png")
    md_image(path, "CNN Spectrogram Training History")

    # 11.3 LSTM ---------------------------------------------------------------
    subtitle("11.3 LSTM / RNN")
    md_text(
        "Long Short-Term Memory networks capture long-range temporal "
        "dependencies in sequential EEG data. Unlike CNNs that focus on local "
        "patterns, LSTMs maintain a memory cell that can selectively retain or "
        f"discard information across the {WINDOW}-sample window — making them "
        "well-suited for modelling brain-state transitions."
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
        ["Accuracy",  f"{res3['Accuracy']:.4f}"],
        ["Precision", f"{res3['Precision']:.4f}"],
        ["Recall",    f"{res3['Recall']:.4f}"],
        ["F1-Score",  f"{res3['F1-Score']:.4f}"],
        ["AUC-ROC",   f"{res3['AUC-ROC']:.4f}"],
        ["Training Time", f"{t3:.3f}s"],
    ])
    path = _plot_history(h3, "LSTM", "lstm_training.png")
    md_image(path, "LSTM Training History")

    # 11.4 Comparison ---------------------------------------------------------
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

    # confusion matrices for NN
    fig, axes_cm = plt.subplots(1, len(nn_results), figsize=(5 * len(nn_results), 4))
    if len(nn_results) == 1:
        axes_cm = [axes_cm]
    nn_preds = {"1D CNN": (yte, yp1), "CNN (Spectrogram)": (yte2, yp2), "LSTM": (yte, yp3)}
    for idx, (nm, (yt, yp)) in enumerate(nn_preds.items()):
        cm = confusion_matrix(yt, (yp > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes_cm[idx],
                    xticklabels=["Closed", "Open"], yticklabels=["Closed", "Open"])
        axes_cm[idx].set_title(nm, fontsize=10)
        axes_cm[idx].set_xlabel("Predicted")
        axes_cm[idx].set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — Neural Networks", fontsize=14, fontweight="bold")
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
        "ranked by F1-Score, and provides a recommendation for the best model."
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

    # final bar chart
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

    # --- Inference -----------------------------------------------------------
    subtitle("12.2 Inference and Recommendation")

    best_name, best_r = sorted_models[0]
    second_name, second_r = sorted_models[1]

    md_text(f"### Best Overall Model: **{best_name}**")
    md_text(
        f"Based on the comprehensive evaluation, **{best_name}** achieves the "
        f"highest F1-Score of **{best_r['F1-Score']:.4f}** with an accuracy of "
        f"**{best_r['Accuracy']:.4f}** and AUC-ROC of "
        f"**{best_r['AUC-ROC']:.4f}**."
    )
    md_text(
        f"The runner-up is **{second_name}** with an F1-Score of "
        f"**{second_r['F1-Score']:.4f}**."
    )

    md_text("**Key Observations:**")
    md_text("")

    best_ml = max(ml_results.items(), key=lambda x: x[1]["F1-Score"])
    best_nn = max(nn_results.items(), key=lambda x: x[1]["F1-Score"])

    if best_nn[1]["F1-Score"] > best_ml[1]["F1-Score"]:
        diff = (best_nn[1]["F1-Score"] - best_ml[1]["F1-Score"]) * 100
        md_text(
            f"- Deep learning (**{best_nn[0]}**) outperforms the best "
            f"classical ML model (**{best_ml[0]}**) by **{diff:.2f}** "
            f"percentage points in F1-Score."
        )
        md_text(
            "- The neural network's ability to learn temporal patterns from "
            "raw EEG windows provides an advantage over feature-level "
            "classification."
        )
    else:
        diff = (best_ml[1]["F1-Score"] - best_nn[1]["F1-Score"]) * 100
        md_text(
            f"- The classical ML model (**{best_ml[0]}**) matches or "
            f"outperforms deep learning (**{best_nn[0]}**) by **{diff:.2f}** "
            f"percentage points in F1-Score."
        )
        md_text(
            "- For this dataset size, ensemble tree methods capture the "
            "relevant patterns without requiring the architectural complexity "
            "of deep learning."
        )

    md_text(
        f"- **For production deployment**, **{best_name}** is recommended "
        f"when maximum classification performance is required."
    )

    fastest = min(all_res.items(), key=lambda x: x[1]["Train Time (s)"])
    md_text(
        f"- **For real-time / low-latency applications**, **{fastest[0]}** "
        f"offers the fastest training ({fastest[1]['Train Time (s)']:.3f}s) "
        f"with an F1-Score of {fastest[1]['F1-Score']:.4f}."
    )

    md_text("")
    md_text("---")
    md_text("*Report generated automatically by the EEG Eye State "
            "Classification Pipeline.*")

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
    section_data_viz(df, suffix="raw", sec="3",
                     sec_title="Data Visualization (Raw Data)")

    # 4. Outlier removal
    progress("[4/12] Removing outliers ...")
    df_clean = section_outlier_removal(df)

    # 5. Visualization (after outlier removal)
    progress("[5/12] Visualising cleaned data ...")
    section_data_viz(df_clean, suffix="cleaned", sec="5",
                     sec_title="Data Visualization (After Outlier Removal)")

    # 6. Log-normalisation
    progress("[6/12] Log-normalising ...")
    df_norm = section_log_normalization(df_clean)

    # 7. Visualization (after normalisation)
    progress("[7/12] Visualising normalised data ...")
    section_data_viz(df_norm, suffix="normalized", sec="7",
                     sec_title="Data Visualization (After Normalization)")

    # 8. FFT / PSD / Spectrogram
    progress("[8/12] Frequency-domain analysis ...")
    section_fft_psd_spectro(df_clean)

    # 9. PCA / LDA
    progress("[9/12] PCA and LDA ...")
    X_scaled, y = section_pca_lda(df_clean)

    # 10. ML classifiers
    progress("[10/12] Training ML models ...")
    ml_results = section_ml(X_scaled, y)

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
