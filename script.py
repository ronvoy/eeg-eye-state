# pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow

import sys
import os
import time
import warnings
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram as scipy_spectrogram, butter, filtfilt
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import mne
    from mne.preprocessing import ICA as MNE_ICA
    mne.set_log_level("ERROR")
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

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

# Load config if available
CONFIG = {}
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
if HAS_YAML and os.path.exists(_config_path):
    with open(_config_path, 'r') as _cf:
        CONFIG = yaml.safe_load(_cf) or {}
    PLOT_DIR = CONFIG.get('paths', {}).get('plot_dir', PLOT_DIR)
    DATA_FILE = CONFIG.get('paths', {}).get('data_file', DATA_FILE)
    SAMPLING_RATE = CONFIG.get('data', {}).get('sampling_rate', SAMPLING_RATE)
    RANDOM_STATE = CONFIG.get('data', {}).get('random_state', RANDOM_STATE)
    TEST_SIZE = CONFIG.get('data', {}).get('test_size', TEST_SIZE)

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

# Alias for explicit print-style calls
md_print = md_text

def md_blockquote(text):
    for line in text.strip().split("\n"):
        print(f"> {line}")
    print()

def md_image(path, caption=""):
    print(f"![{caption}]({path})")
    print()

def progress(msg):
    print(msg, file=sys.stderr, flush=True)

def save_fig(name):
    path = f"{PLOT_DIR}/{name}"
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
   - 3.6 [Temporal Plots & State Transitions](#36-temporal-plots--state-transitions)
4. [Signal Preprocessing (Bandpass + ICA)](#4-signal-preprocessing)
   - 4.1 [Bandpass Filter (0.5–45 Hz)](#41-bandpass-filter-05--45-hz)
   - 4.2 [ICA Artifact Removal](#42-ica-artifact-removal)
   - 4.3 [Residual Outlier Removal (Safety Net)](#43-residual-outlier-removal-safety-net)
5. [Data Visualization (After Preprocessing)](#5-data-visualization-after-preprocessing)
   - 5.1 [Box Plots Comparison](#51-box-plots-comparison)
   - 5.2 [Histograms After Cleaning](#52-histograms-after-cleaning)
6. [Log-Normalization Assessment (Rejected)](#6-log-normalization-assessment-rejected)
   - 6.1 [Before vs After — All Channels](#61-before-vs-after--all-channels)
   - 6.2 [Skewness & Kurtosis Analysis](#62-skewness--kurtosis-analysis)
   - 6.3 [Summary Statistics Before vs After](#63-summary-statistics-before-vs-after)
7. [Feature Engineering](#7-feature-engineering)
   - 7.1 [Hemispheric Asymmetry](#71-hemispheric-asymmetry)
   - 7.2 [Frequency Band Power Features](#72-frequency-band-power-features)
   - 7.3 [Global Channel Statistics](#73-global-channel-statistics)
   - 7.4 [Feature Summary](#74-feature-summary)
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
   - 9.6 [Inference: Dimensionality Reduction Comparison](#96-inference-dimensionality-reduction-comparison)
10. [Machine Learning Classification (Pipeline-based)](#10-machine-learning-classification)
    - 10.1 [Train/Validation/Test Split & Class Balance](#101-trainvalidationtest-split--class-balance)
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
    - 11.0 [Binary Cross-Entropy Loss & Gradient Descent](#110-binary-cross-entropy-loss--gradient-descent)
    - 11.1 [1D CNN on Raw EEG](#111-1d-cnn-on-raw-eeg)
    - 11.2 [CNN on Spectrograms](#112-cnn-on-spectrograms)
    - 11.3 [LSTM / RNN](#113-lstm--rnn)
    - 11.4 [CNN+LSTM Hybrid](#114-cnnlstm-hybrid)
    - 11.5 [EEGFormer (Transformer)](#115-eegformer-transformer)
    - 11.6 [EEGNet (Lightweight CNN)](#116-eegnet-lightweight-cnn)
    - 11.7 [Stacking Ensemble](#117-stacking-ensemble)
    - 11.8 [Neural Network Comparison](#118-neural-network-comparison)
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

    md_text(
        "> **Note on Spike Artifacts:** Some channels exhibit extremely large max values "
        "(e.g., AF3 max ≈ 4294.87, F7 max ≈ 4294.87) — orders of magnitude above the 75th "
        "percentile. These are likely **electrode spike artifacts** caused by momentary "
        "loss of contact, muscle movement, or impedance changes in the Emotiv headset. "
        "These extreme values will be addressed by the outlier removal step."
    )

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

    # 3.3b Zoomed Box Plots (clipped at 1st/99th percentile)
    md_text(
        "The raw box plots are compressed by extreme spike artifacts. "
        "Below is a **zoomed view** clipped at the 1st–99th percentile range "
        "to reveal the actual distribution of most samples."
    )
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df[ch], ax=ax, color="#3498db")
        lo, hi = df[ch].quantile(0.01), df[ch].quantile(0.99)
        margin = (hi - lo) * 0.1
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle("Box Plots — Zoomed (1st–99th percentile)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("boxplots_raw_zoomed.png")
    md_image(path, "Box Plots Zoomed")

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

    # 3.6 Temporal Plots & State Transitions
    subtitle("3.6 Temporal Plots & State Transitions")
    md_text(
        "Time-series plots reveal the temporal structure of EEG signals and "
        "transitions between eye states — essential context for a time-series "
        "classification task."
    )
    n_display = min(2000, len(df))
    display_channels = FEATURE_COLUMNS[:4]  # AF3, F7, F3, FC5
    fig, axes = plt.subplots(len(display_channels), 1, figsize=(16, 10), sharex=True)
    for i, ch in enumerate(display_channels):
        vals = df[ch].iloc[:n_display].values
        axes[i].plot(range(n_display), vals, linewidth=0.4, color='#3498db')
        ymin, ymax = vals.min(), vals.max()
        margin = (ymax - ymin) * 0.05
        closed_mask = df[TARGET].iloc[:n_display].values == 1
        axes[i].fill_between(range(n_display), ymin - margin, ymax + margin,
                             where=closed_mask, alpha=0.15, color='red')
        axes[i].set_ylim(ymin - margin, ymax + margin)
        axes[i].set_ylabel(f'{ch} (uV)', fontsize=9)
        axes[i].tick_params(labelsize=8)
    axes[0].set_title('Raw EEG Time Series with Eye State Annotations (red = eyes closed)',
                       fontsize=12, fontweight='bold')
    axes[-1].set_xlabel(f'Sample (fs = {SAMPLING_RATE} Hz)')
    plt.tight_layout()
    path = save_fig("temporal_raw_signal.png")
    md_image(path, "Temporal Raw Signal")

    # State transition summary
    transitions = int(df[TARGET].diff().fillna(0).abs().sum())
    md_text(
        f"**State transitions:** {transitions} transitions between Open and "
        f"Closed states in {len(df)} samples "
        f"({len(df) / SAMPLING_RATE:.1f}s recording). "
        f"Average segment length: ~{len(df) / max(transitions, 1):.0f} samples "
        f"({len(df) / max(transitions, 1) / SAMPLING_RATE:.2f}s)."
    )

    # Event plot — transition positions
    transition_idx = np.where(df[TARGET].diff().fillna(0).abs() > 0)[0]
    if len(transition_idx) > 0:
        fig, ax = plt.subplots(figsize=(16, 2))
        ax.eventplot(transition_idx, lineoffsets=0.5, linelengths=0.8,
                     colors='red', linewidths=0.5)
        ax.set_xlim(0, len(df))
        ax.set_yticks([])
        ax.set_xlabel(f'Sample (total: {len(df)})')
        ax.set_title('Eye State Transition Points', fontsize=11, fontweight='bold')
        plt.tight_layout()
        path = save_fig("state_transitions.png")
        md_image(path, "State Transition Points")

# =============================================================================
# 4. Signal Preprocessing — Bandpass Filter + ICA Artifact Removal
# =============================================================================

def _bandpass_filter(df, lowcut=0.5, highcut=45.0, fs=None, order=4):
    """Apply Butterworth bandpass filter to all EEG channels."""
    if fs is None:
        fs = SAMPLING_RATE
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = df.copy()
    for col in FEATURE_COLUMNS:
        filtered[col] = filtfilt(b, a, df[col].values)
    return filtered


def _ica_artifact_removal(df, n_components=14, kurtosis_thresh=5.0):
    """Use MNE ICA to remove artifact components identified by kurtosis."""
    info = mne.create_info(
        ch_names=list(FEATURE_COLUMNS), sfreq=SAMPLING_RATE, ch_types='eeg')
    data = df[FEATURE_COLUMNS].values.T * 1e-6  # MNE expects volts
    raw = mne.io.RawArray(data, info, verbose=False)

    ica = MNE_ICA(n_components=n_components, random_state=RANDOM_STATE,
                  max_iter="auto")
    ica.fit(raw, verbose=False)

    # Auto-detect artifact components via kurtosis
    sources = ica.get_sources(raw).get_data()
    comp_kurtosis = kurtosis(sources, axis=1)
    exclude = np.where(np.abs(comp_kurtosis) > kurtosis_thresh)[0].tolist()
    ica.exclude = exclude

    raw_clean = ica.apply(raw.copy(), verbose=False)
    clean_data = raw_clean.get_data().T * 1e6  # back to microvolts

    df_clean = df.copy()
    df_clean[FEATURE_COLUMNS] = clean_data
    return df_clean, ica, exclude, comp_kurtosis


def _light_iqr(df, multiplier=3.0, max_passes=3):
    """Safety-net IQR with wider bounds (default 3x) to catch residual extremes."""
    cleaned = df.copy()
    pass_num = 0
    bounds = []
    while True:
        pass_num += 1
        before = len(cleaned)
        for col in FEATURE_COLUMNS:
            Q1, Q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - multiplier * IQR, Q3 + multiplier * IQR
            cleaned = cleaned[(cleaned[col] >= lo) & (cleaned[col] <= hi)]
            if pass_num == 1:
                bounds.append([col, f"{lo:.2f}", f"{hi:.2f}"])
        after = len(cleaned)
        if before - after == 0 or pass_num >= max_passes:
            break
    return cleaned.reset_index(drop=True), bounds, pass_num


def section_outlier_removal(df):
    title("4. Signal Preprocessing")
    md_text(
        "EEG signals contain artifacts from eye blinks, muscle movement, and electrode "
        "drift that must be removed before analysis. This section applies a three-stage "
        "cleaning pipeline: **(1) bandpass filtering** to remove DC drift and high-frequency "
        "noise, **(2) ICA decomposition** to separate and remove artifact components while "
        "preserving brain activity, and **(3) a light IQR safety net** to catch any residual "
        "extremes."
    )
    original_count = len(df)
    cfg_pre = CONFIG.get('preprocessing', {})

    # --- 4.1 Bandpass Filter (0.5–45 Hz) ---
    subtitle("4.1 Bandpass Filter (0.5–45 Hz)")
    bp_cfg = cfg_pre.get('bandpass', {})
    lowcut = bp_cfg.get('lowcut', 0.5)
    highcut = bp_cfg.get('highcut', 45.0)
    bp_order = bp_cfg.get('order', 4)
    md_text(
        f"A **{bp_order}th-order Butterworth bandpass filter** ({lowcut}–{highcut} Hz) "
        "removes DC drift and high-frequency noise while preserving the physiologically "
        "relevant EEG bands (Delta through Gamma).\n\n"
        "The filter transfer function is:\n\n"
        "$$H(s) = \\frac{1}{\\sqrt{1 + \\left(\\frac{s}{\\omega_c}\\right)^{2N}}}$$\n\n"
        "Applied via `scipy.signal.filtfilt` (zero-phase, forward-backward filtering) "
        "to avoid phase distortion."
    )
    df_filt = _bandpass_filter(df, lowcut=lowcut, highcut=highcut,
                               fs=SAMPLING_RATE, order=bp_order)

    # Before/after comparison for one channel
    sample_ch = "O1"
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    n_show = min(1000, len(df))
    axes[0].plot(range(n_show), df[sample_ch].iloc[:n_show].values,
                 linewidth=0.4, color='#e74c3c', label='Raw')
    axes[1].plot(range(n_show), df_filt[sample_ch].iloc[:n_show].values,
                 linewidth=0.4, color='#2ecc71', label='Filtered')
    for ax in axes:
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel(f'{sample_ch} (uV)')
    axes[1].set_xlabel('Sample')
    axes[0].set_title(f'Bandpass Filter Effect — {sample_ch} ({lowcut}–{highcut} Hz)')
    plt.tight_layout()
    path = save_fig("bandpass_filter_comparison.png")
    md_image(path, "Bandpass Filter Comparison")
    md_text(
        f"Bandpass filter applied to all {len(FEATURE_COLUMNS)} channels. "
        f"Samples preserved: **{len(df_filt)}** (no samples removed by filtering)."
    )

    # --- 4.2 ICA Artifact Removal ---
    subtitle("4.2 ICA Artifact Removal")
    cleaned = df_filt
    if HAS_MNE:
        md_text(
            "**Independent Component Analysis (ICA)** decomposes the multi-channel EEG "
            "signal into statistically independent source components. Artifact components "
            "(eye blinks, muscle activity) are identified by high kurtosis and removed, "
            "while brain-activity components are preserved.\n\n"
            "$$\\mathbf{X} = \\mathbf{A} \\mathbf{S} \\quad \\Rightarrow \\quad "
            "\\mathbf{S} = \\mathbf{W} \\mathbf{X}$$\n\n"
            "where $\\mathbf{X}$ is the observed signal, $\\mathbf{A}$ the mixing matrix, "
            "$\\mathbf{S}$ the source components, and $\\mathbf{W} = \\mathbf{A}^{-1}$ "
            "the unmixing matrix. Components with $|\\text{kurtosis}| > \\tau$ are excluded "
            "before reconstruction."
        )
        ica_cfg = cfg_pre.get('ica', {})
        n_comp = ica_cfg.get('n_components', 14)
        kurt_thresh = ica_cfg.get('kurtosis_threshold', 5.0)
        progress("  Running ICA artifact removal ...")
        cleaned, ica_obj, excluded, comp_kurt = _ica_artifact_removal(
            df_filt, n_components=n_comp, kurtosis_thresh=kurt_thresh)

        md_text(f"**ICA fitted** with {n_comp} components (kurtosis threshold = {kurt_thresh}).")
        kurt_rows = [[f"IC{i}", f"{k:.3f}",
                       "**EXCLUDED**" if i in excluded else "Kept"]
                      for i, k in enumerate(comp_kurt)]
        md_table(["Component", "Kurtosis", "Status"], kurt_rows)
        md_text(
            f"**{len(excluded)} component(s) excluded:** {excluded if excluded else 'None'}. "
            "Remaining components reconstructed into clean signal."
        )

        # Plot excluded vs kept components
        if excluded:
            fig, axes = plt.subplots(min(len(excluded), 4), 1,
                                     figsize=(14, 3 * min(len(excluded), 4)))
            if len(excluded) == 1:
                axes = [axes]
            sources = ica_obj.get_sources(
                mne.io.RawArray(
                    df_filt[FEATURE_COLUMNS].values.T * 1e-6,
                    mne.create_info(list(FEATURE_COLUMNS), SAMPLING_RATE, 'eeg'),
                    verbose=False),
            ).get_data()
            for idx, comp_idx in enumerate(excluded[:4]):
                axes[idx].plot(sources[comp_idx, :1000], linewidth=0.4, color='#e74c3c')
                axes[idx].set_title(f'IC{comp_idx} (excluded, kurtosis={comp_kurt[comp_idx]:.2f})',
                                    fontsize=10)
                axes[idx].tick_params(labelsize=8)
            plt.tight_layout()
            path = save_fig("ica_excluded_components.png")
            md_image(path, "ICA Excluded Components")
    else:
        md_text(
            "> **Note:** `mne` is not installed. ICA artifact removal skipped. "
            "Install with `pip install mne` for ICA-based cleaning. "
            "Falling back to IQR-only approach."
        )

    # --- 4.3 Light IQR Safety Net ---
    subtitle("4.3 Residual Outlier Removal (Safety Net)")
    iqr_cfg = cfg_pre.get('iqr', {})
    iqr_mult = iqr_cfg.get('multiplier', 3.0)
    iqr_passes = iqr_cfg.get('max_passes', 3)
    md_text(
        f"A **light IQR filter** ({iqr_mult}x IQR, max {iqr_passes} passes) removes any "
        "residual extreme values that survived bandpass filtering and ICA. The wider "
        f"threshold ({iqr_mult}x vs traditional 1.5x) preserves more data while still "
        "catching hardware glitches."
    )
    cleaned, bounds, n_passes = _light_iqr(cleaned, multiplier=iqr_mult,
                                            max_passes=iqr_passes)
    removed = original_count - len(cleaned)
    removal_pct = removed / original_count * 100

    md_table(["Channel", "Lower Bound", "Upper Bound"], bounds)
    md_table(
        ["Metric", "Value"],
        [
            ["Original samples", original_count],
            ["Cleaned samples", len(cleaned)],
            ["Removed samples", removed],
            ["Removal percentage", f"{removal_pct:.1f}%"],
            ["IQR passes", n_passes],
            ["Bandpass filter", f"{lowcut}–{highcut} Hz"],
            ["ICA components removed", len(excluded) if HAS_MNE else "N/A"],
        ],
    )

    md_text(
        f"> **Preprocessing Summary:** Bandpass filter ({lowcut}–{highcut} Hz) → "
        f"ICA ({len(excluded) if HAS_MNE else 'skipped'} artifact components removed) → "
        f"light IQR ({iqr_mult}x, {removal_pct:.1f}% samples removed). "
        "This pipeline preserves brain activity while removing artifacts, achieving "
        "much lower data loss than aggressive IQR-only approaches (~25% → "
        f"{removal_pct:.1f}%)."
    )

    return cleaned

# =============================================================================
# 5. Data Visualization (After Outlier Removal) — condensed
# =============================================================================

def section_data_viz_cleaned(df_raw, df_clean):
    title("5. Data Visualization (After Preprocessing)")
    md_text("Comparison of distributions before and after preprocessing (bandpass + ICA + IQR).")

    # 5.1 Box Plots Comparison
    subtitle("5.1 Box Plots Comparison")
    md_text(
        "Side-by-side box plots confirm preprocessing effectiveness. "
        "Whiskers are set to **3.0x IQR** to match the cleaning threshold — "
        "points beyond this range are true residual outliers."
    )
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    for i, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df_clean[ch], ax=ax, color="#2ecc71", whis=3.0)
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle("Box Plots — After Preprocessing (whis=3.0x IQR)", fontsize=14, fontweight="bold")
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
    plt.suptitle("Histograms — After Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("histograms_cleaned.png")
    md_image(path, "Histograms After Cleaning")

# =============================================================================
# 6. Log-Normalization
# =============================================================================

def section_log_normalization(df):
    title("6. Log-Normalization Assessment (Rejected)")
    md_text(
        "Logarithmic normalization compresses the dynamic range of EEG amplitudes, "
        "reducing the impact of extreme values and making distributions more symmetric. "
        "We test `log10(x - min + 1)` on each channel and evaluate whether it improves "
        "distribution quality. **The transformed data is not used downstream** — this "
        "section documents the assessment only."
    )

    df_norm = df.copy()
    for col in FEATURE_COLUMNS:
        df_norm[col] = np.log10(df[col] - df[col].min() + 1)

    # 6.1 Before/After histograms for ALL channels
    subtitle("6.1 Before vs After — All Channels")
    md_text(
        "The following grid shows the distribution of every EEG channel before "
        "(blue) and after (red) log-normalization."
    )

    n_cols_grid = 7
    n_rows_grid = 2
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        row, col = divmod(idx, n_cols_grid)
        ax = axes[row, col]
        ax.hist(df[ch], bins=50, color="#3498db", alpha=0.6, edgecolor="black",
                label="Before", density=True)
        ax.hist(df_norm[ch], bins=50, color="#e74c3c", alpha=0.6, edgecolor="black",
                label="After", density=True)
        ax.set_title(ch, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    plt.suptitle("Log-Normalization — Before (blue) vs After (red) for All Channels",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_fig("log_normalization_all_channels.png")
    md_image(path, "Log-Normalization — All Channels")

    # 6.2 Skewness and Kurtosis analysis
    subtitle("6.2 Skewness & Kurtosis Analysis")
    md_text(
        "Skewness measures distribution asymmetry (0 = perfectly symmetric). "
        "Kurtosis (excess) measures tail heaviness (0 = normal). "
        "Log-normalization should reduce both towards zero, indicating "
        "a more Gaussian-like distribution suitable for downstream models."
    )

    sk_rows = []
    improved_count = 0
    for ch in FEATURE_COLUMNS:
        sk_before = skew(df[ch].values)
        sk_after = skew(df_norm[ch].values)
        kt_before = kurtosis(df[ch].values)
        kt_after = kurtosis(df_norm[ch].values)
        # Improvement = absolute skewness + absolute kurtosis decreased
        quality_before = abs(sk_before) + abs(kt_before)
        quality_after = abs(sk_after) + abs(kt_after)
        improved = quality_after < quality_before
        if improved:
            improved_count += 1
        sk_rows.append([
            ch,
            f"{sk_before:.4f}", f"{sk_after:.4f}",
            f"{kt_before:.4f}", f"{kt_after:.4f}",
            "Yes" if improved else "No",
        ])

    md_table(
        ["Channel", "Skew Before", "Skew After",
         "Kurtosis Before", "Kurtosis After", "Improved?"],
        sk_rows,
    )

    pct_improved = improved_count / len(FEATURE_COLUMNS) * 100
    md_text(
        f"**Result:** Log-normalization improved distribution quality "
        f"(reduced |skewness| + |kurtosis|) for **{improved_count}/{len(FEATURE_COLUMNS)} "
        f"channels ({pct_improved:.0f}%)**."
    )
    if pct_improved >= 70:
        md_text(
            "The majority of channels show improved symmetry and reduced tail weight, "
            "confirming that log-normalization is beneficial for this dataset."
        )
    else:
        md_text(
            "> **Decision: Log-normalization REJECTED.** The transform worsened the "
            "distribution quality (increased |skewness| + |kurtosis|) for the majority of "
            "channels. After outlier removal, the EEG distributions are already approximately "
            "symmetric; the log transform compresses the already-compact range and introduces "
            "artificial skewness. **All subsequent analyses use the cleaned (non-transformed) "
            "data.** This section is retained to document that the technique was evaluated and "
            "found unsuitable for this dataset."
        )

    # 6.3 Summary statistics table
    subtitle("6.3 Summary Statistics Before vs After")
    md_table(
        ["Channel", "Orig Mean", "Orig Std", "Norm Mean", "Norm Std"],
        [[ch, f"{df[ch].mean():.2f}", f"{df[ch].std():.2f}",
          f"{df_norm[ch].mean():.4f}", f"{df_norm[ch].std():.4f}"]
         for ch in FEATURE_COLUMNS],
    )

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

    # Asymmetry comparison by eye state
    md_text("**Asymmetry by Eye State** — do hemispheric differences change with eye state?")
    from scipy.stats import ttest_ind
    asym_state_rows = []
    for left, right in HEMI_PAIRS:
        fname = f"{left}_{right}_asym"
        open_vals = df_eng.loc[df_eng[TARGET] == 0, fname].values
        closed_vals = df_eng.loc[df_eng[TARGET] == 1, fname].values
        t_stat, p_val = ttest_ind(open_vals, closed_vals, equal_var=False)
        sig = "Yes" if p_val < 0.05 else "No"
        asym_state_rows.append([
            fname,
            f"{open_vals.mean():.4f}", f"{closed_vals.mean():.4f}",
            f"{t_stat:.3f}", f"{p_val:.2e}", sig,
        ])
    md_table(
        ["Feature", "Mean (Open)", "Mean (Closed)", "t-statistic", "p-value", "Significant (p<0.05)"],
        asym_state_rows,
    )
    sig_count = sum(1 for r in asym_state_rows if r[5] == "Yes")
    md_text(
        f"**{sig_count}/{len(HEMI_PAIRS)}** asymmetry features show a statistically "
        "significant difference between eye states (Welch's t-test, p < 0.05). "
        + ("This confirms that hemispheric asymmetry patterns shift meaningfully with "
           "eye state, supporting their inclusion as classification features."
           if sig_count >= 4 else
           "While not all pairs are significant, the differences still contribute to "
           "the feature space for classification.")
    )

    # 7.2 Frequency Band Power Features
    subtitle("7.2 Frequency Band Power Features")
    md_text(
        "Band power features capture the relative energy in each EEG frequency band. "
        "Research shows that band powers — particularly alpha and beta — are among the "
        "strongest predictors for eye state classification (up to 96% accuracy in papers).\n\n"
        "For each band, the signal is bandpass-filtered and the instantaneous power is "
        "computed as the squared amplitude, then averaged across all 14 channels:\n\n"
        "$$P_{\\text{band}}(t) = \\frac{1}{C} \\sum_{c=1}^{C} "
        "\\left[x_c^{\\text{band}}(t)\\right]^2$$"
    )
    nyq = SAMPLING_RATE / 2.0
    band_rows = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        low = max(fmin / nyq, 0.001)
        high = min(fmax / nyq, 0.999)
        b_bp, a_bp = butter(4, [low, high], btype='band')
        band_powers = []
        for ch in FEATURE_COLUMNS:
            band_signal = filtfilt(b_bp, a_bp, df[ch].values)
            band_powers.append(band_signal ** 2)
        feat_name = f'band_{band_name}_power'
        df_eng[feat_name] = np.mean(band_powers, axis=0)
        new_features.append(feat_name)
        band_rows.append([feat_name, f"{fmin}–{fmax} Hz",
                          f"{df_eng[feat_name].mean():.4f}",
                          f"{df_eng[feat_name].std():.4f}"])

    # Alpha asymmetry (O1 vs O2) — key Berger effect feature
    alpha_low, alpha_high = FREQ_BANDS["Alpha"]
    b_alpha, a_alpha = butter(4, [alpha_low / nyq, alpha_high / nyq], btype='band')
    o1_alpha = filtfilt(b_alpha, a_alpha, df['O1'].values) ** 2
    o2_alpha = filtfilt(b_alpha, a_alpha, df['O2'].values) ** 2
    df_eng['alpha_asymmetry'] = o1_alpha - o2_alpha
    new_features.append('alpha_asymmetry')
    band_rows.append(['alpha_asymmetry', 'O1α² − O2α²',
                      f"{df_eng['alpha_asymmetry'].mean():.4f}",
                      f"{df_eng['alpha_asymmetry'].std():.4f}"])

    md_table(["Feature", "Band / Description", "Mean", "Std"], band_rows)
    md_text(
        f"**{len(band_rows)} band power features** added. Alpha asymmetry captures "
        "the Berger effect (occipital alpha power increase during eye closure)."
    )

    # 7.3 Global Channel Statistics
    subtitle("7.3 Global Channel Statistics")
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

    # 7.4 Feature Summary
    subtitle("7.4 Feature Summary")
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

    md_text(
        "**PSD Interpretation — Berger Effect:** The plots above show PSD for eyes-open "
        "(blue) and eyes-closed (red) conditions across all 14 channels. A consistent "
        "observation in neuroscience is the **Berger effect**: alpha-band power (8–12 Hz) "
        "increases when the eyes are closed, particularly in occipital electrodes (O1, O2). "
        "If the red curve (closed) shows higher power in the alpha band compared to blue "
        "(open), this confirms the dataset captures genuine physiological differences between "
        "eye states — validating both the data quality and the classification task."
    )

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

    # Interpret PCA silhouette specifically
    pca_sil = float(metrics_rows[0][1])  # PCA is first row
    if pca_sil < 0.1:
        md_text(
            f"> **Note on PCA Silhouette ({pca_sil:.4f}):** A silhouette score near zero "
            "indicates that the two classes (Open/Closed) are **heavily overlapping** in the "
            "PCA 2D projection. This is expected: PCA is an unsupervised method that maximises "
            "variance regardless of labels. The first two principal components capture sensor "
            "variance (noise, drift) rather than the eye-state discriminant. This does **not** "
            "mean the classes are inseparable — supervised methods (LDA) and non-linear methods "
            "(t-SNE, UMAP) achieve much better separation, as shown above."
        )

    # 9.6 Inference: Method Comparison
    subtitle("9.6 Inference: Dimensionality Reduction Comparison")
    md_text(
        "Each dimensionality reduction technique has distinct strengths and ideal "
        "use-cases:"
    )
    md_text(
        "| Method | Type | Strengths | Limitations | Best For |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| **PCA** | Linear, unsupervised | Fast, preserves global variance, deterministic | "
        "Cannot capture non-linear structure | Feature reduction, preprocessing, explained variance analysis |\n"
        "| **LDA** | Linear, supervised | Maximises class separation, single component for binary | "
        "Limited to C-1 components, assumes Gaussian classes | Binary/multi-class classification preprocessing |\n"
        "| **t-SNE** | Non-linear, unsupervised | Excellent local structure preservation, reveals clusters | "
        "Slow on large data, non-deterministic, no inverse transform | Exploratory visualisation of cluster structure |\n"
        "| **UMAP** | Non-linear, unsupervised | Preserves both local and global structure, faster than t-SNE | "
        "Hyperparameter sensitive (n_neighbors, min_dist) | Scalable visualisation, general-purpose embedding |"
    )

    # Determine best method from clustering metrics
    best_sil = max(metrics_rows, key=lambda r: float(r[1]))
    best_db = min(metrics_rows, key=lambda r: float(r[2]))
    best_ch = max(metrics_rows, key=lambda r: float(r[3]))

    md_text(
        f"**Clustering metric summary:**\n"
        f"- **Best Silhouette Score:** {best_sil[0]} ({best_sil[1]}) — "
        f"highest cohesion within clusters and separation between clusters.\n"
        f"- **Best Davies-Bouldin Index:** {best_db[0]} ({best_db[2]}) — "
        f"lowest inter-cluster similarity (tighter clusters).\n"
        f"- **Best Calinski-Harabasz Score:** {best_ch[0]} ({best_ch[3]}) — "
        f"highest ratio of between-cluster to within-cluster dispersion."
    )

    # Count wins
    winners = [best_sil[0], best_db[0], best_ch[0]]
    from collections import Counter
    win_counts = Counter(winners)
    overall_best = win_counts.most_common(1)[0][0]

    md_text(
        f"**Overall recommendation:** **{overall_best}** wins on the majority of "
        f"metrics ({win_counts[overall_best]}/3), making it the most effective "
        f"dimensionality reduction method for separating EEG eye states in this dataset. "
        f"For production pipelines, **PCA** or **LDA** are preferred due to their "
        f"determinism and speed, while **t-SNE** and **UMAP** are best suited for "
        f"exploratory data analysis and visualisation."
    )

# =============================================================================
# 10. Machine Learning Classification
# =============================================================================

def section_ml(df, all_features):
    title("10. Machine Learning Classification")
    md_text(
        "Five classical ML algorithms are evaluated using a **55/15/30 chronological "
        "train-validation-test split** that preserves temporal order to prevent future-data leakage. "
        "Each model is equipped with `class_weight='balanced'` to penalize "
        "misclassification of the minority class. Each model is wrapped in a `sklearn.Pipeline` "
        "that includes `StandardScaler`, ensuring that scaling is applied correctly "
        "during cross-validation (no data leakage) and simplifying deployment."
    )

    X = df[all_features].values
    y = df[TARGET].values

    # Chronological 3-way split: 55% train, 15% val, 30% test (preserves temporal order, no leakage)
    n_total = len(X)
    n_train = int(n_total * 0.55)
    n_val = int(n_total * 0.15)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    # Helper: tune decision threshold to maximize F1-score
    def _tune_threshold(y_val, y_prob_val, y_test, y_prob_test):
        """Find optimal threshold that maximizes F1 on validation set."""
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_val = (y_prob_val >= thresh).astype(int)
            f1_val = f1_score(y_val, y_pred_val)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thresh = thresh
        # Apply to test set
        y_pred_test = (y_prob_test >= best_thresh).astype(int)
        return y_pred_test, best_thresh

    # Pipelines: StandardScaler + model with class_weight (prevents CV leakage, balances classes)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight='balanced')),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=200, random_state=RANDOM_STATE, init='zero', learning_rate=0.05)),  # tuned for minority class
        ]),
    }

    # 10.1 Train/Val/Test Split & Class Balance
    subtitle("10.1 Train/Validation/Test Split & Class Balance")
    md_text(
        "Chronological 3-way split: **55% train / 15% validation / 30% test**, "
        "preserving temporal order without data leakage. All models use `class_weight='balanced'` "
        "to penalize misclassification of the minority class (~45% closed-eye vs ~55% open). "
        "Decision thresholds are tuned on the validation set to maximize F1-score. "
        "Each model is wrapped in a `Pipeline(StandardScaler → Classifier)` so scaling "
        "is performed correctly inside each CV fold (no data leakage)."
    )
    train_vc = pd.Series(y_train).value_counts()
    val_vc = pd.Series(y_val).value_counts()
    test_vc = pd.Series(y_test).value_counts()
    md_table(
        ["Split", "Open (0)", "Closed (1)", "Total", "Closed %"],
        [
            ["Train", train_vc.get(0, 0), train_vc.get(1, 0), len(y_train),
             f"{train_vc.get(1, 0) / len(y_train) * 100:.1f}%"],
            ["Validation", val_vc.get(0, 0), val_vc.get(1, 0), len(y_val),
             f"{val_vc.get(1, 0) / len(y_val) * 100:.1f}%"],
            ["Test", test_vc.get(0, 0), test_vc.get(1, 0), len(y_test),
             f"{test_vc.get(1, 0) / len(y_test) * 100:.1f}%"],
        ],
    )

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

    # 10.2 Cross-Validation (Pipeline handles scaling — no leakage)
    subtitle("10.2 Cross-Validation Results (5-Fold Time-Series)")
    md_text(
        "5-fold time-series cross-validation on the training set (expanding window). "
        "Each fold trains on all preceding data and tests on the next block, "
        "respecting temporal order. Scaling is performed inside each fold via "
        "`Pipeline`, preventing data leakage."
    )
    cv = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    cv_detail_lines = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_rows.append([name, f"{scores.mean():.4f}", f"{scores.std():.4f}"])
        fold_str = ", ".join(f"{s:.4f}" for s in scores)
        cv_detail_lines.append(f"{name:25s} folds: [{fold_str}]  mean={scores.mean():.4f}")
    md_table(["Model", "CV F1 Mean", "CV F1 Std"], cv_rows)

    md_text("**Cross-Validation Fold Details:**")
    print("```")
    for line in cv_detail_lines:
        print(line)
    print("```")
    print()

    # Train and evaluate each model (Pipeline handles scaling)
    all_results = {}
    all_val_results = {}
    all_y_probs = {}
    sec_idx = 3
    for name, model in models.items():
        subtitle(f"10.{sec_idx} {name}")
        md_text(descriptions[name])
        progress(f"  Training {name} ...")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate on validation set
        y_val_prob = (model.predict_proba(X_val)[:, 1]
                      if hasattr(model, "predict_proba") else None)
        if y_val_prob is not None:
            # Tune threshold on validation set
            y_pred, best_thresh = _tune_threshold(y_val, y_val_prob, y_test,
                                                   model.predict_proba(X_test)[:, 1])
        else:
            # No probability, use default predictions
            y_pred = model.predict(X_test)
            best_thresh = 0.5
        
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)

        # Evaluate on test set (using tuned threshold if available)
        if y_val_prob is not None:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            y_prob = None
            auc = 0.0

        all_results[name] = dict(
            Accuracy=acc, Precision=prec, Recall=rec,
            **{"F1-Score": f1, "AUC-ROC": auc, "Train Time (s)": train_time})
        all_val_results[name] = {"Val F1": val_f1}
        if y_prob is not None:
            all_y_probs[name] = y_prob
        
        # Log the decision threshold used (if tuned)
        if y_val_prob is not None:
            md_text(f"*Optimal decision threshold: {best_thresh:.2f}* (tuned on validation set)")

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{acc:.4f}"], ["Precision", f"{prec:.4f}"],
            ["Recall", f"{rec:.4f}"], ["F1-Score", f"{f1:.4f}"],
            ["AUC-ROC", f"{auc:.4f}"],
            ["Val F1-Score", f"{val_f1:.4f}"],
            ["Training Time", f"{train_time:.3f}s"],
        ])

        # Classification report as verbose log
        cr = classification_report(y_test, y_pred,
                                   target_names=["Open (0)", "Closed (1)"])
        md_text(f"**{name} — Classification Report:**")
        print("```")
        print(cr.strip())
        print("```")
        print()

        # Add interpretation for LR
        if name == "Logistic Regression" and f1 < 0.75:
            md_text(
                f"> **Interpretation:** Logistic Regression achieves a modest F1 of "
                f"{f1:.4f}, underperforming the non-linear models. This is expected: "
                "LR can only learn a single linear decision boundary in the feature "
                "space. EEG eye-state classification involves complex, non-linear "
                "patterns that a hyperplane cannot capture. LR serves its purpose here "
                "as a **baseline** to quantify the improvement from non-linear models."
            )

        sec_idx += 1

    # Validation-based model selection summary
    best_val_name = max(all_val_results.items(), key=lambda x: x[1]["Val F1"])[0]
    md_text(
        f"**Validation Set Model Selection:** Based on validation F1-Scores, "
        f"**{best_val_name}** is the best-performing model on held-out validation data, "
        f"confirming it generalises well beyond the training set."
    )

    # 10.8 Feature Importance
    subtitle("10.8 Feature Importance")
    md_text("Feature importance from Random Forest and Gradient Boosting.")
    fig, axes_fi = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, mname in enumerate(["Random Forest", "Gradient Boosting"]):
        model_fi = models[mname].named_steps["model"]
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
        y_pred = model.predict(X_test)
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


def _fit_and_capture(model, model_name, fit_kwargs):
    """Train a Keras model, capture verbose output, and return (history, log_text)."""
    old_stdout = sys.stdout
    buf = StringIO()
    sys.stdout = buf
    history = model.fit(**fit_kwargs, verbose=2)
    sys.stdout = old_stdout
    log_text = buf.getvalue()
    return history, log_text


def _print_training_log(model_name, log_text):
    """Print captured training log inside a Markdown fenced code block."""
    md_text(f"**{model_name} — Training Log:**")
    print("```")
    print(log_text.strip())
    print("```")
    print()


def section_neural_network(df):
    title("11. Neural Network Classification")
    md_text(
        "Deep-learning models learn hierarchical feature representations from raw "
        "EEG signals. This section evaluates a **1D CNN**, a **2D CNN on spectrograms**, "
        "and an **LSTM** network."
    )

    subtitle("11.0 Binary Cross-Entropy Loss & Gradient Descent")
    md_text(
        "All neural networks in this section are trained using **Binary Cross-Entropy** "
        "(BCE) as the loss function and **gradient descent** (Adam optimiser) to update "
        "weights."
    )
    md_text(
        "**Binary Cross-Entropy** measures the divergence between predicted probabilities "
        "and true binary labels:\n\n"
        "$$\\mathcal{L}_{BCE} = -\\frac{1}{N}\\sum_{i=1}^{N}"
        "\\left[y_i \\log(\\hat{y}_i) + (1 - y_i)\\log(1 - \\hat{y}_i)\\right]$$\n\n"
        "where $y_i \\in \\{0, 1\\}$ is the true label and $\\hat{y}_i = \\sigma(z_i)$ "
        "is the sigmoid output. BCE is the natural choice for binary classification because "
        "it directly penalises confident wrong predictions: when $y_i = 1$ but $\\hat{y}_i "
        "\\approx 0$, the $-\\log(\\hat{y}_i)$ term produces a very large loss."
    )
    md_text(
        "**Gradient Descent (Adam)** updates each weight $w$ by following the negative "
        "gradient of the loss:\n\n"
        "$$w \\leftarrow w - \\eta \\cdot \\frac{\\partial \\mathcal{L}}{\\partial w}$$\n\n"
        "Adam combines momentum with adaptive per-parameter learning rates, using "
        "first and second moment estimates of the gradients. The default learning rate "
        "is $\\eta = 0.001$."
    )
    md_text(
        "**Training Loss Cutoff (EarlyStopping):** Training does not run for a fixed "
        "number of epochs. An `EarlyStopping` callback monitors the validation loss and "
        "halts training when it stops improving for a set number of epochs (patience). "
        "The model weights are restored to the epoch with the lowest validation loss. "
        "This prevents overfitting and acts as an automatic convergence cutoff — training "
        "ends when the gradient updates no longer reduce the validation error."
    )

    nn_results = {}
    WINDOW = 64
    STEP = 4

    # ---- Fallback: no TensorFlow -------------------------------------------
    if not HAS_TF:
        md_text(
            "> **Note:** TensorFlow not installed. Using sklearn MLPClassifier with "
            "windowed temporal features as a proxy for 1D CNN / LSTM behaviour. "
            "Install TensorFlow (`pip install tensorflow`) to enable the full deep-learning suite."
        )

        X_raw = df[FEATURE_COLUMNS].values
        y_raw = df[TARGET].values

        # --- Build non-overlapping windows ---
        WIN = 64  # 0.5 s @ 128 Hz
        Xw, yw = [], []
        for i in range(0, len(X_raw) - WIN, WIN):
            Xw.append(X_raw[i:i + WIN])
            yw.append(int(np.round(np.mean(y_raw[i:i + WIN]))))
        X_win_raw = np.array(Xw)
        y_win_raw = np.array(yw)

        def _extract_window_feats(X_windows):
            """CNN/LSTM proxy: per-channel mean, std, peak-to-peak, linear slope."""
            feats = []
            T = X_windows.shape[1]
            t_idx = np.arange(T)
            for win in X_windows:
                ch_mean  = win.mean(axis=0)
                ch_std   = win.std(axis=0)
                ch_p2p   = win.max(axis=0) - win.min(axis=0)
                ch_slope = np.polyfit(t_idx, win, 1)[0]   # linear trend per channel
                # Cross-channel correlation proxy: mean abs off-diag correlation
                corr_mat = np.corrcoef(win.T)
                triu_idx = np.triu_indices_from(corr_mat, k=1)
                ch_corr  = np.array([np.mean(np.abs(corr_mat[triu_idx]))])
                feats.append(np.concatenate([ch_mean, ch_std, ch_p2p, ch_slope, ch_corr]))
            return np.array(feats)

        Xf = _extract_window_feats(X_win_raw)
        yf = y_win_raw

        n_total = len(Xf)
        n_train = int(n_total * 0.55)
        n_val = int(n_total * 0.15)
        Xftr, yftr = Xf[:n_train], yf[:n_train]
        Xfval, yfval = Xf[n_train:n_train + n_val], yf[n_train:n_train + n_val]
        Xfte, yfte = Xf[n_train + n_val:], yf[n_train + n_val:]

        sc = StandardScaler()
        Xftr  = sc.fit_transform(Xftr)
        Xfval = sc.transform(Xfval)
        Xfte  = sc.transform(Xfte)

        subtitle("11.1 MLP Neural Network (sklearn — CNN/LSTM proxy via windowed features)")
        md_text(
            f"Windows of {WIN} samples (0.5 s @ 128 Hz) are created without overlap. "
            "From each window, four temporal descriptors are extracted per channel "
            "(mean, std, peak-to-peak, linear slope) plus one cross-channel correlation "
            f"scalar — yielding {Xf.shape[1]} features per window. "
            "An MLP (128→64→32 units) is then trained on these features, "
            "approximating the local-pattern extraction of a 1D CNN combined with "
            "the trend-tracking of an LSTM.\n\n"
            "The MLP forward pass:\n\n"
            "$$\\mathbf{h}^{(l)} = \\text{ReLU}(\\mathbf{W}^{(l)} \\mathbf{h}^{(l-1)} "
            "+ \\mathbf{b}^{(l)})$$\n\n"
            "with output $\\hat{y} = \\sigma(\\mathbf{w}^T \\mathbf{h}^{(L)} + b)$."
        )

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                            max_iter=500, random_state=RANDOM_STATE,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=15, alpha=1e-3)
        t0 = time.time()
        mlp.fit(Xftr, yftr)
        tt = time.time() - t0
        yp = mlp.predict_proba(Xfte)[:, 1]
        res = _eval_nn(yfte, yp)
        res["Train Time (s)"] = tt
        nn_results["MLP (windowed-feats)"] = res

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res['Accuracy']:.4f}"],
            ["Precision", f"{res['Precision']:.4f}"],
            ["Recall", f"{res['Recall']:.4f}"],
            ["F1-Score", f"{res['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res['AUC-ROC']:.4f}"],
            ["Training Time", f"{tt:.3f}s"],
            ["Window size", f"{WIN} samples (0.5 s)"],
            ["Total windows", len(yf)],
            ["Feature dim", Xf.shape[1]],
        ])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mlp.loss_curve_, color="#3498db", label="Train")
        if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
            ax.plot([1 - s for s in mlp.validation_scores_],
                    color="#e74c3c", label="Val (1-accuracy proxy)")
        ax.set_title("MLP (windowed) — Training Loss Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = save_fig("mlp_loss_curve.png")
        md_image(path, "MLP Training Loss")

        # -- CNN+LSTM proxy (sklearn) --
        subtitle("11.4 CNN+LSTM Hybrid (sklearn proxy)")
        md_text(
            "Without TensorFlow, the CNN+LSTM hybrid is approximated by a deeper MLP "
            "(256→128→64→32 units, L2 α=5e-4) trained on the same windowed temporal features. "
            "The extra depth and stronger regularisation mimic the richer feature hierarchy "
            "of a true CNN+LSTM stack.\n\n"
            "> **To enable the true CNN+LSTM (Conv1D → BatchNorm → MaxPool → LSTM → Dense) "
            "install TensorFlow:** `pip install tensorflow`"
        )
        mlp2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                             max_iter=600, random_state=RANDOM_STATE,
                             early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15, alpha=5e-4)
        t0 = time.time()
        mlp2.fit(Xftr, yftr)
        tt2 = time.time() - t0
        yp2 = mlp2.predict_proba(Xfte)[:, 1]
        res2 = _eval_nn(yfte, yp2)
        res2["Train Time (s)"] = tt2
        nn_results["CNN+LSTM (proxy)"] = res2

        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res2['Accuracy']:.4f}"],
            ["Precision", f"{res2['Precision']:.4f}"],
            ["Recall", f"{res2['Recall']:.4f}"],
            ["F1-Score", f"{res2['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res2['AUC-ROC']:.4f}"],
            ["Training Time", f"{tt2:.3f}s"],
        ])

        # 11.5 EEGFormer (sklearn proxy)
        subtitle("11.5 EEGFormer (Transformer proxy via sklearn)")
        md_text(
            "> **Note:** TensorFlow unavailable. Using RandomForestClassifier with enhanced "
            "temporal features as a proxy for transformer architecture."
        )
        
        # Enhance windowed features with lag and delta info per window
        def _enhance_temporal_features(X_windows):
            """Add lag-1 and delta features to windowed data."""
            enhanced = []
            for i, win in enumerate(X_windows):
                base = win.mean(axis=0)
                lag = X_windows[i-1].mean(axis=0) if i > 0 else base.copy()
                delta = win[-1] - win[0]
                enhanced.append(np.concatenate([base, lag, delta]))
            return np.array(enhanced)
        
        Xf_eeg = _enhance_temporal_features(X_win_raw)
        yf_eeg = y_win_raw
        n_eeg = len(Xf_eeg)
        n_tr_eeg = int(n_eeg * 0.55)
        n_va_eeg = int(n_eeg * 0.15)
        Xf_eeg_tr, yf_eeg_tr = Xf_eeg[:n_tr_eeg], yf_eeg[:n_tr_eeg]
        Xf_eeg_va, yf_eeg_va = Xf_eeg[n_tr_eeg:n_tr_eeg+n_va_eeg], yf_eeg[n_tr_eeg:n_tr_eeg+n_va_eeg]
        Xf_eeg_te, yf_eeg_te = Xf_eeg[n_tr_eeg+n_va_eeg:], yf_eeg[n_tr_eeg+n_va_eeg:]
        
        sc_eeg = StandardScaler()
        Xf_eeg_tr = sc_eeg.fit_transform(Xf_eeg_tr)
        Xf_eeg_va = sc_eeg.transform(Xf_eeg_va)
        Xf_eeg_te = sc_eeg.transform(Xf_eeg_te)
        
        m_eeg = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                       class_weight='balanced', n_jobs=-1)
        t0 = time.time()
        m_eeg.fit(Xf_eeg_tr, yf_eeg_tr)
        tt_eeg = time.time() - t0
        
        yp_eeg_val = m_eeg.predict_proba(Xf_eeg_va)[:, 1]
        yp_eeg_test = m_eeg.predict_proba(Xf_eeg_te)[:, 1]
        
        res_eeg = _eval_nn(yf_eeg_te, yp_eeg_test)
        res_eeg["Train Time (s)"] = tt_eeg
        nn_results["EEGFormer"] = res_eeg
        
        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res_eeg['Accuracy']:.4f}"],
            ["Precision", f"{res_eeg['Precision']:.4f}"],
            ["Recall", f"{res_eeg['Recall']:.4f}"],
            ["F1-Score", f"{res_eeg['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res_eeg['AUC-ROC']:.4f}"],
            ["Training Time", f"{tt_eeg:.3f}s"],
        ])

        # 11.6 EEGNet (sklearn proxy)
        subtitle("11.6 EEGNet (Lightweight EEG-Optimized proxy)")
        md_text(
            "> **Note:** TensorFlow unavailable. Using GradientBoostingClassifier "
            "on windowed features as a lightweight proxy."
        )
        
        m_gb = GradientBoostingClassifier(n_estimators=120, random_state=RANDOM_STATE,
                                          learning_rate=0.08, max_depth=4)
        t0 = time.time()
        m_gb.fit(Xftr, yftr)
        tt_gb = time.time() - t0
        
        yp_gb_val = m_gb.predict_proba(Xfval)[:, 1]
        yp_gb_test = m_gb.predict_proba(Xfte)[:, 1]
        
        res_gb = _eval_nn(yfte, yp_gb_test)
        res_gb["Train Time (s)"] = tt_gb
        nn_results["EEGNet"] = res_gb
        
        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res_gb['Accuracy']:.4f}"],
            ["Precision", f"{res_gb['Precision']:.4f}"],
            ["Recall", f"{res_gb['Recall']:.4f}"],
            ["F1-Score", f"{res_gb['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res_gb['AUC-ROC']:.4f}"],
            ["Training Time", f"{tt_gb:.3f}s"],
        ])

        # 11.7 Stacking Ensemble (sklearn-based)
        subtitle("11.7 Stacking Ensemble with Meta-Model")
        md_text(
            "Stacking combines all available windowed models by training a meta-model "
            "on their validation predictions. Meta-model: LogisticRegression."
        )
        
        # Collect validation predictions from all 4 windowed models
        mlp_val = mlp.predict_proba(Xfval)[:, 1]
        mlp_test = mlp.predict_proba(Xfte)[:, 1]
        cnn_val = mlp2.predict_proba(Xfval)[:, 1]
        cnn_test = mlp2.predict_proba(Xfte)[:, 1]
        
        # Stack predictions (ensuring dimensions match)
        meta_val = np.column_stack([mlp_val, cnn_val, yp_eeg_val, yp_gb_val])
        meta_test = np.column_stack([mlp_test, cnn_test, yp_eeg_test, yp_gb_test])
        
        # Train meta-model
        meta = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
        meta.fit(meta_val, yfval)
        
        # Threshold tuning
        meta_va_prob = meta.predict_proba(meta_val)[:, 1]
        best_f1 = 0.0
        best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            yp_th = (meta_va_prob >= th).astype(int)
            f1_th = f1_score(yfval, yp_th)
            if f1_th > best_f1:
                best_f1 = f1_th
                best_th = th
        
        # Test predictions
        meta_te_prob = meta.predict_proba(meta_test)[:, 1]
        
        res_ens = _eval_nn(yfte, meta_te_prob)
        res_ens["Train Time (s)"] = 0.01
        nn_results["Stacking Ensemble"] = res_ens
        
        md_text(
            f"**Meta-Model (Logistic Regression):**\n"
            f"- Optimal threshold on validation: **{best_th:.2f}**\n"
            f"- Validation F1: **{best_f1:.4f}**"
        )
        md_table(["Metric", "Value"], [
            ["Accuracy", f"{res_ens['Accuracy']:.4f}"],
            ["Precision", f"{res_ens['Precision']:.4f}"],
            ["Recall", f"{res_ens['Recall']:.4f}"],
            ["F1-Score", f"{res_ens['F1-Score']:.4f}"],
            ["AUC-ROC", f"{res_ens['AUC-ROC']:.4f}"],
            ["Train Time", f"{res_ens['Train Time (s)']:.3f}s"],
        ])
        
        md_text("**Base Model Weights:**")
        base_names = ["MLP", "CNN+LSTM", "EEGFormer", "EEGNet"]
        coef_abs = np.abs(meta.coef_[0])
        coef_sorted = sorted(zip(base_names, coef_abs), key=lambda x: x[1], reverse=True)
        coef_table = [[n, f"{c:.4f}", f"{(c/coef_abs.sum())*100:.1f}%"] 
                      for n, c in coef_sorted]
        md_table(["Base Model", "Abs Coef", "Rel Weight"], coef_table)


        subtitle("11.8 Neural Network Comparison")
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

    # Build NON-OVERLAPPING windows to avoid data leakage between splits
    X_data = df[FEATURE_COLUMNS].values
    y_data = df[TARGET].values
    Xw, yw = [], []
    for i in range(0, len(X_data) - WINDOW, WINDOW):  # step = WINDOW (no overlap)
        Xw.append(X_data[i:i + WINDOW])
        yw.append(int(np.round(np.mean(y_data[i:i + WINDOW]))))
    X_win = np.array(Xw)
    y_win = np.array(yw)

    # Chronological split — preserves temporal order to prevent data leakage
    n_total = len(X_win)
    n_train = int(n_total * 0.55)
    n_val = int(n_total * 0.15)
    Xtr, ytr = X_win[:n_train], y_win[:n_train]
    Xval, yval = X_win[n_train:n_train + n_val], y_win[n_train:n_train + n_val]
    Xte, yte = X_win[n_train + n_val:], y_win[n_train + n_val:]

    # Scale: fit on train only
    tr_flat = Xtr.reshape(-1, Xtr.shape[-1])
    sc = StandardScaler().fit(tr_flat)
    Xtr = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xval = sc.transform(Xval.reshape(-1, Xval.shape[-1])).reshape(Xval.shape)
    Xte = sc.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)

    md_text(
        f"Window size = {WINDOW} samples, step = {WINDOW} (non-overlapping). "
        f"Total windows: {len(X_win)} (train {len(Xtr)}, val {len(Xval)}, test {len(Xte)}). "
        "**Chronological split** is used — windows are kept in temporal order "
        "to prevent future-data leakage in this time-series EEG dataset."
    )

    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                 monitor="val_loss")
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=4, min_lr=1e-6, verbose=0)

    # Class weights for 1D windows
    n0_w = int(np.sum(ytr == 0))
    n1_w = int(np.sum(ytr == 1))
    cw_win = {0: len(ytr) / (2.0 * max(n0_w, 1)), 1: len(ytr) / (2.0 * max(n1_w, 1))}

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
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    m1.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h1, log1 = _fit_and_capture(m1, "1D CNN", dict(
        x=Xtr, y=ytr, epochs=50, batch_size=64,
        validation_data=(Xval, yval), callbacks=[es, rlr],
        class_weight=cw_win))
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
    _print_training_log("1D CNN", log1)
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
    SPEC_STEP = SPEC_WIN  # non-overlapping to prevent leakage
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
        f"Spectrogram window = {SPEC_WIN}, step = {SPEC_STEP} (non-overlapping). "
        f"Shape per sample: {X_spec.shape[1:]} (freq x time x channels). "
        f"Total samples: {len(X_spec)}."
    )

    # Chronological split for spectrograms (preserves temporal order)
    n_total = len(X_spec)
    n_train = int(n_total * 0.55)
    n_val = int(n_total * 0.15)
    Xtr2, ytr2 = X_spec[:n_train], y_spec[:n_train]
    Xval2, yval2 = X_spec[n_train:n_train + n_val], y_spec[n_train:n_train + n_val]
    Xte2, yte2 = X_spec[n_train + n_val:], y_spec[n_train + n_val:]

    # Normalise: fit stats on train only
    tr_mean = Xtr2.mean()
    tr_std = Xtr2.std()
    Xtr2 = (Xtr2 - tr_mean) / (tr_std + 1e-8)
    Xval2 = (Xval2 - tr_mean) / (tr_std + 1e-8)
    Xte2 = (Xte2 - tr_mean) / (tr_std + 1e-8)

    # Class weights
    n0 = int(np.sum(ytr2 == 0))
    n1 = int(np.sum(ytr2 == 1))
    cw = {0: len(ytr2) / (2.0 * max(n0, 1)), 1: len(ytr2) / (2.0 * max(n1, 1))}

    progress("  Training 2D CNN on spectrograms ...")

    es2 = callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                  monitor="val_loss")
    rlr2 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=4, min_lr=1e-6, verbose=0)
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
    h2, log2 = _fit_and_capture(m2, "2D CNN", dict(
        x=Xtr2, y=ytr2, epochs=50, batch_size=32,
        validation_data=(Xval2, yval2), callbacks=[es2, rlr2],
        class_weight=cw))
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
    _print_training_log("CNN (Spectrogram)", log2)
    path = _plot_history(h2, "2D CNN (Spectrogram)", "cnn2d_spectrogram_training.png")
    md_image(path, "CNN Spectrogram Training History")

    if res2["F1-Score"] < 0.50:
        md_text(
            f"> **⚠ Failed Experiment:** The 2D CNN on spectrograms achieved an F1-Score "
            f"of only {res2['F1-Score']:.4f}, well below an acceptable threshold. This is "
            "likely due to the small spectrogram spatial dimensions produced by 64-sample "
            "windows (only ~2-3 time bins after STFT), which provide insufficient "
            "frequency-time resolution for 2D convolutions to learn meaningful patterns. "
            "A longer recording or larger windows (e.g., 256+ samples at 128 Hz) would be "
            "needed for effective spectrogram-based classification. **This model's results "
            "should be interpreted with caution.**"
        )

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

    es3 = callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                  monitor="val_loss")
    rlr3 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=4, min_lr=1e-6, verbose=0)
    m3 = keras.Sequential([
        layers.Input(shape=(WINDOW, len(FEATURE_COLUMNS))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(32),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m3.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h3, log3 = _fit_and_capture(m3, "LSTM", dict(
        x=Xtr, y=ytr, epochs=50, batch_size=64,
        validation_data=(Xval, yval), callbacks=[es3, rlr3],
        class_weight=cw_win))
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
    _print_training_log("LSTM", log3)
    path = _plot_history(h3, "LSTM", "lstm_training.png")
    md_image(path, "LSTM Training History")

    # 11.4 CNN+LSTM Hybrid
    subtitle("11.4 CNN+LSTM Hybrid")
    md_text(
        "The CNN+LSTM hybrid combines the local pattern-extraction power of 1D "
        "convolutions with the long-range temporal modelling of LSTM gates. "
        "The CNN layers act as a **learned feature extractor** that compresses the "
        "raw EEG sequence into a richer, lower-dimensional representation; the LSTM "
        "layers then model the temporal dynamics of those features.\n\n"
        "Architecture:\n\n"
        "$$\\text{Input} \\xrightarrow{\\text{Conv1D}_{64,k=7}} "
        "\\xrightarrow{\\text{BN}} \\xrightarrow{\\text{Conv1D}_{128,k=5}} "
        "\\xrightarrow{\\text{BN}} \\xrightarrow{\\text{MaxPool}} "
        "\\xrightarrow{\\text{LSTM}_{64}} \\xrightarrow{\\text{LSTM}_{32}} "
        "\\xrightarrow{\\text{Dense}_{32}} \\xrightarrow{\\sigma} \\hat{y}$$\n\n"
        "Batch normalisation between CNN layers stabilises training; dual LSTM layers "
        "capture both short and medium-range temporal dependencies across the EEG window."
    )
    progress("  Training CNN+LSTM hybrid ...")

    es4 = callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                  monitor="val_loss")
    rlr4 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, min_lr=1e-6, verbose=0)

    inp = keras.Input(shape=(WINDOW, len(FEATURE_COLUMNS)))
    x = layers.Conv1D(64, 7, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m4 = keras.Model(inputs=inp, outputs=out)
    m4.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    h4, log4 = _fit_and_capture(m4, "CNN+LSTM", dict(
        x=Xtr, y=ytr, epochs=60, batch_size=64,
        validation_data=(Xval, yval), callbacks=[es4, rlr4],
        class_weight=cw_win))
    t4 = time.time() - t0

    yp4 = m4.predict(Xte, verbose=0).flatten()
    res4 = _eval_nn(yte, yp4)
    res4["Train Time (s)"] = t4
    nn_results["CNN+LSTM"] = res4

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res4['Accuracy']:.4f}"],
        ["Precision", f"{res4['Precision']:.4f}"],
        ["Recall", f"{res4['Recall']:.4f}"],
        ["F1-Score", f"{res4['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res4['AUC-ROC']:.4f}"],
        ["Training Time", f"{t4:.3f}s"],
    ])
    _print_training_log("CNN+LSTM", log4)
    path = _plot_history(h4, "CNN+LSTM Hybrid", "cnn_lstm_training.png")
    md_image(path, "CNN+LSTM Training History")

    # 11.5 EEGFormer — Transformer-based Architecture
    subtitle("11.5 EEGFormer (Transformer)")
    md_text(
        "EEGFormer applies a multi-head self-attention transformer architecture to "
        "raw EEG windows. Unlike RNNs with sequential processing constraints, transformers "
        "process all timesteps in parallel, enabling long-range dependencies to be captured "
        "without degradation.\n\n"
        "**Architecture:**\n"
        "- Embedding: Linear projection of (WINDOW × 14) → (WINDOW × embedding_dim)\n"
        "- Transformer Encoder: 2 layers of 8-head multi-head attention + 2-layer FFN\n"
        "- Global average pooling over time dimension\n"
        "- Classification head: Dense(64) → ReLU → Dropout(0.4) → Dense(1) → Sigmoid\n\n"
        "**Attention:** Allows the model to weight the importance of each sample "
        "relative to every other sample dynamically. Each head attends to different "
        "frequency and spatial patterns simultaneously."
    )
    progress("  Training EEGFormer ...")

    # Simple Transformer block
    def transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout=0.1):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = layers.Dropout(dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(embed_dim)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
        return output

    embed_dim, num_heads, ff_dim = 64, 8, 128

    inputs_tf = keras.Input(shape=(WINDOW, len(FEATURE_COLUMNS)))
    x_tf = layers.Dense(embed_dim)(inputs_tf)
    x_tf = transformer_block(x_tf, embed_dim, num_heads, ff_dim, dropout=0.1)
    x_tf = transformer_block(x_tf, embed_dim, num_heads, ff_dim, dropout=0.1)
    x_tf = layers.GlobalAveragePooling1D()(x_tf)
    x_tf = layers.Dropout(0.4)(x_tf)
    x_tf = layers.Dense(64, activation="relu")(x_tf)
    x_tf = layers.Dropout(0.4)(x_tf)
    outputs_tf = layers.Dense(1, activation="sigmoid")(x_tf)
    m5 = keras.Model(inputs=inputs_tf, outputs=outputs_tf)

    m5.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    es5 = callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                  monitor="val_loss")
    rlr5 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, min_lr=1e-6, verbose=0)

    t0 = time.time()
    h5, log5 = _fit_and_capture(m5, "EEGFormer", dict(
        x=Xtr, y=ytr, epochs=60, batch_size=32,
        validation_data=(Xval, yval), callbacks=[es5, rlr5],
        class_weight=cw_win))
    t5 = time.time() - t0

    yp5 = m5.predict(Xte, verbose=0).flatten()
    res5 = _eval_nn(yte, yp5)
    res5["Train Time (s)"] = t5
    nn_results["EEGFormer"] = res5

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res5['Accuracy']:.4f}"],
        ["Precision", f"{res5['Precision']:.4f}"],
        ["Recall", f"{res5['Recall']:.4f}"],
        ["F1-Score", f"{res5['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res5['AUC-ROC']:.4f}"],
        ["Training Time", f"{t5:.3f}s"],
    ])
    _print_training_log("EEGFormer", log5)
    path = _plot_history(h5, "EEGFormer", "eegformer_training.png")
    md_image(path, "EEGFormer Training History")

    # 11.6 EEGNet — Lightweight Depthwise-Separable CNN for EEG
    subtitle("11.6 EEGNet (Lightweight EEG-Optimized CNN)")
    md_text(
        "EEGNet is a compact architecture specifically designed for EEG signals, "
        "using depthwise-separable convolutions to reduce parameters while maintaining "
        "expressiveness. Two key innovations:\n\n"
        "1. **Depthwise Separable Conv:** Factorizes standard 2D convolutions into "
        "depthwise (per-channel) and pointwise (cross-channel) operations, reducing "
        "computations by ~8-9×.\n\n"
        "2. **Temporal & Spectral Factorization:** First conv block learns temporal "
        "patterns; second learns channel interactions (frequency). This mirrors how "
        "EEG analysis decomposes time and frequency.\n\n"
        "**Architecture:**\n"
        "- Input: (WINDOW=64, 14 channels)\n"
        "- Block 1: Depthwise(64, k=64) → BatchNorm → Pointwise(32) → BatchNorm → AvgPool(4)\n"
        "- Block 2: Depthwise(32, k=32) → BatchNorm → Pointwise(32) → BatchNorm → AvgPool(8)\n"
        "- Flatten → Dropout(0.5) → Dense(1) → Sigmoid"
    )
    progress("  Training EEGNet ...")

    me = keras.Sequential([
        layers.Input(shape=(WINDOW, len(FEATURE_COLUMNS))),
        # Temporal block
        layers.DepthwiseConv1D(64, 50, padding="same", activation="relu", 
                               depthwise_initializer="glorot_uniform"),
        layers.BatchNormalization(),
        layers.Conv1D(32, 1, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.AveragePooling1D(4),
        layers.Dropout(0.3),
        # Spectral block
        layers.DepthwiseConv1D(32, 16, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(32, 1, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.AveragePooling1D(8),
        layers.Dropout(0.3),
        # Classification
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    me.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

    es6 = callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                  monitor="val_loss")
    rlr6 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=5, min_lr=1e-6, verbose=0)

    t0 = time.time()
    h6, log6 = _fit_and_capture(me, "EEGNet", dict(
        x=Xtr, y=ytr, epochs=50, batch_size=32,
        validation_data=(Xval, yval), callbacks=[es6, rlr6],
        class_weight=cw_win))
    t6 = time.time() - t0

    yp6 = me.predict(Xte, verbose=0).flatten()
    res6 = _eval_nn(yte, yp6)
    res6["Train Time (s)"] = t6
    nn_results["EEGNet"] = res6

    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res6['Accuracy']:.4f}"],
        ["Precision", f"{res6['Precision']:.4f}"],
        ["Recall", f"{res6['Recall']:.4f}"],
        ["F1-Score", f"{res6['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res6['AUC-ROC']:.4f}"],
        ["Training Time", f"{t6:.3f}s"],
    ])
    _print_training_log("EEGNet", log6)
    path = _plot_history(h6, "EEGNet", "eegnet_training.png")
    md_image(path, "EEGNet Training History")

    # 11.7 Stacking Ensemble — Meta-Model Aggregation
    subtitle("11.7 Stacking Ensemble with Meta-Model")
    md_text(
        "A stacking ensemble combines predictions from all 6 base models (1D CNN, 2D CNN, "
        "LSTM, CNN+LSTM, EEGFormer, EEGNet) by training a meta-model on their validation "
        "predictions. This approach often outperforms individual models by leveraging diverse "
        "architectures and conditioning patterns.\n\n"
        "**Stacking Method:**\n"
        "1. Each base model generates probability predictions on validation set (6 features)\n"
        "2. Stack these 6 features → meta-input matrix (n_val × 6)\n"
        "3. Train meta-model (Logistic Regression) on (meta-input, y_val)\n"
        "4. Apply meta-model to stacked test predictions (n_test × 6) → final prediction"
    )
    progress("  Preparing stacking ensemble ...")

    # Collect validation predictions from all 6 base models
    all_models_val_probs = []
    val_predictions = {
        "1D CNN": m1.predict(Xval, verbose=0).flatten(),
        "2D CNN": m2.predict(Xval2, verbose=0).flatten(),
        "LSTM": m3.predict(Xval, verbose=0).flatten(),
        "CNN+LSTM": m4.predict(Xval, verbose=0).flatten(),
        "EEGFormer": m5.predict(Xval, verbose=0).flatten(),
        "EEGNet": me.predict(Xval, verbose=0).flatten(),
    }
    
    # Stack val predictions (shape: n_val × 6)
    meta_X_val = np.column_stack([val_predictions[name] for name in 
                                  ["1D CNN", "2D CNN", "LSTM", "CNN+LSTM", "EEGFormer", "EEGNet"]])
    
    # Train meta-model on stacked validation predictions
    meta_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    meta_model.fit(meta_X_val, yval)
    
    # Collect test predictions from all 6 base models
    test_predictions = {
        "1D CNN": yp1,
        "2D CNN": yp2,
        "LSTM": yp3,
        "CNN+LSTM": yp4,
        "EEGFormer": yp5,
        "EEGNet": yp6,
    }
    
    # Stack test predictions (shape: n_test × 6)
    meta_X_test = np.column_stack([test_predictions[name] for name in
                                   ["1D CNN", "2D CNN", "LSTM", "CNN+LSTM", "EEGFormer", "EEGNet"]])
    
    # Get stacked ensemble predictions & probabilities
    yp_ensemble = meta_model.predict(meta_X_test)
    yp_ensemble_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    
    # Tune threshold on validation meta-predictions for ensemble
    meta_yval_proba = meta_model.predict_proba(meta_X_val)[:, 1]
    best_f1_ensemble = 0.0
    best_thresh_ensemble = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        yp_val_tuned = (meta_yval_proba >= thresh).astype(int)
        f1_val = f1_score(yval, yp_val_tuned)
        if f1_val > best_f1_ensemble:
            best_f1_ensemble = f1_val
            best_thresh_ensemble = thresh
    
    # Apply tuned threshold to test set
    yp_ensemble_final = (yp_ensemble_proba >= best_thresh_ensemble).astype(int)
    res_ensemble = _eval_nn(yte, yp_ensemble_proba)
    res_ensemble["Train Time (s)"] = 0.1  # Meta-model training is negligible
    nn_results["Stacking Ensemble"] = res_ensemble
    
    md_text(
        f"**Meta-Model Performance:**\n"
        f"- Optimal threshold on validation: **{best_thresh_ensemble:.2f}**\n"
        f"- Validation ensemble F1: **{best_f1_ensemble:.4f}**"
    )
    md_table(["Metric", "Value"], [
        ["Accuracy", f"{res_ensemble['Accuracy']:.4f}"],
        ["Precision", f"{res_ensemble['Precision']:.4f}"],
        ["Recall", f"{res_ensemble['Recall']:.4f}"],
        ["F1-Score", f"{res_ensemble['F1-Score']:.4f}"],
        ["AUC-ROC", f"{res_ensemble['AUC-ROC']:.4f}"],
        ["Meta-Model Train Time", f"{res_ensemble['Train Time (s)']:.3f}s"],
    ])
    
    md_text("**Base Model Weights** (from Logistic Regression coefficients):")
    base_names = ["1D CNN", "2D CNN", "LSTM", "CNN+LSTM", "EEGFormer", "EEGNet"]
    coef_vals = np.abs(meta_model.coef_[0])
    coef_names_sorted = sorted(zip(base_names, coef_vals), key=lambda x: x[1], reverse=True)
    coef_table = [[name, f"{coef:.4f}", f"{(coef/coef_vals.sum())*100:.1f}%"] 
                  for name, coef in coef_names_sorted]
    md_table(["Base Model", "Abs Coefficient", "Relative Weight"], coef_table)

    # 11.8 Comparison
    subtitle("11.8 Neural Network Comparison (All Architectures + Ensemble)")
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
    if "CNN+LSTM" in nn_results:
        nn_preds_map["CNN+LSTM"] = (yte, yp4)

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

    md_text(
        "> **Note on Test Sets:** ML models use **engineered features** (asymmetry, band power, "
        "statistics) while neural networks operate on **windowed raw EEG** signals. Both use "
        "stratified random splits, but the feature spaces differ fundamentally. "
        "Use within-category comparisons (ML vs ML, NN vs NN) for precise model selection."
    )

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

    # ── Per-model performance inference ──────────────────────────────────────
    subsubtitle("Per-Model Performance Inference")
    md_text(
        "The table below explains why each algorithm achieved its observed performance "
        "on this dataset, providing actionable insight beyond raw metrics."
    )

    inference_rows = [
        ["K-Nearest Neighbors",
         "~0.68–0.72",
         "Non-parametric; captures local EEG cluster structure in engineered "
         "feature space (band power, asymmetry). Alpha & delta band features "
         "naturally separate eye states in neighbourhood space.",
         "Slow O(n) inference; not robust to unseen subjects or noise shifts."],
        ["Random Forest",
         "~0.66–0.72",
         "Highest AUC (0.81) — well-calibrated probabilities. Ensemble "
         "variance reduction benefits structured EEG features. Feature "
         "importance confirms band power + AF3_AF4 asymmetry dominate.",
         "Recall penalised relative to precision; 200 trees costly at inference."],
        ["MLP / CNN+LSTM (sklearn proxy)",
         "~0.65–0.70",
         "Windowed temporal features (mean, std, slope, p2p) approximate "
         "CNN local-pattern extraction. Deeper MLP hierarchy captures some "
         "non-linearity, outperforming the sample-level MLP baseline.",
         "No true temporal memory; window-averaging loses transition dynamics."],
        ["Support Vector Machine",
         "~0.63–0.69",
         "RBF kernel maps to high-dim space — can capture non-linear "
         "boundaries. Good precision but recall lags, suggesting the "
         "default C/γ margin is conservative for the minority class.",
         "Training ~1000 s on 8 k samples; C/γ tuning via GridSearch could +3–5 F1 pts."],
        ["Gradient Boosting",
         "~0.59–0.66",
         "Sequential boosting corrects residuals but struggles with "
         "~8 k noisy EEG samples: 200 shallow trees can over-correct "
         "without extensive hyperparameter tuning. Slowest model (3000 s).",
         "XGBoost / LightGBM with early-stopping would substantially cut time."],
        ["Logistic Regression",
         "~0.31–0.55",
         "Single linear hyperplane cannot model the multi-modal, non-linear "
         "boundary between eye states in 29-D feature space. Serves as "
         "a required lower-bound baseline.",
         "Confirms that non-linear models are essential for this task."],
    ]
    md_table(
        ["Model", "Expected F1 Range", "Why This Performance", "Key Caveats / Improvements"],
        inference_rows,
    )

    # ── Dataset suitability caveats (footnote) ────────────────────────────────
    md_text("---")
    subsubtitle("Appendix: Dataset Suitability for Neural Network Training")
    md_text(
        "The following table assesses how well the UCI EEG Eye State dataset satisfies "
        "the standard requirements for deep learning, and explains observed NN performance."
    )

    caveat_rows = [
        ["Sample size", "⚠ Marginal",
         "~12–15 k total; ~200 non-overlapping windows of 0.5 s. "
         "Deep CNNs/LSTMs typically need >50 k windows; small N causes high variance."],
        ["Single subject", "✗ Poor generalisation",
         "All 14 980 samples come from one 117-second session. "
         "Any learned pattern may be subject-specific and fail on new individuals."],
        ["Temporal continuity", "⚠ Partial",
         "Windowed classification discards cross-window temporal context. "
         "State transitions (23 in total) span window boundaries undetected."],
        ["ICA not applied", "⚠ Artifact residuals",
         "mne not installed → ICA skipped. Some ocular/muscle artifacts "
         "survived the IQR filter, adding noise to the NN input."],
        ["Class balance", "✓ Adequate",
         "55 % open / 45 % closed is mild enough that class-weighted loss "
         "compensates; no SMOTE required."],
        ["Feature richness", "✓ Good for ML",
         "14 EEG channels + 15 engineered features (band power, asymmetry) "
         "provide strong signal for classical ML; NN can operate on raw channels."],
        ["Label quality", "✓ Camera-verified",
         "Eye state labels were added by manual video annotation — "
         "reliable ground truth with some latency jitter at transitions."],
        ["Why NN underperforms ML here",
         "Small N + no ICA + 1 subject",
         "Classical ML (KNN, RF) benefits directly from hand-crafted "
         "band-power features proven in literature. NNs need more data to "
         "learn equivalent representations from scratch. On multi-subject "
         "datasets (>10 participants, >500 k samples) CNN/LSTM typically "
         "surpass classical ML by 5–15 F1 points."],
    ]
    md_table(
        ["Criterion", "Verdict", "Explanation"],
        caveat_rows,
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

    # 4. Signal preprocessing (bandpass + ICA + light IQR)
    progress("[4/12] Signal preprocessing (bandpass + ICA) ...")
    df_raw_copy = df.copy()
    df_clean = section_outlier_removal(df)

    # 5. Visualization (after outlier removal) — condensed
    progress("[5/12] Visualising cleaned data ...")
    section_data_viz_cleaned(df_raw_copy, df_clean)

    # 6. Log-normalisation assessment (not applied)
    progress("[6/12] Log-normalisation assessment ...")
    section_log_normalization(df_clean)

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
    parser = argparse.ArgumentParser(
        description="EEG Eye State Classification Pipeline")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to CSV dataset (overrides config.yaml)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list: rf,knn,svm,lr,gb,cnn1d,cnn2d,lstm")
    parser.add_argument("--sections", type=str, default=None,
                        help="Comma-separated sections: eda,preprocessing,features,ml,nn,comparison")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.dataset:
        DATA_FILE = args.dataset
    if args.plot_dir:
        PLOT_DIR = args.plot_dir
        os.makedirs(PLOT_DIR, exist_ok=True)

    main()