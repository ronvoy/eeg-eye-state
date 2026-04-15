# EEG Eye State Classification — Complete Analysis Pipeline
# pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost torch
# Run: python script.py > report.md

import sys, os, time, warnings, argparse, math
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram as scipy_spectrogram, butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not found — XGBoost model will be skipped.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Configuration
# =============================================================================

PLOT_DIR      = "analysis-plots"
DATA_FILE     = "dataset/eeg_data_og.csv"
SAMPLING_RATE = 128
FEATURE_COLUMNS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
TARGET       = "eyeDetection"
RANDOM_STATE = 42

EYE_MAP      = {0: "Open", 1: "Closed"}

# DL config
SEQ_LEN      = 64
DL_EPOCHS    = 25
DL_BATCH     = 128
DL_LR        = 1e-3
ENS_TRIALS   = 3_000
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FREQ_BANDS = {
    "Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12),
    "Beta": (12, 30), "Gamma": (30, 45),
}
BAND_COLORS = ["#8B0000", "#FF4500", "#FFD700", "#00CED1", "#9370DB"]

ELECTRODE_INFO = [
    ("AF3", "Anterior Frontal Left",  "Prefrontal Cortex",        "Executive function, attention"),
    ("F7",  "Frontal Left Lateral",   "Left Temporal-Frontal",    "Language processing"),
    ("F3",  "Frontal Left",           "Left Frontal Lobe",        "Motor planning, positive affect"),
    ("FC5", "Fronto-Central Left",    "Left Motor-Frontal",       "Motor preparation"),
    ("T7",  "Temporal Left",          "Left Temporal Lobe",       "Auditory processing, memory"),
    ("P7",  "Parietal Left",          "Left Parietal-Temporal",   "Visual-spatial processing"),
    ("O1",  "Occipital Left",         "Left Visual Cortex",       "Visual processing"),
    ("O2",  "Occipital Right",        "Right Visual Cortex",      "Visual processing"),
    ("P8",  "Parietal Right",         "Right Parietal-Temporal",  "Spatial attention"),
    ("T8",  "Temporal Right",         "Right Temporal Lobe",      "Face / emotion recognition"),
    ("FC6", "Fronto-Central Right",   "Right Motor-Frontal",      "Motor preparation"),
    ("F4",  "Frontal Right",          "Right Frontal Lobe",       "Motor planning, negative affect"),
    ("F8",  "Frontal Right Lateral",  "Right Temporal-Frontal",   "Emotion, social cognition"),
    ("AF4", "Anterior Frontal Right", "Prefrontal Cortex",        "Executive function, attention"),
]

CONFIG = {}
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
if HAS_YAML and os.path.exists(_config_path):
    with open(_config_path) as _cf:
        CONFIG = yaml.safe_load(_cf) or {}
    PLOT_DIR      = CONFIG.get("paths", {}).get("plot_dir", PLOT_DIR)
    DATA_FILE     = CONFIG.get("paths", {}).get("data_file", DATA_FILE)
    SAMPLING_RATE = CONFIG.get("data", {}).get("sampling_rate", SAMPLING_RATE)
    RANDOM_STATE  = CONFIG.get("data", {}).get("random_state", RANDOM_STATE)

os.makedirs(PLOT_DIR, exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Markdown helpers
# =============================================================================

def title(text):         print(f"\n# {text}\n")
def subtitle(text):      print(f"\n## {text}\n")
def subsubtitle(text):   print(f"\n### {text}\n")

def md_table(headers, rows):
    print("| " + " | ".join(str(h) for h in headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print()

def md_text(text):
    print(text)
    print()

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
   - 1.3 [Variable Classification & Electrode Positions](#13-variable-classification--electrode-positions)
   - 1.4 [Basic Statistics](#14-basic-statistics)
   - 1.5 [Class Distribution](#15-class-distribution)
2. [Data Imputation](#2-data-imputation)
3. [Data Visualization (Raw Data)](#3-data-visualization-raw-data)
   - 3.1 [Class Balance](#31-class-balance)
   - 3.2 [Correlation Heatmap](#32-correlation-heatmap)
   - 3.3 [Box Plots](#33-box-plots)
   - 3.4 [Histograms](#34-histograms)
4. [Signal Preprocessing (IQR → Bandpass)](#4-signal-preprocessing)
   - 4.1 [IQR Spike Removal (first)](#41-iqr-spike-removal-applied-first-before-filtering)
   - 4.2 [Bandpass Filter 0.5–45 Hz (second)](#42-bandpass-filter-0545-hz--applied-after-spike-removal)
5. [Data Visualization (After Preprocessing)](#5-data-visualization-after-preprocessing)
   - 5.1 [Corrected Correlation Heatmap](#51-corrected-correlation-heatmap-after-preprocessing)
   - 5.2 [Box Plots Comparison](#52-box-plots-comparison)
   - 5.3 [Histograms After Cleaning](#53-histograms-after-cleaning)
6. [PSD and Spectrogram Analysis](#6-psd-and-spectrogram-analysis)
   - 6.1 [Power Spectral Density (PSD)](#61-power-spectral-density-psd)
   - 6.2 [Spectrogram Analysis](#62-spectrogram-analysis)
7. [Dimensionality Reduction (LDA)](#7-dimensionality-reduction-lda)
8. [Machine Learning Classification](#8-machine-learning-classification)
   - 8.1 [Temporal Concept Drift Diagnosis](#81-temporal-concept-drift-diagnosis)
   - 8.2 [Split Configurations](#82-split-configurations)
   - 8.3 [Cross-Validation Results](#83-cross-validation-results)
   - 8.4 [Hold-Out Split Results](#84-hold-out-split-results)
   - 8.5 [Walk-Forward CV](#85-walk-forward-cv)
9. [Deep Learning Classification](#9-deep-learning-classification)
   - 9.0 [Architecture Overview & Training Setup](#90-architecture-overview--training-setup)
   - 9.1 [LSTM Classifier](#91-lstm-classifier)
   - 9.2 [CNN-LSTM Hybrid](#92-cnn-lstm-hybrid)
   - 9.3 [EEGNet (Lawhern 2018)](#93-eegnet-lawhern-2018)
   - 9.4 [Soft-Vote Ensemble](#94-soft-vote-ensemble)
   - 9.5 [DL Model Comparison](#95-dl-model-comparison)
10. [Final Comparison and Inference](#10-final-comparison-and-inference)
    - 10.1 [Unified Model Comparison](#101-unified-model-comparison)
    - 10.2 [Inference and Recommendation](#102-inference-and-recommendation)"""
    md_text(toc)
    md_text("---")

# =============================================================================
# 1. Data Description
# =============================================================================

def section_data_description(df):
    title("1. Data Description Overview")

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

    subtitle("1.3 Variable Classification & Electrode Positions")
    md_text(
        "**Numerical Variables (Continuous):** 14 EEG electrode channels recording voltage "
        "in micro-volts (µV). The Emotiv EPOC headset uses a modified 10-20 international "
        "system for electrode placement."
    )
    _elec_map = {e[0]: e for e in ELECTRODE_INFO}
    combined_rows = []
    for ch in FEATURE_COLUMNS:
        e = _elec_map.get(ch, (ch, "—", "—", "—"))
        combined_rows.append([ch, "Continuous (float64)", e[1], e[2], e[3]])
    md_table(
        ["Electrode", "Type", "10-20 Position", "Brain Region", "Functional Significance"],
        combined_rows,
    )
    md_text("**Categorical Variable (Target):**")
    md_table(
        ["Variable", "Type", "Values", "Description"],
        [[TARGET, "Binary (int)", "0 = Open, 1 = Closed", "Eye state detected via camera during recording"]],
    )

    subtitle("1.4 Basic Statistics")
    md_text("Descriptive statistics for all 14 EEG channels (µV).")
    desc = df[FEATURE_COLUMNS].describe().T
    rows = []
    for ch in FEATURE_COLUMNS:
        r = desc.loc[ch]
        rows.append([ch, int(r["count"]),
                     f"{r['mean']:.2f}", f"{r['std']:.2f}",
                     f"{r['min']:.2f}", f"{r['25%']:.2f}", f"{r['50%']:.2f}",
                     f"{r['75%']:.2f}", f"{r['max']:.2f}"])
    md_table(["Channel", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"], rows)
    md_text(
        "> **Note on Spike Artifacts:** Some channels exhibit extremely large max values — "
        "orders of magnitude above the 75th percentile. These are likely **electrode spike "
        "artifacts** caused by momentary loss of contact, muscle movement, or impedance "
        "changes in the Emotiv headset. These will be addressed by outlier removal."
    )

    subtitle("1.5 Class Distribution")
    md_text(f"Distribution of the target variable `{TARGET}` (per UCI: 0 = open, 1 = closed).")
    vc = df[TARGET].value_counts()
    md_table(
        ["Eye State", "Count", "Percentage"],
        [
            ["Open (0)",   vc.get(0, 0), f"{vc.get(0, 0) / len(df) * 100:.1f}%"],
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
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        md_text(
            f"**Result:** No missing values detected across any of the "
            f"{len(FEATURE_COLUMNS)} EEG channels. The dataset is complete."
        )
    else:
        md_text(f"**Missing values detected:** {total_missing}")
        rows = [[ch, int(df[ch].isnull().sum())] for ch in FEATURE_COLUMNS if df[ch].isnull().any()]
        md_table(["Channel", "Missing Count"], rows)
        for col in FEATURE_COLUMNS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        md_text("Missing values filled using **median imputation**.")
    return df

# =============================================================================
# 3. Data Visualization (Raw)
# =============================================================================

def section_data_viz_raw(df):
    title("3. Data Visualization (Raw Data)")
    md_text("Visualizations of the raw EEG data before any preprocessing.")

    subtitle("3.1 Class Balance")
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[TARGET].value_counts()
    bars = ax.bar(["Open (0)", "Closed (1)"], [vc.get(0, 0), vc.get(1, 0)],
                  color=["#3498db", "#e74c3c"], edgecolor="black")
    ax.set_title("Class Balance of Eye States")
    ax.set_ylabel("Count")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                str(int(b.get_height())), ha="center", fontweight="bold")
    md_image(save_fig("class_balance_raw.png"), "Class Balance")

    subtitle("3.2 Correlation Heatmap")
    md_text(
        "The correlation heatmap reveals linear relationships between EEG channels. "
        "Computed on data **winsorized at the 1st–99th percentile** to suppress "
        "spike-artifact-driven artificial correlations."
    )
    df_win = df[FEATURE_COLUMNS].clip(
        lower=df[FEATURE_COLUMNS].quantile(0.01),
        upper=df[FEATURE_COLUMNS].quantile(0.99),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_win.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap of EEG Channels (winsorized 1st–99th pct)")
    md_image(save_fig("corr_heatmap_raw.png"), "Correlation Heatmap (Raw)")

    subtitle("3.3 Box Plots")
    md_text("Box plots highlight potential outliers beyond the 1.5x IQR whiskers.")
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        ax.boxplot(df[ch].values, vert=True)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    plt.suptitle("Box Plots — All EEG Channels (Raw)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("boxplots_raw.png"), "Box Plots (Raw)")

    # Zoomed view
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        clipped = df[ch].clip(df[ch].quantile(0.01), df[ch].quantile(0.99))
        for state, color, label in [(0, "#3498db", "Open"), (1, "#e74c3c", "Closed")]:
            data = clipped[df[TARGET] == state]
            ax.boxplot(data.values, positions=[state], widths=0.6,
                       boxprops=dict(color=color), medianprops=dict(color=color),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["O", "C"], fontsize=7)
    plt.suptitle("Box Plots — Zoomed (1st–99th percentile)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("boxplots_raw_zoomed.png"), "Box Plots Zoomed (Raw)")

    subtitle("3.4 Histograms")
    md_text("Amplitude distributions per channel split by eye state.")
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        lo, hi = df[ch].quantile(0.01), df[ch].quantile(0.99)
        for state, color, label in [(0, "#3498db", "Open"), (1, "#e74c3c", "Closed")]:
            data = df.loc[df[TARGET] == state, ch].clip(lo, hi)
            ax.hist(data, bins=50, alpha=0.5, color=color, label=label)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)
    plt.suptitle("Histograms — All Channels by Eye State (Raw)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("histograms_raw.png"), "Histograms (Raw)")

# =============================================================================
# 4. Signal Preprocessing — IQR → Bandpass
# =============================================================================

def _bandpass_filter(df, lowcut=0.5, highcut=45.0, fs=None, order=4):
    if fs is None:
        fs = SAMPLING_RATE
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = df.copy()
    for col in FEATURE_COLUMNS:
        filtered[col] = filtfilt(b, a, df[col].values)
    return filtered


def _light_iqr(df, multiplier=3.0, max_passes=3):
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


def section_preprocessing(df):
    title("4. Signal Preprocessing")
    md_text(
        "EEG signals contain artifacts from eye blinks, muscle movement, and electrode "
        "drift that must be removed before analysis. This section applies a two-stage "
        "cleaning pipeline in the **correct causal order**:\n\n"
        "1. **IQR spike removal first** — raw hardware spike artifacts (up to 715,897 µV) "
        "are removed *before* filtering. Applying `filtfilt` to spikes first smears them "
        "to neighbouring samples via the backward pass, inflating data loss from ~9% to ~19%.\n\n"
        "2. **Bandpass filter (0.5–45 Hz) second** — applied to the already spike-free signal "
        "so no artifact energy is convolved into the physiological EEG bands."
    )
    original_count = len(df)
    cfg_pre  = CONFIG.get("preprocessing", {})
    iqr_cfg  = cfg_pre.get("iqr", {})
    iqr_mult = iqr_cfg.get("multiplier", 3.0)
    iqr_pass = iqr_cfg.get("max_passes", 3)
    bp_cfg   = cfg_pre.get("bandpass", {})
    lowcut   = bp_cfg.get("lowcut", 0.5)
    highcut  = bp_cfg.get("highcut", 45.0)
    bp_order = bp_cfg.get("order", 4)

    # Step 1: IQR spike removal
    subtitle("4.1 IQR Spike Removal (applied first, before filtering)")
    md_text(
        f"A **light IQR filter** ({iqr_mult}x IQR, max {iqr_pass} passes) removes "
        "hardware spike artifacts from the **raw** signal. Applying this step "
        "*before* filtering is critical: `filtfilt` convolves forward then "
        "backward, so a single spike at sample $t$ would contaminate samples "
        "$t - N$ through $t + N$ after filtering."
    )
    df_iqr, bounds, n_passes = _light_iqr(df, iqr_mult, iqr_pass)
    md_table(["Channel", "Lower Bound (µV)", "Upper Bound (µV)"], bounds)
    removed = original_count - len(df_iqr)
    md_table(
        ["Metric", "Value"],
        [
            ["Original samples", original_count],
            ["After IQR removal", len(df_iqr)],
            ["Spike samples removed", removed],
            ["Removal %", f"{removed / original_count * 100:.1f}%"],
            ["IQR passes", n_passes],
            ["IQR multiplier", f"{iqr_mult}x"],
        ],
    )

    # Step 2: Bandpass filter
    subtitle("4.2 Bandpass Filter (0.5–45 Hz) — applied after spike removal")
    md_text(
        f"A {bp_order}th-order Butterworth bandpass filter ({lowcut}–{highcut} Hz) removes "
        "DC drift and high-frequency noise while preserving physiologically relevant "
        "EEG bands (Delta through Gamma). Applied via `scipy.signal.filtfilt` "
        "(zero-phase, forward-backward filtering) to avoid phase distortion.\n\n"
        "$$|H(j\\omega)|^2 = \\frac{1}{1 + \\left(\\frac{\\omega^2 - \\omega_0^2}"
        "{\\omega_c}\\right)^{2N}}$$\n\n"
        f"where $\\omega_0 = \\sqrt{{\\omega_L \\cdot \\omega_H}}$ and $N$ = {bp_order} "
        "is the filter order."
    )
    df_clean = _bandpass_filter(df_iqr, lowcut, highcut, SAMPLING_RATE, bp_order)

    # Show bandpass effect on O1
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    samp = min(1000, len(df_iqr))
    axes[0].plot(df_iqr["O1"].values[:samp], color="#2ecc71", linewidth=0.5)
    axes[0].set_ylabel("O1 (µV)"); axes[0].legend(["After IQR (pre-filter)"], fontsize=8)
    axes[1].plot(df_clean["O1"].values[:samp], color="#e74c3c", linewidth=0.5)
    axes[1].set_ylabel("O1 (µV)"); axes[1].set_xlabel("Sample")
    axes[1].legend(["After Bandpass"], fontsize=8)
    plt.suptitle(f"Bandpass Filter Effect — O1 ({lowcut}–{highcut} Hz) [applied to spike-free signal]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("bandpass_effect.png"), "Bandpass Filter Effect")

    final_count = len(df_clean)
    md_table(
        ["Metric", "Value"],
        [
            ["Original samples", original_count],
            ["After IQR spike removal", len(df_iqr)],
            ["After bandpass filter", final_count],
            ["Total removed", original_count - final_count],
            ["Total removal %", f"{(original_count - final_count) / original_count * 100:.1f}%"],
            ["Bandpass range", f"{lowcut}–{highcut} Hz"],
            ["Filter order", bp_order],
        ],
    )
    md_blockquote(
        f"**Preprocessing Summary (corrected order):** IQR spike removal ({iqr_mult}×, "
        f"{removed / original_count * 100:.1f}% removed) → Bandpass filter ({lowcut}–{highcut} Hz). "
        f"Total retained: **{final_count} / {original_count} samples ({final_count / original_count * 100:.1f}%)**."
    )
    return df_clean

# =============================================================================
# 5. Data Visualization (After Preprocessing)
# =============================================================================

def section_data_viz_cleaned(df_raw, df_clean):
    title("5. Data Visualization (After Preprocessing)")
    md_text("Comparison of distributions before and after preprocessing.")

    subtitle("5.1 Corrected Correlation Heatmap (after preprocessing)")
    md_text(
        "With spike artifacts removed, the correlation heatmap now reflects true "
        "physiological relationships between EEG channels."
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_clean[FEATURE_COLUMNS].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap — After IQR + Bandpass Preprocessing (corrected)")
    md_image(save_fig("corr_heatmap_cleaned.png"), "Correlation Heatmap (Cleaned)")

    subtitle("5.2 Box Plots Comparison")
    md_text("Side-by-side box plots confirm preprocessing effectiveness. "
            "Whiskers are set to 3.0x IQR to match the cleaning threshold.")
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        ax.boxplot(df_clean[ch].values, vert=True, whis=3.0)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    plt.suptitle("Box Plots — After Preprocessing (whis=3.0x IQR)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("boxplots_cleaned.png"), "Box Plots (Cleaned)")

    subtitle("5.3 Histograms After Cleaning")
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        for state, color, label in [(0, "#3498db", "Open"), (1, "#e74c3c", "Closed")]:
            data = df_clean.loc[df_clean[TARGET] == state, ch]
            ax.hist(data, bins=50, alpha=0.5, color=color, label=label)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)
    plt.suptitle("Histograms — After Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("histograms_cleaned.png"), "Histograms (Cleaned)")

# =============================================================================
# 6. PSD and Spectrogram Analysis
# =============================================================================

def section_psd_spectro(df):
    title("6. PSD and Spectrogram Analysis")
    md_text(
        "Frequency-domain analysis reveals the power distribution across brain wave "
        "bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** (8-12 Hz), "
        "**Beta** (12-30 Hz), and **Gamma** (30-45 Hz). Alpha power increases when "
        "eyes are closed (the **Berger effect**)."
    )

    subtitle("6.1 Power Spectral Density (PSD)")
    md_text(
        "Welch's method estimates the PSD for each channel using segment averaging "
        "(Hann window, `nperseg=256`, 50% overlap). Shaded regions indicate standard "
        "EEG frequency bands."
    )
    df_open   = df[df[TARGET] == 0]
    df_closed = df[df[TARGET] == 1]
    nperseg   = min(256, len(df_open), len(df_closed))
    fig, axes = plt.subplots(2, 7, figsize=(28, 10))
    for idx, ch in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        fo, po = welch(df_open[ch].values,   SAMPLING_RATE, nperseg=nperseg)
        fc, pc = welch(df_closed[ch].values, SAMPLING_RATE, nperseg=nperseg)
        ax.semilogy(fo, po, label="Open",   color="blue", linewidth=1, alpha=0.8)
        ax.semilogy(fc, pc, label="Closed", color="red",  linewidth=1, alpha=0.8)
        ax.set_xlim(0, 45)
        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ylims = ax.get_ylim()
        for i, (bname, (lo, hi)) in enumerate(FREQ_BANDS.items()):
            bhi = min(hi, 45)
            ax.axvspan(lo, bhi, alpha=0.08, color=BAND_COLORS[i])
            if bhi <= 45:
                ax.text((lo + bhi) / 2, ylims[1] * 0.3, bname, fontsize=5,
                        ha="center", va="top", color=BAND_COLORS[i],
                        fontweight="bold", rotation=90)
        if idx == 0:
            ax.legend(fontsize=6)
    plt.suptitle("PSD — All Channels (Open vs Closed)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    md_image(save_fig("psd_analysis.png"), "PSD Analysis")
    md_text(
        "**PSD Interpretation — Berger Effect:** Alpha-band power (8–12 Hz) increases "
        "when the eyes are closed, particularly in occipital electrodes (O1, O2). "
        "If the red curve (closed) shows higher power in the alpha band compared to blue "
        "(open), this confirms the dataset captures genuine physiological differences "
        "between eye states."
    )

    subtitle("6.2 Spectrogram Analysis")
    md_text("Spectrograms show the time-frequency power distribution. "
            "Horizontal dashed lines mark band boundaries.")
    for state_name, state_val in [("Open", 0), ("Closed", 1)]:
        fig, axes = plt.subplots(2, 7, figsize=(28, 8))
        data_state = df[df[TARGET] == state_val]
        for idx, ch in enumerate(FEATURE_COLUMNS):
            ax = axes.flatten()[idx]
            data = data_state[ch].values
            seg  = max(4, min(128, len(data) // 4))
            f, t, Sxx = scipy_spectrogram(data, fs=SAMPLING_RATE,
                                          nperseg=seg, noverlap=seg // 2)
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                          shading="gouraud", cmap="viridis")
            ax.set_ylim(0, 45)
            ax.set_title(ch, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)
            for freq in [4, 8, 12, 30, 45]:
                ax.axhline(y=freq, color="white", linestyle="--",
                           linewidth=0.4, alpha=0.5)
        plt.suptitle(f"Spectrograms — Eyes {state_name} (All Channels)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        md_image(save_fig(f"spectrograms_{state_name.lower()}.png"),
                 f"Spectrograms Eyes {state_name}")

# =============================================================================
# 7. Dimensionality Reduction — LDA only
# =============================================================================

def section_dim_reduction(df):
    title("7. Dimensionality Reduction (LDA)")
    md_text(
        "LDA (Linear Discriminant Analysis) maximises the ratio of between-class to "
        "within-class variance, yielding the optimal single linear discriminant for "
        "binary classification. Applied to the raw 14-channel feature space."
    )

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#3498db", "Open"), (1, "#e74c3c", "Closed")]:
        ax.hist(X_lda[y == label], bins=50, alpha=0.6, color=color,
                label=name, edgecolor="black")
    ax.set_title("LDA — 1D Projection (14 EEG Channels)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("Frequency")
    ax.legend()
    md_image(save_fig("lda_1d_projection.png"), "LDA 1D Projection")

    sil = silhouette_score(np.column_stack([X_lda, np.zeros_like(X_lda)]), y)
    db  = davies_bouldin_score(np.column_stack([X_lda, np.zeros_like(X_lda)]), y)
    ch_s = calinski_harabasz_score(np.column_stack([X_lda, np.zeros_like(X_lda)]), y)
    md_table(
        ["Metric", "Value", "Interpretation"],
        [
            ["Silhouette Score", f"{sil:.4f}", "Higher = better separation (max 1.0)"],
            ["Davies-Bouldin Index", f"{db:.4f}", "Lower = better separation"],
            ["Calinski-Harabasz Score", f"{ch_s:.2f}", "Higher = better separation"],
        ]
    )
    md_text(
        f"**Interpretation:** LDA silhouette of {sil:.3f} confirms that eye states are not "
        "trivially separable in the raw amplitude space — classification requires either "
        "temporal context (DL sequence models) or frequency-domain features. This motivates "
        "the use of sequence-based DL architectures like EEGNet and CNN-LSTM."
    )

# =============================================================================
# 8. ML Pipeline — temporal splits, raw features
# =============================================================================

def temporal_three_way_split(X, y, train_frac, cv_frac):
    n  = len(X)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cv_frac))
    return X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]


def walk_forward_cv_indices(n, n_folds=5, min_train_frac=0.50):
    min_tr = int(n * min_train_frac)
    step   = (n - min_tr) // (n_folds + 1)
    return [(slice(0, min_tr + k * step),
             slice(min_tr + k * step, min_tr + (k + 1) * step))
            for k in range(n_folds)]


def _safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def _optimize_threshold(y_cv, probs_cv):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs_cv >= t).astype(int)
        score = f1_score(y_cv, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def _md_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])
    md_table(
        ["", "Pred Open", "Pred Closed"],
        [["True Open",   tn, fp],
         ["True Closed", fn, tp]],
    )
    md_text(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def _evaluate_ml(y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    bf1  = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    auc  = _safe_auc(y_true, y_prob) if y_prob is not None else float("nan")
    return {"acc": acc, "macro_f1": mf1, "binary_f1": bf1,
            "precision": prec, "recall": rec, "auc": auc}


# FIX: Use y_train (not y_all) for XGBoost scale_pos_weight
def _get_ml_models(y_train):
    models = {
        "LogisticRegression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        solver="lbfgs", random_state=RANDOM_SEED))]),
        "RandomForest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_SEED))]),
    }
    if HAS_XGB:
        # FIX: compute scale_pos_weight from training data only
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        models["XGBoost"] = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", XGBClassifier(n_estimators=200, scale_pos_weight=neg / max(pos, 1),
                                   eval_metric="logloss",
                                   random_state=RANDOM_SEED, n_jobs=-1))])
    return models


def _run_single_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    cv_prob  = model.predict_proba(X_cv)[:, 1]
    opt_t, _ = _optimize_threshold(y_cv, cv_prob)
    results  = {}
    # FIX: use optimized threshold consistently for both CV and Test
    for part_name, Xs, ys in [("CV", X_cv, y_cv), ("Test", X_te, y_te)]:
        prob   = model.predict_proba(Xs)[:, 1]
        pred   = (prob >= opt_t).astype(int)
        results[part_name] = _evaluate_ml(ys, pred, prob)
        results[part_name]["raw_prob"] = prob
        results[part_name]["pred"]     = pred
    results["train_time"] = train_time
    results["opt_t"]      = opt_t
    return results


def section_ml(X_all, y_all, N):
    title("8. Machine Learning Classification")
    md_text(
        "The ML pipeline uses **raw 14 EEG channels** with temporal (chronological) splits "
        "to evaluate classification under realistic deployment conditions. Key design choices:\n\n"
        "- **No shuffling**: all splits are chronological to prevent data leakage\n"
        "- **Class weighting**: `class_weight='balanced'` (LogReg, RF) and "
        "`scale_pos_weight` (XGBoost) compensate for temporal class drift\n"
        "- **Threshold optimization**: CV-optimised decision threshold applied consistently "
        "to both CV and test partitions\n"
        "- **Primary metric: Macro-F1** — equally weights both eye states under distribution shift\n\n"
        "**Models selected:** LogisticRegression (well-calibrated baseline), "
        "RandomForest (robust nonlinear), XGBoost (best gradient boosting with native "
        "imbalance handling). GradientBoosting and SVM are excluded — GB lacks "
        "`class_weight` support, and SVM is prohibitively slow with no accuracy advantage."
    )

    subtitle("8.1 Temporal Concept Drift Diagnosis")
    md_text(
        "The subject's eye-state distribution changes dramatically over the recording. "
        "Every hold-out split places the test window in the heavily open-dominant tail."
    )
    drift_rows = []
    for q in range(4):
        s, e = q * N // 4, (q + 1) * N // 4
        seg = y_all[s:e]
        cnts = np.bincount(seg, minlength=2)
        drift_rows.append([f"Q{q+1} [{s}–{e}]", cnts[0], cnts[1],
                           f"{cnts[1]/len(seg)*100:.1f}%"])
    for label, s in [("Last 10%", int(N * 0.90)),
                      ("Last 15%", int(N * 0.85)),
                      ("Last 20%", int(N * 0.80))]:
        seg = y_all[s:]
        cnts = np.bincount(seg, minlength=2)
        drift_rows.append([label, cnts[0], cnts[1],
                           f"{cnts[1]/len(seg)*100:.1f}%"])
    md_table(["Segment", "Open", "Closed", "% Closed"], drift_rows)
    md_blockquote(
        "**Warning:** The last 15% of the recording is only ~8% closed-eye. "
        "Models trained on balanced data (~50% closed) face a ~45% distribution shift. "
        "Accuracy is misleading — **Macro-F1 is the honest metric**."
    )

    subtitle("8.2 Split Configurations")
    SPLIT_CONFIGS = [
        ("70/15/15", 0.70, 0.15),
        ("60/20/20", 0.60, 0.20),
        ("80/10/10", 0.80, 0.10),
    ]
    split_info_rows = []
    for sl, tr_f, cv_f in SPLIT_CONFIGS:
        X_tr, y_tr, X_cv, y_cv, X_te, y_te = temporal_three_way_split(X_all, y_all, tr_f, cv_f)
        split_info_rows.append([sl, len(X_tr), len(X_cv), len(X_te),
                                 f"{y_tr.mean():.1%}", f"{y_cv.mean():.1%}",
                                 f"{y_te.mean():.1%}",
                                 f"{abs(y_tr.mean() - y_te.mean()):.1%}"])
    md_table(["Split", "Train N", "CV N", "Test N",
              "Train Closed%", "CV Closed%", "Test Closed%", "Δ Shift"], split_info_rows)

    # 8.3 Cross-Validation (TimeSeriesSplit)
    subtitle("8.3 Cross-Validation Results (5-Fold TimeSeriesSplit)")
    md_text(
        "5-fold time-series CV on the 70/15 training portion. Each fold trains on "
        "all preceding data, respecting temporal order."
    )
    X_tr70, y_tr70, _, _, _, _ = temporal_three_way_split(X_all, y_all, 0.70, 0.15)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    for name, model in _get_ml_models(y_tr70).items():
        scores = cross_val_score(model, X_tr70, y_tr70, cv=tscv, scoring="f1_macro")
        cv_rows.append([name, f"{scores.mean():.4f}", f"{scores.std():.4f}"])
    md_table(["Model", "CV Macro-F1 Mean", "CV Macro-F1 Std"], cv_rows)

    # 8.4 Hold-Out Split Results
    subtitle("8.4 Hold-Out Split Results")

    model_math = {
        "LogisticRegression": (
            "Logistic Regression models the posterior probability:\n\n"
            "$$P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = "
            "\\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$\n\n"
            "Uses `class_weight='balanced'` to penalise minority-class misclassification."
        ),
        "RandomForest": (
            "Random Forest builds 200 decision trees, each trained on a bootstrapped subset:\n\n"
            "$$\\hat{y} = \\text{mode}\\{h_b(\\mathbf{x})\\}_{b=1}^{200}$$\n\n"
            "Uses `class_weight='balanced'` and splits by Gini impurity."
        ),
        "XGBoost": (
            "XGBoost uses `scale_pos_weight = n_neg / n_pos` (computed from **training data only**) "
            "to handle class imbalance directly in the gradient computation."
        ),
    }

    summary_rows_ml = []
    all_roc_data = {}  # store for ROC plot

    for split_label, tr_frac, cv_frac in SPLIT_CONFIGS:
        subsubtitle(f"Split {split_label}")
        X_tr, y_tr, X_cv, y_cv, X_te, y_te = temporal_three_way_split(
            X_all, y_all, tr_frac, cv_frac)
        md_text(
            f"Train={len(X_tr)} ({y_tr.mean():.1%} closed) | "
            f"CV={len(X_cv)} ({y_cv.mean():.1%} closed) | "
            f"Test={len(X_te)} ({y_te.mean():.1%} closed) | "
            f"Δ shift={abs(y_tr.mean()-y_te.mean()):.1%}"
        )

        split_test_rows = []
        # FIX: pass y_tr (not y_all) for proper class weight computation
        for name, model in _get_ml_models(y_tr).items():
            if split_label == "70/15/15" and name in model_math:
                md_print(f"**{name}:** {model_math[name]}")
            res = _run_single_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te)
            te  = res["Test"]
            md_text(
                f"Acc={te['acc']:.4f} | MacroF1={te['macro_f1']:.4f} | "
                f"BinaryF1={te['binary_f1']:.4f} | AUC={te['auc']:.4f} | "
                f"Threshold={res['opt_t']:.2f} | TrainTime={res['train_time']:.1f}s"
            )
            _md_confusion_matrix(y_te, te["pred"])
            split_test_rows.append([name,
                                     f"{te['acc']:.4f}", f"{te['macro_f1']:.4f}",
                                     f"{te['precision']:.4f}", f"{te['recall']:.4f}",
                                     f"{te['auc']:.4f}", f"{res['opt_t']:.2f}"])
            summary_rows_ml.append({"split": split_label, "model": name,
                                     **{k: v for k, v in te.items() if k not in ("raw_prob", "pred")},
                                     "threshold": res["opt_t"], "type": "ML"})
            # Save ROC data for 70/15/15
            if split_label == "70/15/15":
                all_roc_data[name] = te["raw_prob"]

        md_text(f"**{split_label} — ML Test Summary (ranked by Macro-F1):**")
        split_test_rows.sort(key=lambda r: float(r[2]), reverse=True)
        md_table(["Model", "Acc", "MacroF1", "Prec(M)", "Rec(M)", "AUC", "Thresh"],
                 split_test_rows)

    # 8.5 Walk-Forward CV
    subtitle("8.5 Walk-Forward CV (Expanding Window) — 5 Folds")
    md_text(
        "Expanding-window walk-forward CV simulates real deployment: the model "
        "always trains on all available past data before predicting the next window. "
        "A fixed threshold of 0.5 is used for unbiased evaluation."
    )
    wf_agg = defaultdict(list)
    for fi, (tr_sl, val_sl) in enumerate(walk_forward_cv_indices(N)):
        X_tr, y_tr   = X_all[tr_sl], y_all[tr_sl]
        X_val, y_val = X_all[val_sl], y_all[val_sl]
        md_text(f"Fold {fi+1} — train={len(X_tr)} | val={len(X_val)} | val_closed={y_val.mean():.2%}")
        for name, model in _get_ml_models(y_tr).items():
            model.fit(X_tr, y_tr)
            prob  = model.predict_proba(X_val)[:, 1]
            pred  = (prob >= 0.5).astype(int)
            m = _evaluate_ml(y_val, pred, prob)
            wf_agg[name].append(m)
            md_print(f"  {name}: Acc={m['acc']:.4f} MacroF1={m['macro_f1']:.4f} AUC={m['auc']:.4f}")

    md_text("**Walk-Forward CV — Mean ± Std (primary: Macro-F1):**")
    wf_rows = []
    for name, folds in wf_agg.items():
        mf1s = [f["macro_f1"] for f in folds]
        accs = [f["acc"]      for f in folds]
        aucs = [f["auc"] if not math.isnan(f["auc"]) else 0.0 for f in folds]
        wf_rows.append([name,
                         f"{np.mean(mf1s):.4f}±{np.std(mf1s):.4f}",
                         f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
                         f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"])
    md_table(["Model", "MacroF1 Mean±Std", "Acc Mean±Std", "AUC Mean±Std"], wf_rows)

    # Feature importance
    md_text("**Feature Importance (RandomForest — 70/15/15 training partition):**")
    X_tr70, y_tr70, _, _, _, _ = temporal_three_way_split(X_all, y_all, 0.70, 0.15)
    scaler_fi = StandardScaler()
    X_tr70_s  = scaler_fi.fit_transform(X_tr70)
    rf_fi = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=RANDOM_SEED)
    rf_fi.fit(X_tr70_s, y_tr70)
    importances = rf_fi.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([FEATURE_COLUMNS[i] for i in sorted_idx],
           importances[sorted_idx], color="#3498db", edgecolor="black")
    ax.set_title("RandomForest Feature Importance (raw 14 channels)")
    ax.set_xlabel("Channel"); ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    md_image(save_fig("ml_feature_importance.png"), "Feature Importance")

    # ROC curves (using saved probs, no re-training)
    _, _, _, _, X_te70, y_te70 = temporal_three_way_split(X_all, y_all, 0.70, 0.15)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in all_roc_data.items():
        auc_val = _safe_auc(y_te70, y_prob)
        if not math.isnan(auc_val):
            fpr, tpr, _ = roc_curve(y_te70, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ML Models (70/15/15 Test Partition)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plt.tight_layout()
    md_image(save_fig("ml_roc_curves.png"), "ML ROC Curves")

    return summary_rows_ml

# =============================================================================
# 9. DL Pipeline — PyTorch
# =============================================================================

# FIX: label is last sample in window, not next sample after window
def build_sequences(X_flat, y_flat, seq_len):
    """(N, F) → (M, seq_len, F) overlapping windows. Label = last sample in window."""
    Xs, ys = [], []
    for i in range(len(X_flat) - seq_len + 1):
        Xs.append(X_flat[i: i + seq_len])
        ys.append(y_flat[i + seq_len - 1])  # FIX: was i + seq_len (look-ahead leak)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def make_dl_loaders(X_tr, y_tr, X_cv, y_cv, seq_len, batch):
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_cv_s   = scaler.transform(X_cv)
    Xs_tr, ys_tr = build_sequences(X_tr_s, y_tr, seq_len)
    Xs_cv, ys_cv = build_sequences(X_cv_s, y_cv, seq_len)
    classes, counts = np.unique(ys_tr, return_counts=True)
    class_weights   = torch.tensor(
        [counts.sum() / (len(classes) * c) for c in counts], dtype=torch.float32)
    def to_loader(Xs, ys, shuffle=False):
        return DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                          batch_size=batch, shuffle=shuffle)
    # FIX: shuffle=True for training loader to improve convergence
    return (to_loader(Xs_tr, ys_tr, shuffle=True),
            to_loader(Xs_cv, ys_cv, shuffle=False),
            scaler, class_weights, ys_cv)


# ── Architectures ──────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Stacked bidirectional LSTM → global average pool → MLP head."""
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


class CNNLSTMClassifier(nn.Module):
    """1-D conv feature extractor → bidirectional LSTM → classifier."""
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64,  3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,        128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))
    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018) — depthwise + separable 2D convolutions.
    Input: (B, T, C) → (B, 1, C, T).
    """
    def __init__(self, n_channels=14, T=64, F1=8, D=2, dropout=0.25):
        super().__init__()
        F2     = F1 * D
        kern_t = T // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, n_channels, T)
            flat_dim = self.flatten(self.block2(self.block1(dummy))).shape[1]
        self.head = nn.Linear(flat_dim, 2)
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        return self.head(self.flatten(self.block2(self.block1(x))))


# ── Training loop ──────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimiser, criterion):
    model.train(); total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def _predict_dl(model, loader):
    model.eval(); preds, probs = [], []
    for Xb, _ in loader:
        logits = model(Xb.to(DEVICE))
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def _cv_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            total += criterion(model(Xb), yb).item() * len(yb)
    return total / len(loader.dataset)


def _run_dl(name, model, tr_loader, cv_loader, y_cv_seq,
             X_te, y_te, scaler, seq_len, class_weights):
    model.to(DEVICE)
    cw        = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=DL_EPOCHS)

    epoch_rows = []
    train_losses = []
    cv_losses    = []

    # FIX: detect training failure (mode collapse)
    best_cv_mf1 = 0.0
    stale_count = 0

    for epoch in range(1, DL_EPOCHS + 1):
        loss = _train_epoch(model, tr_loader, optimiser, criterion)
        scheduler.step()
        train_losses.append(loss)
        val_loss = _cv_loss(model, cv_loader, criterion)
        cv_losses.append(val_loss)
        if epoch % 5 == 0:
            cv_preds, _ = _predict_dl(model, cv_loader)
            mf1 = f1_score(y_cv_seq, cv_preds, average="macro", zero_division=0)
            epoch_rows.append([epoch, f"{loss:.4f}", f"{val_loss:.4f}", f"{mf1:.4f}"])
            if mf1 > best_cv_mf1:
                best_cv_mf1 = mf1
                stale_count = 0
            else:
                stale_count += 1

    md_table(["Epoch", "Train Loss", "CV Loss", "CV Macro-F1"], epoch_rows)

    if best_cv_mf1 < 0.30:
        md_text(f"> ⚠ **Training warning:** {name} CV Macro-F1 never exceeded {best_cv_mf1:.3f}. "
                "This may indicate mode collapse or architecture mismatch with this dataset size.")

    # Loss curve plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, DL_EPOCHS + 1), train_losses, label="Train Loss", color="#3498db", linewidth=1.5)
    ax.plot(range(1, DL_EPOCHS + 1), cv_losses, label="CV Loss", color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted Cross-Entropy Loss")
    ax.set_title(f"{name} — Train vs CV Loss Curve"); ax.legend()
    ax.grid(True, alpha=0.3); plt.tight_layout()
    md_image(save_fig(f"dl_loss_curve_{name.lower()}.png"), f"{name} Loss Curve")

    cv_preds, cv_probs = _predict_dl(model, cv_loader)
    opt_t, _           = _optimize_threshold(y_cv_seq, cv_probs)
    cv_pred_opt        = (cv_probs >= opt_t).astype(int)
    cv_res             = _evaluate_ml(y_cv_seq, cv_pred_opt, cv_probs)

    X_te_s         = scaler.transform(X_te)
    Xs_te, ys_te   = build_sequences(X_te_s, y_te, seq_len)
    te_loader      = DataLoader(
        TensorDataset(torch.tensor(Xs_te), torch.tensor(ys_te)),
        batch_size=DL_BATCH, shuffle=False)
    te_preds, te_probs = _predict_dl(model, te_loader)
    te_pred_opt        = (te_probs >= opt_t).astype(int)
    te_res             = _evaluate_ml(ys_te, te_pred_opt, te_probs)
    te_res["pred"]     = te_pred_opt
    te_res["raw_prob"] = te_probs

    md_text(f"Optimal threshold (CV-optimised): **{opt_t:.2f}**")
    md_table(
        ["Partition", "Acc", "MacroF1", "BinaryF1", "Prec(M)", "Rec(M)", "AUC"],
        [["CV",   f"{cv_res['acc']:.4f}", f"{cv_res['macro_f1']:.4f}",
                  f"{cv_res['binary_f1']:.4f}", f"{cv_res['precision']:.4f}",
                  f"{cv_res['recall']:.4f}", f"{cv_res['auc']:.4f}"],
         ["Test", f"{te_res['acc']:.4f}", f"{te_res['macro_f1']:.4f}",
                  f"{te_res['binary_f1']:.4f}", f"{te_res['precision']:.4f}",
                  f"{te_res['recall']:.4f}", f"{te_res['auc']:.4f}"]])
    md_text("**Test Confusion Matrix:**")
    _md_confusion_matrix(ys_te, te_pred_opt)

    return {"CV": cv_res, "Test": te_res, "opt_t": opt_t,
            "y_te_seq": ys_te, "te_probs": te_probs}


# ── Ensemble ──────────────────────────────────────────────────────────────────

class EnsembleOptimizer:
    def __init__(self):
        self.best_weights = None
        self.model_names  = []

    def optimize(self, probs_cv_dict, y_cv, n_trials=ENS_TRIALS):
        self.model_names = list(probs_cv_dict.keys())
        k        = len(self.model_names)
        prob_mat = np.column_stack([probs_cv_dict[n] for n in self.model_names])

        best_t_ens = 0.5
        def eval_w(w, threshold=0.5):
            ens_prob = prob_mat @ w
            preds    = (ens_prob >= threshold).astype(int)
            return f1_score(y_cv, preds, average="macro", zero_division=0)

        rng     = np.random.RandomState(RANDOM_SEED)
        best_w  = np.ones(k) / k
        best_f1 = eval_w(best_w)

        for _ in range(n_trials):
            w = rng.dirichlet(np.ones(k))
            # FIX: also search over thresholds during weight optimization
            for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                s = eval_w(w, t)
                if s > best_f1:
                    best_f1, best_w, best_t_ens = s, w.copy(), t

        self.best_weights = best_w
        self.best_threshold = best_t_ens
        return dict(zip(self.model_names, best_w.tolist())), best_f1, best_t_ens

    def predict(self, probs_test_dict, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        prob_mat = np.column_stack([probs_test_dict[n] for n in self.model_names])
        ens_prob = prob_mat @ self.best_weights
        return (ens_prob >= threshold).astype(int), ens_prob


def section_dl(X_all, y_all):
    title("9. Deep Learning Classification")
    md_text(
        "All DL models use PyTorch with: "
        "**(1) weighted CrossEntropyLoss** (inverse class frequency), "
        "**(2) AdamW + CosineAnnealingLR**, "
        "**(3) CV-optimised decision threshold**, and "
        "**(4) Macro-F1 as primary metric**. "
        "Sequences are built per partition with no cross-boundary leakage. "
        "Label = last sample in window (not look-ahead).\n\n"
        "**Models selected:** LSTM (temporal baseline), CNN-LSTM (local+temporal), "
        "EEGNet (EEG-specific, ~1.1K params). EEGTransformer and PatchTST are excluded "
        "— they suffer mode collapse on this dataset size (~14K samples)."
    )

    subtitle("9.0 Architecture Overview & Training Setup")
    md_text(
        "**Weighted Cross-Entropy Loss:**\n\n"
        "$$\\mathcal{L} = -\\frac{1}{N}\\sum_{i=1}^{N} w_{y_i} "
        "\\log\\left(\\frac{e^{z_{y_i}}}{\\sum_{c=0}^{1} e^{z_c}}\\right)$$\n\n"
        "where $w_c = \\frac{N}{2 \\cdot N_c}$ is the per-class weight. "
        "**Sequence length:** SEQ_LEN=64 samples (≈500ms at 128 Hz). "
        "**Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. "
        "**Scheduler:** CosineAnnealingLR over 25 epochs."
    )
    md_table(
        ["Model", "Architecture", "Parameters", "Key Innovation"],
        [
            ["LSTM",     "BiLSTM(128)×2 → AvgPool → MLP",     "~200K", "Long-range temporal dependencies"],
            ["CNN-LSTM", "Conv1D(64,128) → BiLSTM(64) → MLP", "~150K", "Local feature extraction + sequence memory"],
            ["EEGNet",   "Depthwise Conv2D blocks → Linear",   "~1.1K", "Electrode-aware, compact, best calibrated"],
        ]
    )

    SPLIT_CONFIGS = [
        ("70/15/15", 0.70, 0.15),
        ("60/20/20", 0.60, 0.20),
        ("80/10/10", 0.80, 0.10),
    ]

    N_FEATURES    = len(FEATURE_COLUMNS)
    summary_rows_dl = []

    dl_model_factories = {
        "LSTM":     lambda: LSTMClassifier(N_FEATURES),
        "CNN_LSTM": lambda: CNNLSTMClassifier(N_FEATURES),
        "EEGNet":   lambda: EEGNet(n_channels=N_FEATURES, T=SEQ_LEN),
    }

    arch_descriptions = {
        "LSTM": (
            "Stacked bidirectional LSTM captures long-range temporal dependencies.\n\n"
            "$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, "
            "\\quad h_t = o_t \\odot \\tanh(c_t)$$"
        ),
        "CNN_LSTM": (
            "Two 1D convolutional blocks extract local temporal features; "
            "a bidirectional LSTM then models sequence dynamics.\n\n"
            "$$y_t^{(f)} = \\text{ReLU}\\left(\\sum_{k,c} w_{k,c}^{(f)} \\cdot x_{t+k,c} + b^{(f)}\\right)$$"
        ),
        "EEGNet": (
            "EEGNet (Lawhern et al. 2018) uses depthwise-separable 2D convolutions that "
            "explicitly model temporal patterns (Block 1 temporal kernel ≈ 250ms) "
            "and cross-electrode spatial patterns. Only ~1.1K parameters — highly "
            "resistant to overfitting on limited data."
        ),
    }

    for split_label, tr_frac, cv_frac in SPLIT_CONFIGS:
        subsubtitle(f"Split {split_label}")
        X_tr, y_tr, X_cv, y_cv, X_te, y_te = temporal_three_way_split(
            X_all, y_all, tr_frac, cv_frac)
        md_text(
            f"Train={len(X_tr)} ({y_tr.mean():.1%} closed) | "
            f"CV={len(X_cv)} ({y_cv.mean():.1%} closed) | "
            f"Test={len(X_te)} ({y_te.mean():.1%} closed)"
        )

        dl_cv_probs  = {}
        dl_te_probs  = {}
        y_te_seq_ref = None
        split_dl_rows = []

        for arch_name in dl_model_factories:
            if split_label == "70/15/15":
                subtitle(f"9.{list(dl_model_factories.keys()).index(arch_name)+1} {arch_name}")
                if arch_name in arch_descriptions:
                    md_text(arch_descriptions[arch_name])

            (tr_loader, cv_loader,
             scaler, cw, y_cv_seq) = make_dl_loaders(
                X_tr, y_tr, X_cv, y_cv, SEQ_LEN, DL_BATCH)

            model = dl_model_factories[arch_name]()
            res   = _run_dl(arch_name, model, tr_loader, cv_loader, y_cv_seq,
                             X_te, y_te, scaler, SEQ_LEN, cw)

            _, cv_p  = _predict_dl(model, cv_loader)
            dl_cv_probs[arch_name] = cv_p
            dl_te_probs[arch_name] = res["te_probs"]
            if y_te_seq_ref is None:
                y_te_seq_ref = res["y_te_seq"]

            te = res["Test"]
            split_dl_rows.append([arch_name,
                                   f"{te['acc']:.4f}", f"{te['macro_f1']:.4f}",
                                   f"{te['precision']:.4f}", f"{te['recall']:.4f}",
                                   f"{te['auc']:.4f}", f"{res['opt_t']:.2f}"])
            summary_rows_dl.append({"split": split_label, "model": arch_name,
                                     **{k: v for k, v in te.items() if k not in ("raw_prob", "pred")},
                                     "threshold": res["opt_t"], "type": "DL"})

        # Ensemble
        if split_label == "70/15/15":
            subtitle("9.4 Soft-Vote Ensemble")
        md_text(f"**Soft-Vote Ensemble — {split_label}**")
        md_text("Random-weight Dirichlet search with threshold co-optimization.")
        ens_opt = EnsembleOptimizer()
        w_dict, best_f1, best_t = ens_opt.optimize(dl_cv_probs, y_cv_seq)

        md_text(f"Optimal weights (CV Macro-F1 = {best_f1:.4f}, threshold = {best_t:.2f}):")
        w_rows = sorted(w_dict.items(), key=lambda x: -x[1])
        bar_max = max(w_dict.values())
        md_table(["Model", "Weight", "Contribution"],
                 [[n, f"{w:.4f}", "█" * max(1, int(w / bar_max * 25))] for n, w in w_rows])

        ens_pred, ens_prob = ens_opt.predict(dl_te_probs)
        ens_res = _evaluate_ml(y_te_seq_ref, ens_pred, ens_prob)
        md_text(f"Ensemble Test (t={best_t:.2f}): "
                f"Acc={ens_res['acc']:.4f} | MacroF1={ens_res['macro_f1']:.4f} | "
                f"AUC={ens_res['auc']:.4f}")
        _md_confusion_matrix(y_te_seq_ref, ens_pred)

        split_dl_rows.append(["Ensemble",
                               f"{ens_res['acc']:.4f}", f"{ens_res['macro_f1']:.4f}",
                               f"{ens_res['precision']:.4f}", f"{ens_res['recall']:.4f}",
                               f"{ens_res['auc']:.4f}", f"{best_t:.2f}"])
        summary_rows_dl.append({"split": split_label, "model": "Ensemble",
                                 **{k: v for k, v in ens_res.items() if k not in ("raw_prob", "pred")},
                                 "threshold": best_t, "type": "DL"})

        if split_label == "70/15/15":
            subtitle("9.5 DL Model Comparison")
        md_text(f"**DL Model Comparison — {split_label}:**")
        split_dl_rows.sort(key=lambda r: float(r[2]), reverse=True)
        md_table(["Model", "Acc", "MacroF1", "Prec(M)", "Rec(M)", "AUC", "Thresh"],
                 split_dl_rows)

    return summary_rows_dl

# =============================================================================
# 10. Final Comparison
# =============================================================================

def section_final_comparison(summary_rows_ml, summary_rows_dl):
    title("10. Final Comparison and Inference")
    md_text(
        "This section unifies all models across all evaluation protocols. "
        "Primary metric: **Macro-F1**."
    )

    df_sum = pd.DataFrame(summary_rows_ml + summary_rows_dl)

    subtitle("10.1 Unified Model Comparison")
    md_text("All test-partition results across all hold-out splits, sorted by Macro-F1.")

    for split_label in ["70/15/15", "60/20/20", "80/10/10"]:
        subsubtitle(f"Split {split_label}")
        sub  = df_sum[df_sum["split"] == split_label].sort_values("macro_f1", ascending=False)
        rows = [[r["model"], r["type"],
                 f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}",
                 f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                 f"{r.get('auc', float('nan')):.4f}",
                 f"{r.get('threshold', 0.5):.2f}"]
                for _, r in sub.iterrows()]
        md_table(["Model", "Type", "Acc", "MacroF1", "Prec(M)", "Rec(M)", "AUC", "Thresh"], rows)

    # Bar chart — 70/15/15 test results
    sub70 = df_sum[df_sum["split"] == "70/15/15"].sort_values("macro_f1", ascending=False)
    if not sub70.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        names = sub70["model"].tolist()
        x = np.arange(len(names))
        w = 0.35
        axes[0].bar(x - w/2, sub70["acc"].tolist(),      w, label="Accuracy",  color="#3498db", edgecolor="black")
        axes[0].bar(x + w/2, sub70["macro_f1"].tolist(), w, label="MacroF1",   color="#e74c3c", edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("All Models — Acc vs MacroF1 (70/15/15)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

        aucs = [v if not math.isnan(v) else 0.0 for v in sub70["auc"].tolist()]
        axes[1].bar(names, aucs, color="#9370DB", edgecolor="black")
        axes[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("AUC-ROC (70/15/15 Test Partition)")
        axes[1].grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        md_image(save_fig("final_comparison.png"), "Final Model Comparison")

    subtitle("10.2 Inference and Recommendation")

    # Best models per split
    best_per_split = []
    for split_label in ["70/15/15", "60/20/20", "80/10/10"]:
        sub = df_sum[df_sum["split"] == split_label]
        if sub.empty:
            continue
        best = sub.loc[sub["macro_f1"].idxmax()]
        best_per_split.append([split_label, best["model"], best.get("type", "?"),
                                f"{best['macro_f1']:.4f}", f"{best['acc']:.4f}",
                                f"{best.get('auc', float('nan')):.4f}"])

    md_text("**Best model per hold-out split (by Macro-F1):**")
    md_table(["Split", "Best Model", "Type", "MacroF1", "Acc", "AUC"], best_per_split)

    # Cross-split average Macro-F1
    mean_mf1 = df_sum.groupby("model")["macro_f1"].mean().sort_values(ascending=False)
    mean_rows = [[m, f"{v:.4f}"] for m, v in mean_mf1.items()]
    md_text("**Mean Macro-F1 across all three splits (stability ranking):**")
    md_table(["Model", "Mean MacroF1"], mean_rows)

    best_overall = mean_mf1.index[0]
    best_mf1_val = mean_mf1.iloc[0]
    md_text(f"### Best Overall Model: **{best_overall}**")
    md_text(
        f"Based on mean Macro-F1 across all three temporal hold-out splits, "
        f"**{best_overall}** achieves the highest average score of **{best_mf1_val:.4f}**."
    )

    md_text("**Key Observations:**")
    md_text(
        "- The last 15% of the recording is ~8% closed-eye, creating a ~45% distribution "
        "shift between training and test. This is the root cause of all metric paradoxes.\n"
        "- Models with well-calibrated probabilities (LogReg, EEGNet) transfer thresholds "
        "across the distribution shift more reliably than uncalibrated models.\n"
        "- All models struggle under severe concept drift; Macro-F1 values near 0.50 "
        "indicate performance only marginally above the balanced-accuracy baseline."
    )

    # Recommendations
    _best_overall = mean_mf1.index[0]
    _best_ml_candidates = [m for m in mean_mf1.index if df_sum[df_sum["model"]==m]["type"].iloc[0]=="ML"]
    _best_ml = _best_ml_candidates[0] if _best_ml_candidates else "LogisticRegression"
    md_text("**Recommended Model Per Use Case:**")
    md_table(
        ["Use Case", "Model", "Reason"],
        [
            ["Balanced accuracy (research)",  _best_overall, f"Highest mean Macro-F1 ({mean_mf1.iloc[0]:.4f})"],
            ["Stable production ML",          _best_ml,      "Most consistent ML model across splits, fast inference"],
            ["Online/streaming BCI",          "EEGNet",      "~1.1K params, fast inference, electrode-aware architecture"],
        ]
    )

    # Dataset suitability appendix
    md_text("---")
    subsubtitle("Appendix: Dataset Suitability for Neural Network Training")
    md_table(
        ["Criterion", "Verdict", "Explanation"],
        [
            ["Sample size",          "⚠ Marginal",          "~14 k total; DL typically needs >50 k sequences"],
            ["Single subject",       "✗ Poor generalisation","All 14,980 samples from one 117-second session"],
            ["Temporal continuity",  "⚠ Concept drift",     "Eye-state ratio shifts from 50% to 6% closed over recording"],
            ["Preprocessing",        "✓ Bandpass + IQR",    "Bandpass 0.5–45 Hz + IQR cleaning preserves EEG integrity"],
            ["Class balance",        "✓ Adequate globally", "55% open / 45% closed globally; drifts at end"],
            ["Label quality",        "✓ Camera-verified",   "Eye state labels added by manual video annotation"],
        ]
    )
    md_text("---")

# =============================================================================
# Main pipeline
# =============================================================================

def main():
    progress("=" * 60)
    progress("EEG Eye State Classification — Pipeline Started")
    progress(f"[info] Device: {DEVICE}")
    progress("=" * 60)

    print_toc()

    progress("[1/10] Loading data ...")
    df = pd.read_csv(DATA_FILE)
    section_data_description(df)

    progress("[2/10] Data imputation ...")
    df = section_data_imputation(df)

    progress("[3/10] Visualising raw data ...")
    section_data_viz_raw(df)

    progress("[4/10] Signal preprocessing (bandpass + IQR) ...")
    df_raw_copy = df.copy()
    df_clean = section_preprocessing(df)

    progress("[5/10] Visualising cleaned data ...")
    section_data_viz_cleaned(df_raw_copy, df_clean)

    progress("[6/10] PSD and spectrogram analysis ...")
    section_psd_spectro(df_clean)

    progress("[7/10] Dimensionality reduction (LDA) ...")
    section_dim_reduction(df_clean)

    # ML + DL pipeline uses raw 14 channels on the cleaned dataset
    X_all = df_clean[FEATURE_COLUMNS].values.astype(np.float32)
    y_all = df_clean[TARGET].values.astype(np.int64)
    N     = len(X_all)

    progress("[8/10] Training ML models ...")
    summary_rows_ml = section_ml(X_all, y_all, N)

    progress("[9/10] Training DL models (PyTorch) ...")
    summary_rows_dl = section_dl(X_all, y_all)

    progress("[10/10] Generating final comparison ...")
    section_final_comparison(summary_rows_ml, summary_rows_dl)

    progress("=" * 60)
    progress("Pipeline complete.  Run:  python script.py > report.md")
    progress("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Eye State Classification Pipeline")
    parser.add_argument("--dataset",  type=str, default=None,
                        help="Path to CSV dataset (overrides config.yaml)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.dataset:
        DATA_FILE = args.dataset
    if args.plot_dir:
        PLOT_DIR = args.plot_dir
        os.makedirs(PLOT_DIR, exist_ok=True)

    main()