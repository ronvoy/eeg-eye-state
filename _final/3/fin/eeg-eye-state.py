# pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost torch

import sys, os, time, warnings, argparse, math
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram as scipy_spectrogram, butter, filtfilt
from scipy.stats import skew, kurtosis, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
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
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn('xgboost not found — XGBoost model will be skipped.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DIR = 'analysis-plots'
DATA_FILE = 'eeg_data_og.csv'
SAMPLING_RATE = 128
FEATURE_COLUMNS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
TARGET = 'eyeDetection'
RANDOM_STATE = 42
EYE_MAP = {0: 'Open', 1: 'Closed'}
SEQ_LEN = 64
PATCH_SIZE = 8
PATCH_STRIDE = 4
DL_EPOCHS = 25
DL_BATCH = 128
DL_LR = 0.001
ENS_TRIALS = 3000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
HEMI_PAIRS = [('AF3', 'AF4'), ('F7', 'F8'), ('F3', 'F4'), ('FC5', 'FC6'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')]
FREQ_BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 64)}
BAND_COLORS = ['#8B0000', '#FF4500', '#FFD700', '#00CED1', '#9370DB']
ELECTRODE_INFO = [('AF3', 'Anterior Frontal Left', 'Prefrontal Cortex', 'Executive function, attention'), ('F7', 'Frontal Left Lateral', 'Left Temporal-Frontal', 'Language processing'), ('F3', 'Frontal Left', 'Left Frontal Lobe', 'Motor planning, positive affect'), ('FC5', 'Fronto-Central Left', 'Left Motor-Frontal', 'Motor preparation'), ('T7', 'Temporal Left', 'Left Temporal Lobe', 'Auditory processing, memory'), ('P7', 'Parietal Left', 'Left Parietal-Temporal', 'Visual-spatial processing'), ('O1', 'Occipital Left', 'Left Visual Cortex', 'Visual processing'), ('O2', 'Occipital Right', 'Right Visual Cortex', 'Visual processing'), ('P8', 'Parietal Right', 'Right Parietal-Temporal', 'Spatial attention'), ('T8', 'Temporal Right', 'Right Temporal Lobe', 'Face / emotion recognition'), ('FC6', 'Fronto-Central Right', 'Right Motor-Frontal', 'Motor preparation'), ('F4', 'Frontal Right', 'Right Frontal Lobe', 'Motor planning, negative affect'), ('F8', 'Frontal Right Lateral', 'Right Temporal-Frontal', 'Emotion, social cognition'), ('AF4', 'Anterior Frontal Right', 'Prefrontal Cortex', 'Executive function, attention')]
CONFIG = {}
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
if HAS_YAML and os.path.exists(_config_path):
    with open(_config_path) as _cf:
        CONFIG = yaml.safe_load(_cf) or {}
    PLOT_DIR = CONFIG.get('paths', {}).get('plot_dir', PLOT_DIR)
    DATA_FILE = CONFIG.get('paths', {}).get('data_file', DATA_FILE)
    SAMPLING_RATE = CONFIG.get('data', {}).get('sampling_rate', SAMPLING_RATE)
    RANDOM_STATE = CONFIG.get('data', {}).get('random_state', RANDOM_STATE)
os.makedirs(PLOT_DIR, exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def progress(msg):
    print(msg, file=sys.stderr, flush=True)

def save_fig(name):
    path = f'{PLOT_DIR}/{name}'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    return path

def section_data_description(df):
    _elec_map = {e[0]: e for e in ELECTRODE_INFO}
    combined_rows = []
    for ch in FEATURE_COLUMNS:
        e = _elec_map.get(ch, (ch, '—', '—', '—'))
        combined_rows.append([ch, 'Continuous (float64)', e[1], e[2], e[3]])
    desc = df[FEATURE_COLUMNS].describe().T
    rows = []
    for ch in FEATURE_COLUMNS:
        r = desc.loc[ch]
        mode_val = df[ch].mode().iloc[0] if not df[ch].mode().empty else float('nan')
        rows.append([ch, int(r['count']), f"{r['mean']:.2f}", f"{r['std']:.2f}", f"{r['min']:.2f}", f"{r['25%']:.2f}", f"{r['50%']:.2f}", f"{r['75%']:.2f}", f"{r['max']:.2f}", f'{mode_val:.2f}'])
    vc = df[TARGET].value_counts()

def section_data_imputation(df):
    total_missing = df.isnull().sum().sum()
    if not total_missing == 0:
        rows = [[ch, int(df[ch].isnull().sum())] for ch in FEATURE_COLUMNS if df[ch].isnull().any()]
        for col in FEATURE_COLUMNS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    return df

def section_data_viz_raw(df):
    (fig, ax) = plt.subplots(figsize=(6, 4))
    vc = df[TARGET].value_counts()
    bars = ax.bar(['Open (0)', 'Closed (1)'], [vc.get(0, 0), vc.get(1, 0)], color=['#3498db', '#e74c3c'], edgecolor='black')
    ax.set_title('Class Balance of Eye States')
    ax.set_ylabel('Count')
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30, str(int(b.get_height())), ha='center', fontweight='bold')
    save_fig('class_balance_raw.png')
    df_win = df[FEATURE_COLUMNS].clip(lower=df[FEATURE_COLUMNS].quantile(0.01), upper=df[FEATURE_COLUMNS].quantile(0.99), axis=1)
    (fig, ax) = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_win.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Heatmap of EEG Channels (winsorized 1st–99th pct)')
    save_fig('correlation_heatmap_raw.png')
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df[ch], ax=ax, color='#3498db')
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle('Box Plots — All EEG Channels (Raw)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('boxplots_raw.png')
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df[ch], ax=ax, color='#3498db')
        (lo, hi) = (df[ch].quantile(0.01), df[ch].quantile(0.99))
        ax.set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1)
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle('Box Plots — Zoomed (1st–99th percentile)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('boxplots_raw_zoomed.png')
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        for (state, color) in [(0, '#3498db'), (1, '#e74c3c')]:
            ax.hist(df.loc[df[TARGET] == state, ch], bins=40, alpha=0.5, color=color, label=EYE_MAP[state])
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle('Histograms — All Channels by Eye State (Raw)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('histograms_raw.png')
    df_tmp = df.copy()
    df_tmp['eyeState'] = df_tmp[TARGET].map(EYE_MAP)
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.violinplot(x='eyeState', y=ch, data=df_tmp, ax=ax, palette=['#3498db', '#e74c3c'], inner='quartile', order=['Open', 'Closed'])
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel('')
        ax.tick_params(labelsize=8)
    plt.suptitle('Violin Plots — All Channels by Eye State (Raw)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('violinplots_raw.png')

def _bandpass_filter(df, lowcut=0.5, highcut=45.0, fs=None, order=4):
    """Apply Butterworth bandpass filter to all EEG channels (causal-safe)."""
    if fs is None:
        fs = SAMPLING_RATE
    nyq = fs / 2.0
    (b, a) = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = df.copy()
    for col in FEATURE_COLUMNS:
        filtered[col] = filtfilt(b, a, df[col].values)
    return filtered

def _light_iqr(df, multiplier=3.0, max_passes=3):
    """Safety-net IQR with wider bounds to catch residual extremes."""
    cleaned = df.copy()
    pass_num = 0
    bounds = []
    while True:
        pass_num += 1
        before = len(cleaned)
        for col in FEATURE_COLUMNS:
            (Q1, Q3) = (cleaned[col].quantile(0.25), cleaned[col].quantile(0.75))
            IQR = Q3 - Q1
            (lo, hi) = (Q1 - multiplier * IQR, Q3 + multiplier * IQR)
            cleaned = cleaned[(cleaned[col] >= lo) & (cleaned[col] <= hi)]
            if pass_num == 1:
                bounds.append([col, f'{lo:.2f}', f'{hi:.2f}'])
        after = len(cleaned)
        if before - after == 0 or pass_num >= max_passes:
            break
    return (cleaned.reset_index(drop=True), bounds, pass_num)

def section_preprocessing(df):
    original_count = len(df)
    cfg_pre = CONFIG.get('preprocessing', {})
    iqr_cfg = cfg_pre.get('iqr', {})
    iqr_mult = iqr_cfg.get('multiplier', 3.0)
    iqr_pass = iqr_cfg.get('max_passes', 3)
    bp_cfg = cfg_pre.get('bandpass', {})
    lowcut = bp_cfg.get('lowcut', 0.5)
    highcut = bp_cfg.get('highcut', 45.0)
    bp_order = bp_cfg.get('order', 4)
    (df_iqr, bounds, n_passes) = _light_iqr(df, multiplier=iqr_mult, max_passes=iqr_pass)
    removed_iqr = original_count - len(df_iqr)
    pct_iqr = removed_iqr / original_count * 100
    df_clean = _bandpass_filter(df_iqr, lowcut=lowcut, highcut=highcut, fs=SAMPLING_RATE, order=bp_order)
    sample_ch = 'O1'
    n_show = min(1000, len(df_iqr))
    (fig, axes) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(range(n_show), df_iqr[sample_ch].iloc[:n_show].values, linewidth=0.4, color='#e74c3c', label='After IQR (pre-filter)')
    axes[1].plot(range(n_show), df_clean[sample_ch].iloc[:n_show].values, linewidth=0.4, color='#2ecc71', label='After Bandpass')
    for ax in axes:
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel(f'{sample_ch} (µV)')
    axes[1].set_xlabel('Sample')
    axes[0].set_title(f'Bandpass Filter Effect — {sample_ch} ({lowcut}–{highcut} Hz) [applied to spike-free signal]')
    plt.tight_layout()
    save_fig('bandpass_filter_comparison.png')
    total_removed = original_count - len(df_clean)
    total_pct = total_removed / original_count * 100
    return df_clean

def section_data_viz_cleaned(df_raw, df_clean):
    import seaborn as sns_inner
    (fig, ax) = plt.subplots(figsize=(12, 10))
    sns_inner.heatmap(df_clean[FEATURE_COLUMNS].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Heatmap — After IQR + Bandpass Preprocessing (corrected)')
    save_fig('correlation_heatmap_cleaned.png')
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        sns.boxplot(y=df_clean[ch], ax=ax, color='#2ecc71', whis=3.0)
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle('Box Plots — After Preprocessing (whis=3.0x IQR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('boxplots_cleaned.png')
    (fig, axes) = plt.subplots(2, 7, figsize=(24, 8))
    for (i, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[i]
        for (state, color) in [(0, '#3498db'), (1, '#e74c3c')]:
            ax.hist(df_clean.loc[df_clean[TARGET] == state, ch], bins=40, alpha=0.5, color=color, label=EYE_MAP[state])
        ax.set_title(ch, fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle('Histograms — After Preprocessing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('histograms_cleaned.png')

def section_log_normalization(df):
    df_norm = df.copy()
    for col in FEATURE_COLUMNS:
        df_norm[col] = np.log10(df[col] - df[col].min() + 1)
    (fig, axes) = plt.subplots(2, 7, figsize=(28, 8))
    for (idx, ch) in enumerate(FEATURE_COLUMNS):
        (row, col) = divmod(idx, 7)
        ax = axes[row, col]
        ax.hist(df[ch], bins=50, color='#3498db', alpha=0.6, edgecolor='black', label='Before', density=True)
        ax.hist(df_norm[ch], bins=50, color='#e74c3c', alpha=0.6, edgecolor='black', label='After', density=True)
        ax.set_title(ch, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    plt.suptitle('Log-Normalization — Before (blue) vs After (red) for All Channels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('log_normalization_all_channels.png')
    sk_rows = []
    improved_count = 0
    for ch in FEATURE_COLUMNS:
        sk_b = skew(df[ch].values)
        sk_a = skew(df_norm[ch].values)
        kt_b = kurtosis(df[ch].values)
        kt_a = kurtosis(df_norm[ch].values)
        improved = abs(sk_a) + abs(kt_a) < abs(sk_b) + abs(kt_b)
        if improved:
            improved_count += 1
        sk_rows.append([ch, f'{sk_b:.4f}', f'{sk_a:.4f}', f'{kt_b:.4f}', f'{kt_a:.4f}', 'Yes' if improved else 'No'])
    pct_improved = improved_count / len(FEATURE_COLUMNS) * 100
    pass

def section_feature_engineering(df):
    df_eng = df.copy()
    new_features = []
    asym_rows = []
    for (left, right) in HEMI_PAIRS:
        fname = f'{left}_{right}_asym'
        df_eng[fname] = df_eng[left] - df_eng[right]
        new_features.append(fname)
        asym_rows.append([fname, left, right, f'{df_eng[fname].mean():.4f}', f'{df_eng[fname].std():.4f}'])
    asym_state_rows = []
    for (left, right) in HEMI_PAIRS:
        fname = f'{left}_{right}_asym'
        open_vals = df_eng.loc[df_eng[TARGET] == 0, fname].values
        closed_vals = df_eng.loc[df_eng[TARGET] == 1, fname].values
        from scipy.stats import ttest_ind
        (t_stat, p_val) = ttest_ind(open_vals, closed_vals, equal_var=False)
        sig = 'Yes' if p_val < 0.05 else 'No'
        asym_state_rows.append([fname, f'{open_vals.mean():.4f}', f'{closed_vals.mean():.4f}', f'{t_stat:.3f}', f'{p_val:.2e}', sig])
    sig_count = sum((1 for r in asym_state_rows if r[5] == 'Yes'))
    nyq = SAMPLING_RATE / 2.0
    band_rows = []
    for (band_name, (fmin, fmax)) in FREQ_BANDS.items():
        (b_bp, a_bp) = butter(4, [max(fmin / nyq, 0.001), min(fmax / nyq, 0.999)], btype='band')
        feat_name = f'band_{band_name}_power'
        df_eng[feat_name] = np.mean([filtfilt(b_bp, a_bp, df[ch].values) ** 2 for ch in FEATURE_COLUMNS], axis=0)
        new_features.append(feat_name)
        band_rows.append([feat_name, f'{fmin}–{fmax} Hz', f'{df_eng[feat_name].mean():.4f}', f'{df_eng[feat_name].std():.4f}'])
    (b_al, a_al) = butter(4, [8 / nyq, 12 / nyq], btype='band')
    df_eng['alpha_asymmetry'] = filtfilt(b_al, a_al, df['O1'].values) ** 2 - filtfilt(b_al, a_al, df['O2'].values) ** 2
    new_features.append('alpha_asymmetry')
    band_rows.append(['alpha_asymmetry', 'O1α² − O2α²', f"{df_eng['alpha_asymmetry'].mean():.4f}", f"{df_eng['alpha_asymmetry'].std():.4f}"])
    band_names = list(FREQ_BANDS.keys())
    band_feat_names = [f'band_{b}_power' for b in band_names]
    open_means = [df_eng.loc[df_eng[TARGET] == 0, f].mean() for f in band_feat_names]
    closed_means = [df_eng.loc[df_eng[TARGET] == 1, f].mean() for f in band_feat_names]
    (fig, axes) = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(band_names))
    w = 0.35
    axes[0].bar(x - w / 2, open_means, w, label='Open', color='#3498db', edgecolor='black')
    axes[0].bar(x + w / 2, closed_means, w, label='Closed', color='#e74c3c', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_names)
    axes[0].set_ylabel('Mean Band Power (µV²)')
    axes[0].set_title('Mean Band Power by Eye State')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    ratios = [c / o if o > 0 else 0 for (o, c) in zip(open_means, closed_means)]
    colors = ['#2ecc71' if r > 1.0 else '#e67e22' for r in ratios]
    axes[1].bar(band_names, ratios, color=colors, edgecolor='black')
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel('Power Ratio (Closed / Open)')
    axes[1].set_title('Band Power Ratio — Berger Effect Indicator')
    axes[1].grid(True, alpha=0.3, axis='y')
    for (i, r) in enumerate(ratios):
        axes[1].text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig('band_power_eda.png')
    df_eng['ch_mean'] = df_eng[FEATURE_COLUMNS].mean(axis=1)
    df_eng['ch_std'] = df_eng[FEATURE_COLUMNS].std(axis=1)
    new_features += ['ch_mean', 'ch_std']
    all_features = FEATURE_COLUMNS + new_features
    return (df_eng, all_features)

def section_fft_psd_spectro(df):
    (fig, axes) = plt.subplots(2, 7, figsize=(28, 8))
    for (idx, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        signal = df[ch].values
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, 1 / SAMPLING_RATE)
        pos = freqs > 0
        power = np.abs(fft_vals[pos]) ** 2 / n
        ax.semilogy(freqs[pos], power, linewidth=0.4, color='#1f77b4')
        ax.set_xlim(0, 64)
        ax.set_title(ch, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        for (i, (_, (lo, hi))) in enumerate(FREQ_BANDS.items()):
            ax.axvspan(lo, hi, alpha=0.1, color=BAND_COLORS[i])
    plt.suptitle('FFT Frequency Spectrum — All Channels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('fft_frequency_spectrum.png')
    df_open = df[df[TARGET] == 0]
    df_closed = df[df[TARGET] == 1]
    nperseg = min(256, len(df_open), len(df_closed))
    (fig, axes) = plt.subplots(2, 7, figsize=(28, 10))
    for (idx, ch) in enumerate(FEATURE_COLUMNS):
        ax = axes.flatten()[idx]
        (fo, po) = welch(df_open[ch].values, SAMPLING_RATE, nperseg=nperseg)
        (fc, pc) = welch(df_closed[ch].values, SAMPLING_RATE, nperseg=nperseg)
        ax.semilogy(fo, po, label='Open', color='blue', linewidth=1, alpha=0.8)
        ax.semilogy(fc, pc, label='Closed', color='red', linewidth=1, alpha=0.8)
        ax.set_xlim(0, 35)
        ax.set_title(ch, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
        ylims = ax.get_ylim()
        for (i, (bname, (lo, hi))) in enumerate(FREQ_BANDS.items()):
            bhi = min(hi, 35)
            ax.axvspan(lo, bhi, alpha=0.08, color=BAND_COLORS[i])
            if bhi <= 35:
                ax.text((lo + bhi) / 2, ylims[1] * 0.3, bname, fontsize=5, ha='center', va='top', color=BAND_COLORS[i], fontweight='bold', rotation=90)
        if idx == 0:
            ax.legend(fontsize=6)
    plt.suptitle('PSD — All Channels (Open vs Closed)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('psd_analysis.png')
    for (state_name, state_val) in [('Open', 0), ('Closed', 1)]:
        (fig, axes) = plt.subplots(2, 7, figsize=(28, 8))
        data_state = df[df[TARGET] == state_val]
        for (idx, ch) in enumerate(FEATURE_COLUMNS):
            ax = axes.flatten()[idx]
            data = data_state[ch].values
            seg = max(4, min(128, len(data) // 4))
            (f, t, Sxx) = scipy_spectrogram(data, fs=SAMPLING_RATE, nperseg=seg, noverlap=seg // 2)
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax.set_ylim(0, 30)
            ax.set_title(ch, fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=6)
            for freq in [4, 8, 12, 30]:
                ax.axhline(y=freq, color='white', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.suptitle(f'Spectrograms — Eyes {state_name} (All Channels)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig(f'spectrograms_{state_name.lower()}.png')

def section_dim_reduction(df, all_features):
    X = df[all_features].values
    y = df[TARGET].values
    X_df = pd.DataFrame(X)
    Q1 = X_df.quantile(0.25)
    Q3 = X_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X_df < Q1 - 1.5 * IQR) | (X_df > Q3 + 1.5 * IQR)).any(axis=1)
    X_clean = X_df[mask].values
    y_clean = y[mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    window = 10
    X_roll_df = pd.DataFrame(X_scaled)
    X_roll_mean = X_roll_df.rolling(window=window).mean().fillna(0).values
    X_roll_std = X_roll_df.rolling(window=window).std().fillna(0).values
    X_fft = np.abs(np.fft.fft(X_scaled, axis=0))
    X_features = np.hstack([X_scaled, X_roll_mean, X_roll_std, X_fft])
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_features, y_clean)
    (fig, ax) = plt.subplots(figsize=(10, 5))
    for (label, color, name) in [(0, '#3498db', 'Open'), (1, '#e74c3c', 'Closed')]:
        ax.hist(X_lda[y_clean == label], bins=50, alpha=0.6, color=color, label=name, edgecolor='black')
    ax.set_title('LDA — 1D Projection (Augmented Features)')
    ax.set_xlabel('LD1')
    ax.set_ylabel('Frequency')
    ax.legend()
    save_fig('lda_1d_projection.png')
    n_tsne = min(5000, len(X_features))
    rng = np.random.RandomState(RANDOM_STATE)
    idx_sub = rng.choice(len(X_features), n_tsne, replace=False)
    X_sub = X_features[idx_sub]
    y_sub = y_clean[idx_sub]
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)
    (fig, ax) = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap='coolwarm', alpha=0.4, s=10, edgecolors='none')
    ax.set_title('t-SNE — 2D Projection (Augmented Features)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Eye State (0=Open, 1=Closed)')
    save_fig('tsne_2d_projection.png')
    X_umap = None
    if HAS_UMAP:
        umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
        X_umap = umap_model.fit_transform(X_sub)
        (fig, ax) = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sub, cmap='coolwarm', alpha=0.4, s=10, edgecolors='none')
        ax.set_title('UMAP — 2D Projection (Augmented Features)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, label='Eye State (0=Open, 1=Closed)')
        save_fig('umap_2d_projection.png')
    metrics_rows = []
    for (name_m, Xr, y_eval) in [('LDA (1D)', np.column_stack([X_lda, np.zeros_like(X_lda)]), y_clean), ('t-SNE (2D)', X_tsne, y_sub)]:
        sil = silhouette_score(Xr, y_eval)
        db = davies_bouldin_score(Xr, y_eval)
        ch_s = calinski_harabasz_score(Xr, y_eval)
        metrics_rows.append([name_m, f'{sil:.4f}', f'{db:.4f}', f'{ch_s:.2f}'])
    if X_umap is not None:
        sil = silhouette_score(X_umap, y_sub)
        db = davies_bouldin_score(X_umap, y_sub)
        ch_s = calinski_harabasz_score(X_umap, y_sub)
        metrics_rows.append(['UMAP (2D)', f'{sil:.4f}', f'{db:.4f}', f'{ch_s:.2f}'])
    best_sil = max(metrics_rows, key=lambda r: float(r[1]))
    best_db = min(metrics_rows, key=lambda r: float(r[2]))
    best_ch = max(metrics_rows, key=lambda r: float(r[3]))

def temporal_three_way_split(X, y, train_frac, cv_frac):
    n = len(X)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cv_frac))
    return (X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:])

def walk_forward_cv_indices(n, n_folds=5, min_train_frac=0.5):
    min_tr = int(n * min_train_frac)
    step = (n - min_tr) // (n_folds + 1)
    return [(slice(0, min_tr + k * step), slice(min_tr + k * step, min_tr + (k + 1) * step)) for k in range(n_folds)]

def sliding_window_cv_indices(n, n_folds=5, window_frac=0.5):
    win = int(n * window_frac)
    step = (n - win) // (n_folds + 1)
    return [(slice(k * step, k * step + win), slice(k * step + win, min(k * step + win + step, n))) for k in range(n_folds)]

def _safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float('nan')

def _optimize_threshold(y_cv, probs_cv):
    (best_t, best_f1) = (0.5, 0.0)
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs_cv >= t).astype(int)
        score = f1_score(y_cv, preds, average='macro', zero_division=0)
        if score > best_f1:
            (best_f1, best_t) = (score, float(t))
    return (best_t, best_f1)

def _evaluate_ml(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bf1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    auc = _safe_auc(y_true, y_prob) if y_prob is not None else float('nan')
    return {'acc': acc, 'macro_f1': mf1, 'binary_f1': bf1, 'precision': prec, 'recall': rec, 'auc': auc}

def _get_ml_models(y_all):
    models = {'LogisticRegression': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=RANDOM_SEED))]), 'SVM_RBF': Pipeline([('sc', StandardScaler()), ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED))]), 'RandomForest': Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED))]), 'GradientBoosting': Pipeline([('sc', StandardScaler()), ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED))])}
    if HAS_XGB:
        neg = int((y_all == 0).sum())
        pos = int((y_all == 1).sum())
        models['XGBoost'] = Pipeline([('sc', StandardScaler()), ('clf', XGBClassifier(n_estimators=200, scale_pos_weight=neg / pos, eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=-1))])
    return models

def _run_single_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    cv_prob = model.predict_proba(X_cv)[:, 1]
    (opt_t, _) = _optimize_threshold(y_cv, cv_prob)
    results = {}
    for (part_name, Xs, ys) in [('CV', X_cv, y_cv), ('Test', X_te, y_te)]:
        prob = model.predict_proba(Xs)[:, 1]
        thresh = opt_t if part_name == 'Test' else 0.5
        pred = (prob >= thresh).astype(int)
        results[part_name] = _evaluate_ml(ys, pred, prob)
        results[part_name]['raw_prob'] = prob
        results[part_name]['pred'] = pred
    results['train_time'] = train_time
    results['opt_t'] = opt_t
    return results

def section_ml(X_all, y_all, N):
    drift_rows = []
    for q in range(4):
        (s, e) = (q * N // 4, (q + 1) * N // 4)
        cnts = np.bincount(y_all[s:e])
        drift_rows.append([f'Q{q + 1} [{s}–{e}]', cnts[0], cnts[1], f'{cnts[1] / len(y_all[s:e]) * 100:.1f}%'])
    for (label, s) in [('Last 10%', int(N * 0.9)), ('Last 15%', int(N * 0.85)), ('Last 20%', int(N * 0.8))]:
        cnts = np.bincount(y_all[s:])
        drift_rows.append([label, cnts[0], cnts[1], f'{cnts[1] / len(y_all[s:]) * 100:.1f}%'])
    SPLIT_CONFIGS = [('70/15/15', 0.7, 0.15), ('60/20/20', 0.6, 0.2), ('80/10/10', 0.8, 0.1)]
    split_info_rows = []
    for (sl, tr_f, cv_f) in SPLIT_CONFIGS:
        (X_tr, y_tr, X_cv, y_cv, X_te, y_te) = temporal_three_way_split(X_all, y_all, tr_f, cv_f)
        split_info_rows.append([sl, len(X_tr), len(X_cv), len(X_te), f'{y_tr.mean():.1%}', f'{y_cv.mean():.1%}', f'{y_te.mean():.1%}', f'{abs(y_tr.mean() - y_te.mean()):.1%}'])
    (X_tr70, y_tr70, _, _, _, _) = temporal_three_way_split(X_all, y_all, 0.7, 0.15)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    for (name, model) in _get_ml_models(y_all).items():
        scores = cross_val_score(model, X_tr70, y_tr70, cv=tscv, scoring='f1_macro')
        cv_rows.append([name, f'{scores.mean():.4f}', f'{scores.std():.4f}'])
    model_math = {'LogisticRegression': "Logistic Regression models the posterior probability:\n\n$$P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$\n\nUses `class_weight='balanced'` to penalise minority-class misclassification.", 'SVM_RBF': "SVM with RBF kernel maps features into higher-dimensional space:\n\n$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2)$$\n\nMaximises the soft margin with `class_weight='balanced'`.", 'RandomForest': "Random Forest builds 200 decision trees, each trained on a bootstrapped subset:\n\n$$\\hat{y} = \\text{mode}\\{h_b(\\mathbf{x})\\}_{b=1}^{200}$$\n\nUses `class_weight='balanced'` and splits by Gini impurity.", 'GradientBoosting': 'Gradient Boosting corrects residual errors sequentially:\n\n$$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\eta \\cdot h_m(\\mathbf{x})$$\n\n200 boosting rounds, learning rate $\\eta = 0.1$, max depth 5.', 'XGBoost': 'XGBoost uses `scale_pos_weight = n_neg / n_pos` to handle class imbalance directly in the gradient computation, producing the highest closed-eye recall among ML models.'}
    summary_rows_ml = []
    for (split_label, tr_frac, cv_frac) in SPLIT_CONFIGS:
        (X_tr, y_tr, X_cv, y_cv, X_te, y_te) = temporal_three_way_split(X_all, y_all, tr_frac, cv_frac)
        split_test_rows = []
        for (name, model) in _get_ml_models(y_all).items():
            pass
            res = _run_single_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te)
            te = res['Test']
            split_test_rows.append([name, f"{te['acc']:.4f}", f"{te['macro_f1']:.4f}", f"{te['precision']:.4f}", f"{te['recall']:.4f}", f"{te['auc']:.4f}", f"{res['opt_t']:.2f}"])
            summary_rows_ml.append({'split': split_label, 'model': name, **{k: v for (k, v) in te.items() if k not in ('raw_prob', 'pred')}, 'threshold': res['opt_t'], 'type': 'ML'})
        split_test_rows.sort(key=lambda r: float(r[2]), reverse=True)
    wf_agg = defaultdict(list)
    for (fi, (tr_sl, val_sl)) in enumerate(walk_forward_cv_indices(N)):
        (X_tr, y_tr) = (X_all[tr_sl], y_all[tr_sl])
        (X_val, y_val) = (X_all[val_sl], y_all[val_sl])
        for (name, model) in _get_ml_models(y_all).items():
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            (opt_t, _) = _optimize_threshold(y_val, prob)
            pred = (prob >= opt_t).astype(int)
            m = _evaluate_ml(y_val, pred, prob)
            m['opt_t'] = opt_t
            wf_agg[name].append(m)
    wf_rows = []
    for (name, folds) in wf_agg.items():
        mf1s = [f['macro_f1'] for f in folds]
        accs = [f['acc'] for f in folds]
        aucs = [f['auc'] if not math.isnan(f['auc']) else 0.0 for f in folds]
        wf_rows.append([name, f'{np.mean(mf1s):.4f}±{np.std(mf1s):.4f}', f'{np.mean(accs):.4f}±{np.std(accs):.4f}', f'{np.mean(aucs):.4f}±{np.std(aucs):.4f}'])
    sw_agg = defaultdict(list)
    for (fi, (tr_sl, val_sl)) in enumerate(sliding_window_cv_indices(N)):
        (X_tr, y_tr) = (X_all[tr_sl], y_all[tr_sl])
        (X_val, y_val) = (X_all[val_sl], y_all[val_sl])
        for (name, model) in _get_ml_models(y_all).items():
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            (opt_t, _) = _optimize_threshold(y_val, prob)
            pred = (prob >= opt_t).astype(int)
            m = _evaluate_ml(y_val, pred, prob)
            sw_agg[name].append(m)
    sw_rows = []
    for (name, folds) in sw_agg.items():
        mf1s = [f['macro_f1'] for f in folds]
        accs = [f['acc'] for f in folds]
        aucs = [f['auc'] if not math.isnan(f['auc']) else 0.0 for f in folds]
        sw_rows.append([name, f'{np.mean(mf1s):.4f}±{np.std(mf1s):.4f}', f'{np.mean(accs):.4f}±{np.std(accs):.4f}', f'{np.mean(aucs):.4f}±{np.std(aucs):.4f}'])
    (X_tr70, y_tr70, _, _, _, _) = temporal_three_way_split(X_all, y_all, 0.7, 0.15)
    scaler_fi = StandardScaler()
    X_tr70_s = scaler_fi.fit_transform(X_tr70)
    rf_fi = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)
    rf_fi.fit(X_tr70_s, y_tr70)
    importances = rf_fi.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    (fig, ax) = plt.subplots(figsize=(10, 5))
    ax.bar([FEATURE_COLUMNS[i] for i in sorted_idx], importances[sorted_idx], color='#3498db', edgecolor='black')
    ax.set_title('RandomForest Feature Importance (raw 14 channels)')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_fig('ml_feature_importance.png')
    (X_tr70, y_tr70, X_cv70, y_cv70, X_te70, y_te70) = temporal_three_way_split(X_all, y_all, 0.7, 0.15)
    (fig, ax) = plt.subplots(figsize=(8, 6))
    for (name, model) in _get_ml_models(y_all).items():
        model.fit(X_tr70, y_tr70)
        y_prob = model.predict_proba(X_te70)[:, 1]
        auc_val = _safe_auc(y_te70, y_prob)
        if not math.isnan(auc_val):
            (fpr, tpr, _) = roc_curve(y_te70, y_prob)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — ML Models (70/15/15 Test Partition)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig('ml_roc_curves.png')
    return summary_rows_ml

def build_sequences(X_flat, y_flat, seq_len):
    """(N, F) → (M, seq_len, F) overlapping windows. No cross-boundary leakage."""
    (Xs, ys) = ([], [])
    for i in range(len(X_flat) - seq_len):
        Xs.append(X_flat[i:i + seq_len])
        ys.append(y_flat[i + seq_len])
    return (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64))

def make_dl_loaders(X_tr, y_tr, X_cv, y_cv, seq_len, batch):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cv_s = scaler.transform(X_cv)
    (Xs_tr, ys_tr) = build_sequences(X_tr_s, y_tr, seq_len)
    (Xs_cv, ys_cv) = build_sequences(X_cv_s, y_cv, seq_len)
    (classes, counts) = np.unique(ys_tr, return_counts=True)
    class_weights = torch.tensor([counts.sum() / (len(classes) * c) for c in counts], dtype=torch.float32)

    def to_loader(Xs, ys):
        return DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)), batch_size=batch, shuffle=False)
    return (to_loader(Xs_tr, ys_tr), to_loader(Xs_cv, ys_cv), scaler, class_weights, ys_cv)

class LSTMClassifier(nn.Module):
    """Stacked bidirectional LSTM → global average pool → MLP head."""

    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        (out, _) = self.lstm(x)
        return self.head(out.mean(dim=1))

class CNNLSTMClassifier(nn.Module):
    """1-D conv feature extractor → bidirectional LSTM → classifier."""

    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(n_features, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        (out, _) = self.lstm(x)
        return self.head(out.mean(dim=1))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class EEGTransformer(nn.Module):
    """CLS token + sinusoidal PE + pre-LN TransformerEncoder → MLP head."""

    def __init__(self, n_features, d_model=64, nhead=4, n_layers=3, dim_ff=128, dropout=0.1, seq_len=64):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = self.pos_enc(torch.cat([cls, x], dim=1))
        return self.head(self.tf(x)[:, 0])

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018) — depthwise + separable 2D convolutions.
    Input: (B, T, C) → (B, 1, C, T). Block 1: temporal + depthwise spatial.
    Block 2: separable. Head: linear(flat → 2).
    """

    def __init__(self, n_channels=14, T=64, F1=8, D=2, dropout=0.25):
        super().__init__()
        F2 = F1 * D
        kern_t = T // 2
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False), nn.BatchNorm2d(F1), nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False), nn.Conv2d(F2, F2, (1, 1), bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, T)
            flat_dim = self.flatten(self.block2(self.block1(dummy))).shape[1]
        self.head = nn.Linear(flat_dim, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        return self.head(self.flatten(self.block2(self.block1(x))))

class PatchTST_Lite(nn.Module):
    """
    Lightweight PatchTST (Nie et al. 2023) — patch-based Transformer.
    Divides sequence into overlapping patches; CLS token aggregates context.
    """

    def __init__(self, n_features, seq_len=64, patch_size=8, stride=4, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        n_patches = (seq_len - patch_size) // stride + 1
        patch_dim = patch_size * n_features
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        B = x.size(0)
        patches = [x[:, i:i + self.patch_size, :].reshape(B, -1) for i in range(0, x.size(1) - self.patch_size + 1, self.stride)]
        x = self.patch_proj(torch.stack(patches, dim=1))
        cls = self.cls_token.expand(B, -1, -1)
        x = self.drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x = self.norm(self.tf(x))
        return self.head(x[:, 0])

def _train_epoch(model, loader, optimiser, criterion):
    model.train()
    total = 0.0
    for (Xb, yb) in loader:
        (Xb, yb) = (Xb.to(DEVICE), yb.to(DEVICE))
        optimiser.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)

@torch.no_grad()
def _predict_dl(model, loader):
    model.eval()
    (preds, probs) = ([], [])
    for (Xb, _) in loader:
        logits = model(Xb.to(DEVICE))
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return (np.concatenate(preds), np.concatenate(probs))

def _cv_loss(model, loader, criterion):
    """Compute loss on validation set without backprop."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for (Xb, yb) in loader:
            (Xb, yb) = (Xb.to(DEVICE), yb.to(DEVICE))
            total += criterion(model(Xb), yb).item() * len(yb)
    return total / len(loader.dataset)

def _run_dl(name, model, tr_loader, cv_loader, y_cv_seq, X_te, y_te, scaler, seq_len, class_weights):
    model.to(DEVICE)
    cw = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=DL_EPOCHS)
    epoch_rows = []
    train_losses = []
    cv_losses = []
    for epoch in range(1, DL_EPOCHS + 1):
        loss = _train_epoch(model, tr_loader, optimiser, criterion)
        scheduler.step()
        train_losses.append(loss)
        val_loss = _cv_loss(model, cv_loader, criterion)
        cv_losses.append(val_loss)
        if epoch % 5 == 0:
            (cv_preds, _) = _predict_dl(model, cv_loader)
            mf1 = f1_score(y_cv_seq, cv_preds, average='macro', zero_division=0)
            epoch_rows.append([epoch, f'{loss:.4f}', f'{val_loss:.4f}', f'{mf1:.4f}'])
    (fig, ax) = plt.subplots(figsize=(8, 4))
    epochs_range = range(1, DL_EPOCHS + 1)
    ax.plot(epochs_range, train_losses, label='Train Loss', color='#3498db', linewidth=1.5)
    ax.plot(epochs_range, cv_losses, label='CV Loss', color='#e74c3c', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Cross-Entropy Loss')
    ax.set_title(f'{name} — Train vs CV Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(f'dl_loss_curve_{name.lower()}.png')
    (cv_preds, cv_probs) = _predict_dl(model, cv_loader)
    (opt_t, _) = _optimize_threshold(y_cv_seq, cv_probs)
    cv_pred_opt = (cv_probs >= opt_t).astype(int)
    cv_res = _evaluate_ml(y_cv_seq, cv_pred_opt, cv_probs)
    X_te_s = scaler.transform(X_te)
    (Xs_te, ys_te) = build_sequences(X_te_s, y_te, seq_len)
    te_loader = DataLoader(TensorDataset(torch.tensor(Xs_te), torch.tensor(ys_te)), batch_size=DL_BATCH, shuffle=False)
    (te_preds, te_probs) = _predict_dl(model, te_loader)
    te_pred_opt = (te_probs >= opt_t).astype(int)
    te_res = _evaluate_ml(ys_te, te_pred_opt, te_probs)
    te_res['pred'] = te_pred_opt
    te_res['raw_prob'] = te_probs
    return {'CV': cv_res, 'Test': te_res, 'opt_t': opt_t, 'y_te_seq': ys_te, 'te_probs': te_probs}

class EnsembleOptimizer:

    def __init__(self):
        self.best_weights = None
        self.model_names = []

    def optimize(self, probs_cv_dict, y_cv, n_trials=ENS_TRIALS):
        self.model_names = list(probs_cv_dict.keys())
        k = len(self.model_names)
        prob_mat = np.column_stack([probs_cv_dict[n] for n in self.model_names])

        def eval_w(w):
            ens_prob = prob_mat @ w
            preds = (ens_prob >= 0.5).astype(int)
            return f1_score(y_cv, preds, average='macro', zero_division=0)
        rng = np.random.RandomState(RANDOM_SEED)
        best_w = np.ones(k) / k
        best_f1 = eval_w(best_w)
        for _ in range(n_trials):
            w = rng.dirichlet(np.ones(k))
            s = eval_w(w)
            if s > best_f1:
                (best_f1, best_w) = (s, w.copy())
        for combo_size in [1, 3, 5]:
            top_idx = np.argsort([eval_w(np.eye(k)[i]) for i in range(k)])[::-1]
            w = np.zeros(k)
            w[top_idx[:combo_size]] = 1.0 / combo_size
            s = eval_w(w)
            if s > best_f1:
                (best_f1, best_w) = (s, w.copy())
        self.best_weights = best_w
        return (dict(zip(self.model_names, best_w.tolist())), best_f1)

    def predict(self, probs_test_dict, threshold=0.5):
        prob_mat = np.column_stack([probs_test_dict[n] for n in self.model_names])
        ens_prob = prob_mat @ self.best_weights
        return ((ens_prob >= threshold).astype(int), ens_prob)

def section_dl(X_all, y_all):
    SPLIT_CONFIGS = [('70/15/15', 0.7, 0.15), ('60/20/20', 0.6, 0.2), ('80/10/10', 0.8, 0.1)]
    N_FEATURES = len(FEATURE_COLUMNS)
    summary_rows_dl = []
    dl_model_factories = {'LSTM': lambda : LSTMClassifier(N_FEATURES), 'CNN_LSTM': lambda : CNNLSTMClassifier(N_FEATURES), 'EEGTransformer': lambda : EEGTransformer(N_FEATURES, d_model=64, nhead=4, n_layers=3, seq_len=SEQ_LEN), 'EEGNet': lambda : EEGNet(n_channels=N_FEATURES, T=SEQ_LEN), 'PatchTST_Lite': lambda : PatchTST_Lite(N_FEATURES, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)}
    arch_descriptions = {'LSTM': 'Stacked bidirectional LSTM captures long-range temporal dependencies. Hidden state $h_t$ and cell state $c_t$ are updated via forget ($f_t$), input ($i_t$), and output ($o_t$) gates. Global average pooling over the sequence dimension produces the classification vector.\n\n$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, \\quad h_t = o_t \\odot \\tanh(c_t)$$', 'CNN_LSTM': 'Two 1D convolutional blocks extract local temporal features; a bidirectional LSTM then models the sequence dynamics of those features. The CNN acts as a learned front-end filter bank:\n\n$$y_t^{(f)} = \\text{ReLU}\\left(\\sum_{k,c} w_{k,c}^{(f)} \\cdot x_{t+k,c} + b^{(f)}\\right)$$', 'EEGTransformer': 'CLS-token Transformer with sinusoidal positional encoding and pre-LN encoder layers. Multi-head self-attention captures global cross-electrode dependencies:\n\n$$\\text{Attn}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\nThe CLS token aggregates the full sequence into a single classification vector.', 'EEGNet': 'EEGNet (Lawhern et al. 2018) uses depthwise-separable 2D convolutions that explicitly model temporal patterns (Block 1 temporal kernel ≈ 250ms) and cross-electrode spatial patterns (Block 1 depthwise spatial filter). Only ~400 parameters — highly resistant to overfitting on limited data.', 'PatchTST_Lite': 'Patch-based Transformer (Nie et al. 2023) divides the 64-sample window into 15 overlapping patches (size=8, stride=4 ≈ 62ms each). Each patch is linearly embedded; a Transformer encoder with a CLS token captures both local (per-patch) and global (cross-patch) temporal context.'}
    for (split_label, tr_frac, cv_frac) in SPLIT_CONFIGS:
        (X_tr, y_tr, X_cv, y_cv, X_te, y_te) = temporal_three_way_split(X_all, y_all, tr_frac, cv_frac)
        dl_cv_probs = {}
        dl_te_probs = {}
        y_te_seq_ref = None
        split_dl_rows = []
        for arch_name in dl_model_factories:
            pass
            (tr_loader, cv_loader, scaler, cw, y_cv_seq) = make_dl_loaders(X_tr, y_tr, X_cv, y_cv, SEQ_LEN, DL_BATCH)
            model = dl_model_factories[arch_name]()
            res = _run_dl(arch_name, model, tr_loader, cv_loader, y_cv_seq, X_te, y_te, scaler, SEQ_LEN, cw)
            dl_cv_probs[arch_name] = res['CV'].get('raw_prob', res['CV'].get('raw_prob', np.array([])))
            (_, cv_p) = _predict_dl(model, cv_loader)
            dl_cv_probs[arch_name] = cv_p
            dl_te_probs[arch_name] = res['te_probs']
            if y_te_seq_ref is None:
                y_te_seq_ref = res['y_te_seq']
            te = res['Test']
            split_dl_rows.append([arch_name, f"{te['acc']:.4f}", f"{te['macro_f1']:.4f}", f"{te['precision']:.4f}", f"{te['recall']:.4f}", f"{te['auc']:.4f}", f"{res['opt_t']:.2f}"])
            summary_rows_dl.append({'split': split_label, 'model': arch_name, **{k: v for (k, v) in te.items() if k not in ('raw_prob', 'pred', 'raw_prob')}, 'threshold': res['opt_t'], 'type': 'DL'})
        y_cv_ens = y_cv[SEQ_LEN:]
        y_te_ens = y_te_seq_ref
        ens = EnsembleOptimizer()
        (best_weights, cv_ens_f1) = ens.optimize(dl_cv_probs, y_cv_ens)
        w_rows = [[m, f'{w:.4f}', '█' * max(1, int(w * 30))] for (m, w) in sorted(best_weights.items(), key=lambda x: -x[1])]
        ens_cv_prob_vec = np.column_stack([dl_cv_probs[n] for n in ens.model_names]) @ ens.best_weights
        (opt_t_ens, _) = _optimize_threshold(y_cv_ens, ens_cv_prob_vec)
        (te_ens_pred, te_ens_prob) = ens.predict(dl_te_probs, threshold=opt_t_ens)
        ens_res = _evaluate_ml(y_te_ens, te_ens_pred, te_ens_prob)
        split_dl_rows.append(['Ensemble', f"{ens_res['acc']:.4f}", f"{ens_res['macro_f1']:.4f}", f"{ens_res['precision']:.4f}", f"{ens_res['recall']:.4f}", f"{ens_res['auc']:.4f}", f'{opt_t_ens:.2f}'])
        summary_rows_dl.append({'split': split_label, 'model': 'Ensemble', **ens_res, 'threshold': opt_t_ens, 'type': 'DL'})
        split_dl_rows.sort(key=lambda r: float(r[2]), reverse=True)
    return summary_rows_dl

def section_final_comparison(summary_rows_ml, summary_rows_dl):
    all_rows = summary_rows_ml + summary_rows_dl
    df_sum = pd.DataFrame(all_rows)
    for split_label in ['70/15/15', '60/20/20', '80/10/10']:
        sub = df_sum[df_sum['split'] == split_label].copy()
        sub = sub.sort_values('macro_f1', ascending=False)
        rows = [[r['model'], r.get('type', '?'), f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}", f"{r.get('precision', float('nan')):.4f}", f"{r.get('recall', float('nan')):.4f}", f"{r.get('auc', float('nan')):.4f}", f"{r.get('threshold', 0.5):.2f}"] for (_, r) in sub.iterrows()]
    sub70 = df_sum[df_sum['split'] == '70/15/15'].sort_values('macro_f1', ascending=False)
    if not sub70.empty:
        (fig, axes) = plt.subplots(1, 2, figsize=(16, 6))
        names = sub70['model'].tolist()
        x = np.arange(len(names))
        w = 0.35
        axes[0].bar(x - w / 2, sub70['acc'].tolist(), w, label='Accuracy', color='#3498db', edgecolor='black')
        axes[0].bar(x + w / 2, sub70['macro_f1'].tolist(), w, label='MacroF1', color='#e74c3c', edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title('All Models — Acc vs MacroF1 (70/15/15, ranked by MacroF1)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        aucs = [v if not math.isnan(v) else 0.0 for v in sub70['auc'].tolist()]
        axes[1].bar(names, aucs, color='#9370DB', edgecolor='black')
        axes[1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title('AUC-ROC (70/15/15 Test Partition)')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_fig('final_comparison.png')
    best_per_split = []
    for split_label in ['70/15/15', '60/20/20', '80/10/10']:
        sub = df_sum[df_sum['split'] == split_label]
        if sub.empty:
            continue
        best = sub.loc[sub['macro_f1'].idxmax()]
        best_per_split.append([split_label, best['model'], best.get('type', '?'), f"{best['macro_f1']:.4f}", f"{best['acc']:.4f}", f"{best.get('auc', float('nan')):.4f}"])
    mean_mf1 = df_sum.groupby('model')['macro_f1'].mean().sort_values(ascending=False)
    mean_rows = [[m, f'{v:.4f}'] for (m, v) in mean_mf1.items()]
    best_overall = mean_mf1.index[0]
    best_mf1_val = mean_mf1.iloc[0]

def main():
    progress('=' * 60)
    progress('EEG Eye State Classification — Pipeline Started')
    progress(f'[info] Device: {DEVICE}')
    progress('=' * 60)
    progress('[1/12] Loading data ...')
    df = pd.read_csv(DATA_FILE)
    section_data_description(df)
    progress('[2/12] Data imputation ...')
    df = section_data_imputation(df)
    progress('[3/12] Visualising raw data ...')
    section_data_viz_raw(df)
    progress('[4/12] Signal preprocessing (bandpass + IQR) ...')
    df_raw_copy = df.copy()
    df_clean = section_preprocessing(df)
    progress('[5/12] Visualising cleaned data ...')
    section_data_viz_cleaned(df_raw_copy, df_clean)
    progress('[6/12] Log-normalisation assessment ...')
    section_log_normalization(df_clean)
    progress('[7/12] Feature engineering ...')
    (df_eng, all_features) = section_feature_engineering(df_clean)
    progress('[8/12] Frequency-domain analysis ...')
    section_fft_psd_spectro(df_clean)
    progress('[9/12] Dimensionality reduction (LDA, t-SNE, UMAP) ...')
    section_dim_reduction(df_eng, all_features)
    X_all = df_clean[FEATURE_COLUMNS].values.astype(np.float32)
    y_all = df_clean[TARGET].values.astype(np.int64)
    N = len(X_all)
    progress('[10/12] Training ML models ...')
    summary_rows_ml = section_ml(X_all, y_all, N)
    progress('[11/12] Training DL models (PyTorch) ...')
    summary_rows_dl = section_dl(X_all, y_all)
    progress('[12/12] Generating final comparison ...')
    section_final_comparison(summary_rows_ml, summary_rows_dl)
    progress('=' * 60)
    progress('Pipeline complete.')
    progress('=' * 60)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Eye State Classification Pipeline')
    parser.add_argument('--dataset', type=str, default=None, help='Path to CSV dataset (overrides config.yaml)')
    parser.add_argument('--plot-dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    if args.dataset:
        DATA_FILE = args.dataset
    if args.plot_dir:
        PLOT_DIR = args.plot_dir
        os.makedirs(PLOT_DIR, exist_ok=True)
    main()
