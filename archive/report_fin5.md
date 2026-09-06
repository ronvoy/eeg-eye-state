
# EEG Eye State Classification — Complete Analysis Report

---

**Dataset Source:** [UCI Machine Learning Repository — EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)

---


## Table of Contents

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
    - 10.2 [Inference and Recommendation](#102-inference-and-recommendation)

---


# 1. Data Description Overview


## 1.1 Dataset Citation & Source

**Source:** [UCI Machine Learning Repository — EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)

> All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.


## 1.2 Dataset Loading

The dataset is loaded from `dataset/eeg_data_og.csv`.

| Property | Value |
| --- | --- |
| Samples | 14980 |
| Features | 14 |
| Target Column | eyeDetection |
| Sampling Rate | 128 Hz |
| Recording Duration | 117.0 seconds |


## 1.3 Variable Classification & Electrode Positions

**Numerical Variables (Continuous):** 14 EEG electrode channels recording voltage in micro-volts (µV). The Emotiv EPOC headset uses a modified 10-20 international system for electrode placement.

| Electrode | Type | 10-20 Position | Brain Region | Functional Significance |
| --- | --- | --- | --- | --- |
| AF3 | Continuous (float64) | Anterior Frontal Left | Prefrontal Cortex | Executive function, attention |
| F7 | Continuous (float64) | Frontal Left Lateral | Left Temporal-Frontal | Language processing |
| F3 | Continuous (float64) | Frontal Left | Left Frontal Lobe | Motor planning, positive affect |
| FC5 | Continuous (float64) | Fronto-Central Left | Left Motor-Frontal | Motor preparation |
| T7 | Continuous (float64) | Temporal Left | Left Temporal Lobe | Auditory processing, memory |
| P7 | Continuous (float64) | Parietal Left | Left Parietal-Temporal | Visual-spatial processing |
| O1 | Continuous (float64) | Occipital Left | Left Visual Cortex | Visual processing |
| O2 | Continuous (float64) | Occipital Right | Right Visual Cortex | Visual processing |
| P8 | Continuous (float64) | Parietal Right | Right Parietal-Temporal | Spatial attention |
| T8 | Continuous (float64) | Temporal Right | Right Temporal Lobe | Face / emotion recognition |
| FC6 | Continuous (float64) | Fronto-Central Right | Right Motor-Frontal | Motor preparation |
| F4 | Continuous (float64) | Frontal Right | Right Frontal Lobe | Motor planning, negative affect |
| F8 | Continuous (float64) | Frontal Right Lateral | Right Temporal-Frontal | Emotion, social cognition |
| AF4 | Continuous (float64) | Anterior Frontal Right | Prefrontal Cortex | Executive function, attention |

**Categorical Variable (Target):**

| Variable | Type | Values | Description |
| --- | --- | --- | --- |
| eyeDetection | Binary (int) | 0 = Open, 1 = Closed | Eye state detected via camera during recording |


## 1.4 Basic Statistics

Descriptive statistics for all 14 EEG channels (µV).

| Channel | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AF3 | 14980 | 4321.92 | 2492.07 | 1030.77 | 4280.51 | 4294.36 | 4311.79 | 309231.00 |
| F7 | 14980 | 4009.77 | 45.94 | 2830.77 | 3990.77 | 4005.64 | 4023.08 | 7804.62 |
| F3 | 14980 | 4264.02 | 44.43 | 1040.00 | 4250.26 | 4262.56 | 4270.77 | 6880.51 |
| FC5 | 14980 | 4164.95 | 5216.40 | 2453.33 | 4108.21 | 4120.51 | 4132.31 | 642564.00 |
| T7 | 14980 | 4341.74 | 34.74 | 2089.74 | 4331.79 | 4338.97 | 4347.18 | 6474.36 |
| P7 | 14980 | 4644.02 | 2924.79 | 2768.21 | 4611.79 | 4617.95 | 4626.67 | 362564.00 |
| O1 | 14980 | 4110.40 | 4600.93 | 2086.15 | 4057.95 | 4070.26 | 4083.59 | 567179.00 |
| O2 | 14980 | 4616.06 | 29.29 | 4567.18 | 4604.62 | 4613.33 | 4624.10 | 7264.10 |
| P8 | 14980 | 4218.83 | 2136.41 | 1357.95 | 4190.77 | 4199.49 | 4209.23 | 265641.00 |
| T8 | 14980 | 4231.32 | 38.05 | 1816.41 | 4220.51 | 4229.23 | 4239.49 | 6674.36 |
| FC6 | 14980 | 4202.46 | 37.79 | 3273.33 | 4190.26 | 4200.51 | 4211.28 | 6823.08 |
| F4 | 14980 | 4279.23 | 41.54 | 2257.95 | 4267.69 | 4276.92 | 4287.18 | 7002.56 |
| F8 | 14980 | 4615.21 | 1208.37 | 86.67 | 4590.77 | 4603.08 | 4617.44 | 152308.00 |
| AF4 | 14980 | 4416.44 | 5891.29 | 1366.15 | 4342.05 | 4354.87 | 4372.82 | 715897.00 |

> **Note on Spike Artifacts:** Some channels exhibit extremely large max values — orders of magnitude above the 75th percentile. These are likely **electrode spike artifacts** caused by momentary loss of contact, muscle movement, or impedance changes in the Emotiv headset. These will be addressed by outlier removal.


## 1.5 Class Distribution

Distribution of the target variable `eyeDetection` (per UCI: 0 = open, 1 = closed).

| Eye State | Count | Percentage |
| --- | --- | --- |
| Open (0) | 8257 | 55.1% |
| Closed (1) | 6723 | 44.9% |


# 2. Data Imputation

Missing values are detected and filled using column-wise **median imputation** to preserve the statistical properties of each EEG channel.

**Result:** No missing values detected across any of the 14 EEG channels. The dataset is complete.


# 3. Data Visualization (Raw Data)

Visualizations of the raw EEG data before any preprocessing.


## 3.1 Class Balance

![Class Balance](analysis-plots/class_balance_raw.png)


## 3.2 Correlation Heatmap

The correlation heatmap reveals linear relationships between EEG channels. Computed on data **winsorized at the 1st–99th percentile** to suppress spike-artifact-driven artificial correlations.

![Correlation Heatmap (Raw)](analysis-plots/corr_heatmap_raw.png)


## 3.3 Box Plots

Box plots highlight potential outliers beyond the 1.5x IQR whiskers.

![Box Plots (Raw)](analysis-plots/boxplots_raw.png)

![Box Plots Zoomed (Raw)](analysis-plots/boxplots_raw_zoomed.png)


## 3.4 Histograms

Amplitude distributions per channel split by eye state.

![Histograms (Raw)](analysis-plots/histograms_raw.png)


# 4. Signal Preprocessing

EEG signals contain artifacts from eye blinks, muscle movement, and electrode drift that must be removed before analysis. This section applies a two-stage cleaning pipeline in the **correct causal order**:

1. **IQR spike removal first** — raw hardware spike artifacts (up to 715,897 µV) are removed *before* filtering. Applying `filtfilt` to spikes first smears them to neighbouring samples via the backward pass, inflating data loss from ~9% to ~19%.

2. **Bandpass filter (0.5–45 Hz) second** — applied to the already spike-free signal so no artifact energy is convolved into the physiological EEG bands.


## 4.1 IQR Spike Removal (applied first, before filtering)

A **light IQR filter** (3.0x IQR, max 3 passes) removes hardware spike artifacts from the **raw** signal. Applying this step *before* filtering is critical: `filtfilt` convolves forward then backward, so a single spike at sample $t$ would contaminate samples $t - N$ through $t + N$ after filtering.

| Channel | Lower Bound (µV) | Upper Bound (µV) |
| --- | --- | --- |
| AF3 | 4186.67 | 4405.63 |
| F7 | 3897.95 | 4113.34 |
| F3 | 4193.35 | 4326.14 |
| FC5 | 4047.69 | 4187.69 |
| T7 | 4288.71 | 4389.23 |
| P7 | 4570.24 | 4667.19 |
| O1 | 3982.08 | 4157.92 |
| O2 | 4549.24 | 4678.46 |
| P8 | 4138.45 | 4260.53 |
| T8 | 4164.62 | 4293.84 |
| FC6 | 4127.70 | 4271.27 |
| F4 | 4213.33 | 4338.98 |
| F8 | 4517.41 | 4686.18 |
| AF4 | 4260.00 | 4450.26 |

| Metric | Value |
| --- | --- |
| Original samples | 14980 |
| After IQR removal | 13606 |
| Spike samples removed | 1374 |
| Removal % | 9.2% |
| IQR passes | 3 |
| IQR multiplier | 3.0x |


## 4.2 Bandpass Filter (0.5–45 Hz) — applied after spike removal

A 4th-order Butterworth bandpass filter (0.5–45.0 Hz) removes DC drift and high-frequency noise while preserving physiologically relevant EEG bands (Delta through Gamma). Applied via `scipy.signal.filtfilt` (zero-phase, forward-backward filtering) to avoid phase distortion.

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega^2 - \omega_0^2}{\omega_c}\right)^{2N}}$$

where $\omega_0 = \sqrt{\omega_L \cdot \omega_H}$ and $N$ = 4 is the filter order.

![Bandpass Filter Effect](analysis-plots/bandpass_effect.png)

| Metric | Value |
| --- | --- |
| Original samples | 14980 |
| After IQR spike removal | 13606 |
| After bandpass filter | 13606 |
| Total removed | 1374 |
| Total removal % | 9.2% |
| Bandpass range | 0.5–45.0 Hz |
| Filter order | 4 |

> **Preprocessing Summary (corrected order):** IQR spike removal (3.0×, 9.2% removed) → Bandpass filter (0.5–45.0 Hz). Total retained: **13606 / 14980 samples (90.8%)**.


# 5. Data Visualization (After Preprocessing)

Comparison of distributions before and after preprocessing.


## 5.1 Corrected Correlation Heatmap (after preprocessing)

With spike artifacts removed, the correlation heatmap now reflects true physiological relationships between EEG channels.

![Correlation Heatmap (Cleaned)](analysis-plots/corr_heatmap_cleaned.png)


## 5.2 Box Plots Comparison

Side-by-side box plots confirm preprocessing effectiveness. Whiskers are set to 3.0x IQR to match the cleaning threshold.

![Box Plots (Cleaned)](analysis-plots/boxplots_cleaned.png)


## 5.3 Histograms After Cleaning

![Histograms (Cleaned)](analysis-plots/histograms_cleaned.png)


# 6. PSD and Spectrogram Analysis

Frequency-domain analysis reveals the power distribution across brain wave bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** (8-12 Hz), **Beta** (12-30 Hz), and **Gamma** (30-45 Hz). Alpha power increases when eyes are closed (the **Berger effect**).


## 6.1 Power Spectral Density (PSD)

Welch's method estimates the PSD for each channel using segment averaging (Hann window, `nperseg=256`, 50% overlap). Shaded regions indicate standard EEG frequency bands.

![PSD Analysis](analysis-plots/psd_analysis.png)

**PSD Interpretation — Berger Effect:** Alpha-band power (8–12 Hz) increases when the eyes are closed, particularly in occipital electrodes (O1, O2). If the red curve (closed) shows higher power in the alpha band compared to blue (open), this confirms the dataset captures genuine physiological differences between eye states.


## 6.2 Spectrogram Analysis

Spectrograms show the time-frequency power distribution. Horizontal dashed lines mark band boundaries.

![Spectrograms Eyes Open](analysis-plots/spectrograms_open.png)

![Spectrograms Eyes Closed](analysis-plots/spectrograms_closed.png)


# 7. Dimensionality Reduction (LDA)

LDA (Linear Discriminant Analysis) maximises the ratio of between-class to within-class variance, yielding the optimal single linear discriminant for binary classification. Applied to the raw 14-channel feature space.

![LDA 1D Projection](analysis-plots/lda_1d_projection.png)

| Metric | Value | Interpretation |
| --- | --- | --- |
| Silhouette Score | 0.0022 | Higher = better separation (max 1.0) |
| Davies-Bouldin Index | 7.8939 | Lower = better separation |
| Calinski-Harabasz Score | 121.56 | Higher = better separation |

**Interpretation:** LDA silhouette of 0.002 confirms that eye states are not trivially separable in the raw amplitude space — classification requires either temporal context (DL sequence models) or frequency-domain features. This motivates the use of sequence-based DL architectures like EEGNet and CNN-LSTM.


# 8. Machine Learning Classification

The ML pipeline uses **raw 14 EEG channels** with temporal (chronological) splits to evaluate classification under realistic deployment conditions. Key design choices:

- **No shuffling**: all splits are chronological to prevent data leakage
- **Class weighting**: `class_weight='balanced'` (LogReg, RF) and `scale_pos_weight` (XGBoost) compensate for temporal class drift
- **Threshold optimization**: CV-optimised decision threshold applied consistently to both CV and test partitions
- **Primary metric: Macro-F1** — equally weights both eye states under distribution shift

**Models selected:** LogisticRegression (well-calibrated baseline), RandomForest (robust nonlinear), XGBoost (best gradient boosting with native imbalance handling). GradientBoosting and SVM are excluded — GB lacks `class_weight` support, and SVM is prohibitively slow with no accuracy advantage.


## 8.1 Temporal Concept Drift Diagnosis

The subject's eye-state distribution changes dramatically over the recording. Every hold-out split places the test window in the heavily open-dominant tail.

| Segment | Open | Closed | % Closed |
| --- | --- | --- | --- |
| Q1 [0–3401] | 1707 | 1694 | 49.8% |
| Q2 [3401–6803] | 1374 | 2028 | 59.6% |
| Q3 [6803–10204] | 1780 | 1621 | 47.7% |
| Q4 [10204–13606] | 2579 | 823 | 24.2% |
| Last 10% | 1309 | 52 | 3.8% |
| Last 15% | 1937 | 104 | 5.1% |
| Last 20% | 2579 | 143 | 5.3% |

> **Warning:** The last 15% of the recording is only ~8% closed-eye. Models trained on balanced data (~50% closed) face a ~45% distribution shift. Accuracy is misleading — **Macro-F1 is the honest metric**.


## 8.2 Split Configurations

| Split | Train N | CV N | Test N | Train Closed% | CV Closed% | Test Closed% | Δ Shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 70/15/15 | 9524 | 2041 | 2041 | 56.0% | 35.9% | 5.1% | 50.9% |
| 60/20/20 | 8163 | 2721 | 2722 | 62.3% | 34.6% | 5.3% | 57.0% |
| 80/10/10 | 10884 | 1361 | 1361 | 55.3% | 6.7% | 3.8% | 51.5% |


## 8.3 Cross-Validation Results (5-Fold TimeSeriesSplit)

5-fold time-series CV on the 70/15 training portion. Each fold trains on all preceding data, respecting temporal order.

| Model | CV Macro-F1 Mean | CV Macro-F1 Std |
| --- | --- | --- |
| LogisticRegression | 0.4653 | 0.0705 |
| RandomForest | 0.4222 | 0.0518 |
| XGBoost | 0.4511 | 0.0617 |


## 8.4 Hold-Out Split Results


### Split 70/15/15

Train=9524 (56.0% closed) | CV=2041 (35.9% closed) | Test=2041 (5.1% closed) | Δ shift=50.9%

**LogisticRegression:** Logistic Regression models the posterior probability:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Uses `class_weight='balanced'` to penalise minority-class misclassification.

Acc=0.7423 | MacroF1=0.4540 | BinaryF1=0.0573 | AUC=0.3627 | Threshold=0.53 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1499 | 438 |
| True Closed | 88 | 16 |

TP=16  FP=438  FN=88  TN=1499

**RandomForest:** Random Forest builds 200 decision trees, each trained on a bootstrapped subset:

$$\hat{y} = \text{mode}\{h_b(\mathbf{x})\}_{b=1}^{200}$$

Uses `class_weight='balanced'` and splits by Gini impurity.

Acc=0.6164 | MacroF1=0.4009 | BinaryF1=0.0416 | AUC=0.3984 | Threshold=0.61 | TrainTime=2.2s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1241 | 696 |
| True Closed | 87 | 17 |

TP=17  FP=696  FN=87  TN=1241

**XGBoost:** XGBoost uses `scale_pos_weight = n_neg / n_pos` (computed from **training data only**) to handle class imbalance directly in the gradient computation.

Acc=0.6629 | MacroF1=0.4310 | BinaryF1=0.0678 | AUC=0.4155 | Threshold=0.72 | TrainTime=0.6s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1328 | 609 |
| True Closed | 79 | 25 |

TP=25  FP=609  FN=79  TN=1328

**70/15/15 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.7423 | 0.4540 | 0.4899 | 0.4639 | 0.3627 | 0.53 |
| XGBoost | 0.6629 | 0.4310 | 0.4916 | 0.4630 | 0.4155 | 0.72 |
| RandomForest | 0.6164 | 0.4009 | 0.4792 | 0.4021 | 0.3984 | 0.61 |


### Split 60/20/20

Train=8163 (62.3% closed) | CV=2721 (34.6% closed) | Test=2722 (5.3% closed) | Δ shift=57.0%

Acc=0.7439 | MacroF1=0.4812 | BinaryF1=0.1121 | AUC=0.4831 | Threshold=0.54 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1981 | 598 |
| True Closed | 99 | 44 |

TP=44  FP=598  FN=99  TN=1981

Acc=0.6323 | MacroF1=0.4271 | BinaryF1=0.0842 | AUC=0.4515 | Threshold=0.68 | TrainTime=1.7s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1675 | 904 |
| True Closed | 97 | 46 |

TP=46  FP=904  FN=97  TN=1675

Acc=0.6462 | MacroF1=0.4355 | BinaryF1=0.0907 | AUC=0.4896 | Threshold=0.79 | TrainTime=0.6s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1711 | 868 |
| True Closed | 95 | 48 |

TP=48  FP=868  FN=95  TN=1711

**60/20/20 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.7439 | 0.4812 | 0.5105 | 0.5379 | 0.4831 | 0.54 |
| XGBoost | 0.6462 | 0.4355 | 0.4999 | 0.4995 | 0.4896 | 0.79 |
| RandomForest | 0.6323 | 0.4271 | 0.4968 | 0.4856 | 0.4515 | 0.68 |


### Split 80/10/10

Train=10884 (55.3% closed) | CV=1361 (6.7% closed) | Test=1361 (3.8% closed) | Δ shift=51.5%

Acc=0.8663 | MacroF1=0.4642 | BinaryF1=0.0000 | AUC=0.2041 | Threshold=0.56 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1179 | 130 |
| True Closed | 52 | 0 |

TP=0  FP=130  FN=52  TN=1179

Acc=0.9030 | MacroF1=0.5030 | BinaryF1=0.0571 | AUC=0.3742 | Threshold=0.73 | TrainTime=2.6s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1225 | 84 |
| True Closed | 48 | 4 |

TP=4  FP=84  FN=48  TN=1225

Acc=0.9295 | MacroF1=0.5201 | BinaryF1=0.0769 | AUC=0.4156 | Threshold=0.95 | TrainTime=0.7s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1261 | 48 |
| True Closed | 48 | 4 |

TP=4  FP=48  FN=48  TN=1261

**80/10/10 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9295 | 0.5201 | 0.5201 | 0.5201 | 0.4156 | 0.95 |
| RandomForest | 0.9030 | 0.5030 | 0.5039 | 0.5064 | 0.3742 | 0.73 |
| LogisticRegression | 0.8663 | 0.4642 | 0.4789 | 0.4503 | 0.2041 | 0.56 |


## 8.5 Walk-Forward CV (Expanding Window) — 5 Folds

Expanding-window walk-forward CV simulates real deployment: the model always trains on all available past data before predicting the next window. A fixed threshold of 0.5 is used for unbiased evaluation.

Fold 1 — train=6803 | val=1133 | val_closed=100.00%

  LogisticRegression: Acc=0.5128 MacroF1=0.3390 AUC=nan

  RandomForest: Acc=0.6628 MacroF1=0.3986 AUC=nan

  XGBoost: Acc=0.5631 MacroF1=0.3602 AUC=nan

Fold 2 — train=7936 | val=1133 | val_closed=41.92%

  LogisticRegression: Acc=0.5022 MacroF1=0.5004 AUC=0.5059

  RandomForest: Acc=0.4978 MacroF1=0.4685 AUC=0.5905

  XGBoost: Acc=0.5234 MacroF1=0.5229 AUC=0.5772

Fold 3 — train=9069 | val=1133 | val_closed=0.97%

  LogisticRegression: Acc=0.5199 MacroF1=0.3594 AUC=0.9927

  RandomForest: Acc=0.2577 MacroF1=0.2130 AUC=0.9952

  XGBoost: Acc=0.4228 MacroF1=0.3106 AUC=0.9843

Fold 4 — train=10202 | val=1133 | val_closed=60.19%

  LogisticRegression: Acc=0.4810 MacroF1=0.4711 AUC=0.4801

  RandomForest: Acc=0.5375 MacroF1=0.5128 AUC=0.4911

  XGBoost: Acc=0.5172 MacroF1=0.5023 AUC=0.5057

Fold 5 — train=11335 | val=1133 | val_closed=8.03%

  LogisticRegression: Acc=0.5322 MacroF1=0.4294 AUC=0.6395

  RandomForest: Acc=0.3883 MacroF1=0.3353 AUC=0.5356

  XGBoost: Acc=0.4484 MacroF1=0.3671 AUC=0.4961

**Walk-Forward CV — Mean ± Std (primary: Macro-F1):**

| Model | MacroF1 Mean±Std | Acc Mean±Std | AUC Mean±Std |
| --- | --- | --- | --- |
| LogisticRegression | 0.4199±0.0623 | 0.5096±0.0173 | 0.5236±0.3193 |
| RandomForest | 0.3857±0.1054 | 0.4688±0.1373 | 0.5225±0.3169 |
| XGBoost | 0.4126±0.0842 | 0.4950±0.0516 | 0.5127±0.3130 |

**Feature Importance (RandomForest — 70/15/15 training partition):**

![Feature Importance](analysis-plots/ml_feature_importance.png)

![ML ROC Curves](analysis-plots/ml_roc_curves.png)


# 9. Deep Learning Classification

All DL models use PyTorch with: **(1) weighted CrossEntropyLoss** (inverse class frequency), **(2) AdamW + CosineAnnealingLR**, **(3) CV-optimised decision threshold**, and **(4) Macro-F1 as primary metric**. Sequences are built per partition with no cross-boundary leakage. Label = last sample in window (not look-ahead).

**Models selected:** LSTM (temporal baseline), CNN-LSTM (local+temporal), EEGNet (EEG-specific, ~1.1K params). EEGTransformer and PatchTST are excluded — they suffer mode collapse on this dataset size (~14K samples).


## 9.0 Architecture Overview & Training Setup

**Weighted Cross-Entropy Loss:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \log\left(\frac{e^{z_{y_i}}}{\sum_{c=0}^{1} e^{z_c}}\right)$$

where $w_c = \frac{N}{2 \cdot N_c}$ is the per-class weight. **Sequence length:** SEQ_LEN=64 samples (≈500ms at 128 Hz). **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. **Scheduler:** CosineAnnealingLR over 25 epochs.

| Model | Architecture | Parameters | Key Innovation |
| --- | --- | --- | --- |
| LSTM | BiLSTM(128)×2 → AvgPool → MLP | ~200K | Long-range temporal dependencies |
| CNN-LSTM | Conv1D(64,128) → BiLSTM(64) → MLP | ~150K | Local feature extraction + sequence memory |
| EEGNet | Depthwise Conv2D blocks → Linear | ~1.1K | Electrode-aware, compact, best calibrated |


### Split 70/15/15

Train=9524 (56.0% closed) | CV=2041 (35.9% closed) | Test=2041 (5.1% closed)


## 9.1 LSTM

Stacked bidirectional LSTM captures long-range temporal dependencies.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)$$

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0221 | 3.9437 | 0.5855 |
| 10 | 0.0064 | 5.0002 | 0.5784 |
| 15 | 0.0024 | 5.8041 | 0.5843 |
| 20 | 0.0010 | 6.0942 | 0.5909 |
| 25 | 0.0007 | 6.3217 | 0.5914 |

![LSTM Loss Curve](analysis-plots/dl_loss_curve_lstm.png)

Optimal threshold (CV-optimised): **0.92**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.5961 | 0.5938 | 0.5636 | 0.6119 | 0.6185 | 0.6411 |
| Test | 0.4646 | 0.3675 | 0.1197 | 0.5146 | 0.5721 | 0.6022 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 847 | 1027 |
| True Closed | 32 | 72 |

TP=72  FP=1027  FN=32  TN=847


## 9.2 CNN_LSTM

Two 1D convolutional blocks extract local temporal features; a bidirectional LSTM then models sequence dynamics.

$$y_t^{(f)} = \text{ReLU}\left(\sum_{k,c} w_{k,c}^{(f)} \cdot x_{t+k,c} + b^{(f)}\right)$$

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0111 | 2.2347 | 0.5436 |
| 10 | 0.0034 | 2.7421 | 0.5680 |
| 15 | 0.0011 | 3.5244 | 0.5410 |
| 20 | 0.0002 | 3.6680 | 0.5513 |
| 25 | 0.0002 | 3.6650 | 0.5466 |

![CNN_LSTM Loss Curve](analysis-plots/dl_loss_curve_cnn_lstm.png)

Optimal threshold (CV-optimised): **0.81**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.5556 | 0.5528 | 0.5173 | 0.5695 | 0.5737 | 0.5824 |
| Test | 0.4135 | 0.3307 | 0.0952 | 0.4990 | 0.4952 | 0.4395 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 757 | 1117 |
| True Closed | 43 | 61 |

TP=61  FP=1117  FN=43  TN=757


## 9.3 EEGNet

EEGNet (Lawhern et al. 2018) uses depthwise-separable 2D convolutions that explicitly model temporal patterns (Block 1 temporal kernel ≈ 250ms) and cross-electrode spatial patterns. Only ~1.1K parameters — highly resistant to overfitting on limited data.

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.5403 | 0.7493 | 0.5623 |
| 10 | 0.3925 | 0.9283 | 0.5459 |
| 15 | 0.3396 | 0.9787 | 0.5637 |
| 20 | 0.3081 | 0.9940 | 0.5540 |
| 25 | 0.2908 | 0.9756 | 0.5521 |

![EEGNet Loss Curve](analysis-plots/dl_loss_curve_eegnet.png)

Optimal threshold (CV-optimised): **0.83**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.6587 | 0.6193 | 0.4966 | 0.6277 | 0.6167 | 0.6683 |
| Test | 0.6512 | 0.4533 | 0.1244 | 0.5146 | 0.5662 | 0.5854 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1239 | 635 |
| True Closed | 55 | 49 |

TP=49  FP=635  FN=55  TN=1239


## 9.4 Soft-Vote Ensemble

**Soft-Vote Ensemble — 70/15/15**

Random-weight Dirichlet search with threshold co-optimization.

Optimal weights (CV Macro-F1 = 0.6422, threshold = 0.80):

| Model | Weight | Contribution |
| --- | --- | --- |
| LSTM | 0.4568 | █████████████████████████ |
| EEGNet | 0.3624 | ███████████████████ |
| CNN_LSTM | 0.1808 | █████████ |

Ensemble Test (t=0.80): Acc=0.6173 | MacroF1=0.4372 | AUC=0.5886

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1170 | 704 |
| True Closed | 53 | 51 |

TP=51  FP=704  FN=53  TN=1170


## 9.5 DL Model Comparison

**DL Model Comparison — 70/15/15:**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| EEGNet | 0.6512 | 0.4533 | 0.5146 | 0.5662 | 0.5854 | 0.83 |
| Ensemble | 0.6173 | 0.4372 | 0.5121 | 0.5574 | 0.5886 | 0.80 |
| LSTM | 0.4646 | 0.3675 | 0.5146 | 0.5721 | 0.6022 | 0.92 |
| CNN_LSTM | 0.4135 | 0.3307 | 0.4990 | 0.4952 | 0.4395 | 0.81 |


### Split 60/20/20

Train=8163 (62.3% closed) | CV=2721 (34.6% closed) | Test=2722 (5.3% closed)

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0261 | 1.8116 | 0.6345 |
| 10 | 0.0072 | 3.6008 | 0.6175 |
| 15 | 0.0029 | 3.2524 | 0.6387 |
| 20 | 0.0011 | 3.7134 | 0.6325 |
| 25 | 0.0008 | 3.7732 | 0.6347 |

![LSTM Loss Curve](analysis-plots/dl_loss_curve_lstm.png)

Optimal threshold (CV-optimised): **0.94**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.6531 | 0.6448 | 0.5902 | 0.6588 | 0.6793 | 0.7633 |
| Test | 0.3780 | 0.3212 | 0.1249 | 0.5201 | 0.5889 | 0.6074 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 887 | 1629 |
| True Closed | 25 | 118 |

TP=118  FP=1629  FN=25  TN=887

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0107 | 1.8023 | 0.6240 |
| 10 | 0.0031 | 2.7045 | 0.6038 |
| 15 | 0.0013 | 3.0827 | 0.6192 |
| 20 | 0.0016 | 3.6119 | 0.5721 |
| 25 | 0.0003 | 3.3869 | 0.5714 |

![CNN_LSTM Loss Curve](analysis-plots/dl_loss_curve_cnn_lstm.png)

Optimal threshold (CV-optimised): **0.93**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.5959 | 0.5920 | 0.5521 | 0.6229 | 0.6360 | 0.6516 |
| Test | 0.4069 | 0.3358 | 0.1185 | 0.5139 | 0.5646 | 0.6953 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 976 | 1540 |
| True Closed | 37 | 106 |

TP=106  FP=1540  FN=37  TN=976

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.5505 | 0.6499 | 0.6140 |
| 10 | 0.4173 | 0.6715 | 0.6664 |
| 15 | 0.3698 | 0.6769 | 0.6637 |
| 20 | 0.3454 | 0.6906 | 0.6578 |
| 25 | 0.3315 | 0.6902 | 0.6600 |

![EEGNet Loss Curve](analysis-plots/dl_loss_curve_eegnet.png)

Optimal threshold (CV-optimised): **0.84**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.7705 | 0.7283 | 0.6211 | 0.7434 | 0.7196 | 0.7763 |
| Test | 0.7796 | 0.4892 | 0.1040 | 0.5079 | 0.5241 | 0.5911 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 2039 | 477 |
| True Closed | 109 | 34 |

TP=34  FP=477  FN=109  TN=2039

**Soft-Vote Ensemble — 60/20/20**

Random-weight Dirichlet search with threshold co-optimization.

Optimal weights (CV Macro-F1 = 0.7447, threshold = 0.90):

| Model | Weight | Contribution |
| --- | --- | --- |
| LSTM | 0.5509 | █████████████████████████ |
| EEGNet | 0.4476 | ████████████████████ |
| CNN_LSTM | 0.0015 | █ |

Ensemble Test (t=0.90): Acc=0.7623 | MacroF1=0.5127 | AUC=0.6198

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1965 | 551 |
| True Closed | 81 | 62 |

TP=62  FP=551  FN=81  TN=1965

**DL Model Comparison — 60/20/20:**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| Ensemble | 0.7623 | 0.5127 | 0.5308 | 0.6073 | 0.6198 | 0.90 |
| EEGNet | 0.7796 | 0.4892 | 0.5079 | 0.5241 | 0.5911 | 0.84 |
| CNN_LSTM | 0.4069 | 0.3358 | 0.5139 | 0.5646 | 0.6953 | 0.93 |
| LSTM | 0.3780 | 0.3212 | 0.5201 | 0.5889 | 0.6074 | 0.94 |


### Split 80/10/10

Train=10884 (55.3% closed) | CV=1361 (6.7% closed) | Test=1361 (3.8% closed)

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0182 | 4.8211 | 0.3408 |
| 10 | 0.0051 | 7.6892 | 0.3121 |
| 15 | 0.0014 | 8.1952 | 0.3795 |
| 20 | 0.0008 | 8.5202 | 0.3777 |
| 25 | 0.0005 | 8.7569 | 0.3845 |

![LSTM Loss Curve](analysis-plots/dl_loss_curve_lstm.png)

Optimal threshold (CV-optimised): **0.91**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.4923 | 0.3983 | 0.1605 | 0.5222 | 0.5848 | 0.5170 |
| Test | 0.4006 | 0.3145 | 0.0716 | 0.4976 | 0.4851 | 0.4450 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 490 | 756 |
| True Closed | 22 | 30 |

TP=30  FP=756  FN=22  TN=490

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.0112 | 4.4888 | 0.2730 |
| 10 | 0.0027 | 4.0370 | 0.3224 |
| 15 | 0.0009 | 3.9388 | 0.3279 |
| 20 | 0.0003 | 4.7959 | 0.3279 |
| 25 | 0.0002 | 4.9617 | 0.3180 |

![CNN_LSTM Loss Curve](analysis-plots/dl_loss_curve_cnn_lstm.png)

Optimal threshold (CV-optimised): **0.95**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.4499 | 0.3763 | 0.1620 | 0.5249 | 0.5925 | 0.6341 |
| Test | 0.3582 | 0.2927 | 0.0775 | 0.5015 | 0.5091 | 0.5539 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 430 | 816 |
| True Closed | 17 | 35 |

TP=35  FP=816  FN=17  TN=430

| Epoch | Train Loss | CV Loss | CV Macro-F1 |
| --- | --- | --- | --- |
| 5 | 0.5475 | 0.8384 | 0.3985 |
| 10 | 0.4070 | 1.1503 | 0.4084 |
| 15 | 0.3532 | 1.2750 | 0.3859 |
| 20 | 0.3278 | 1.3677 | 0.3899 |
| 25 | 0.3168 | 1.3721 | 0.3815 |

![EEGNet Loss Curve](analysis-plots/dl_loss_curve_eegnet.png)

Optimal threshold (CV-optimised): **0.94**

| Partition | Acc | MacroF1 | BinaryF1 | Prec(M) | Rec(M) | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CV | 0.7627 | 0.4914 | 0.1200 | 0.5069 | 0.5168 | 0.5667 |
| Test | 0.8529 | 0.4603 | 0.0000 | 0.4776 | 0.4442 | 0.6564 |

**Test Confusion Matrix:**

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1107 | 139 |
| True Closed | 52 | 0 |

TP=0  FP=139  FN=52  TN=1107

**Soft-Vote Ensemble — 80/10/10**

Random-weight Dirichlet search with threshold co-optimization.

Optimal weights (CV Macro-F1 = 0.5428, threshold = 0.90):

| Model | Weight | Contribution |
| --- | --- | --- |
| CNN_LSTM | 0.7010 | █████████████████████████ |
| EEGNet | 0.1802 | ██████ |
| LSTM | 0.1188 | ████ |

Ensemble Test (t=0.90): Acc=0.6441 | MacroF1=0.4396 | AUC=0.5764

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 810 | 436 |
| True Closed | 26 | 26 |

TP=26  FP=436  FN=26  TN=810

**DL Model Comparison — 80/10/10:**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| EEGNet | 0.8529 | 0.4603 | 0.4776 | 0.4442 | 0.6564 | 0.94 |
| Ensemble | 0.6441 | 0.4396 | 0.5126 | 0.5750 | 0.5764 | 0.90 |
| LSTM | 0.4006 | 0.3145 | 0.4976 | 0.4851 | 0.4450 | 0.91 |
| CNN_LSTM | 0.3582 | 0.2927 | 0.5015 | 0.5091 | 0.5539 | 0.95 |


# 10. Final Comparison and Inference

This section unifies all models across all evaluation protocols. Primary metric: **Macro-F1**.


## 10.1 Unified Model Comparison

All test-partition results across all hold-out splits, sorted by Macro-F1.


### Split 70/15/15

| Model | Type | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | ML | 0.7423 | 0.4540 | 0.4899 | 0.4639 | 0.3627 | 0.53 |
| EEGNet | DL | 0.6512 | 0.4533 | 0.5146 | 0.5662 | 0.5854 | 0.83 |
| Ensemble | DL | 0.6173 | 0.4372 | 0.5121 | 0.5574 | 0.5886 | 0.80 |
| XGBoost | ML | 0.6629 | 0.4310 | 0.4916 | 0.4630 | 0.4155 | 0.72 |
| RandomForest | ML | 0.6164 | 0.4009 | 0.4792 | 0.4021 | 0.3984 | 0.61 |
| LSTM | DL | 0.4646 | 0.3675 | 0.5146 | 0.5721 | 0.6022 | 0.92 |
| CNN_LSTM | DL | 0.4135 | 0.3307 | 0.4990 | 0.4952 | 0.4395 | 0.81 |


### Split 60/20/20

| Model | Type | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ensemble | DL | 0.7623 | 0.5127 | 0.5308 | 0.6073 | 0.6198 | 0.90 |
| EEGNet | DL | 0.7796 | 0.4892 | 0.5079 | 0.5241 | 0.5911 | 0.84 |
| LogisticRegression | ML | 0.7439 | 0.4812 | 0.5105 | 0.5379 | 0.4831 | 0.54 |
| XGBoost | ML | 0.6462 | 0.4355 | 0.4999 | 0.4995 | 0.4896 | 0.79 |
| RandomForest | ML | 0.6323 | 0.4271 | 0.4968 | 0.4856 | 0.4515 | 0.68 |
| CNN_LSTM | DL | 0.4069 | 0.3358 | 0.5139 | 0.5646 | 0.6953 | 0.93 |
| LSTM | DL | 0.3780 | 0.3212 | 0.5201 | 0.5889 | 0.6074 | 0.94 |


### Split 80/10/10

| Model | Type | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | ML | 0.9295 | 0.5201 | 0.5201 | 0.5201 | 0.4156 | 0.95 |
| RandomForest | ML | 0.9030 | 0.5030 | 0.5039 | 0.5064 | 0.3742 | 0.73 |
| LogisticRegression | ML | 0.8663 | 0.4642 | 0.4789 | 0.4503 | 0.2041 | 0.56 |
| EEGNet | DL | 0.8529 | 0.4603 | 0.4776 | 0.4442 | 0.6564 | 0.94 |
| Ensemble | DL | 0.6441 | 0.4396 | 0.5126 | 0.5750 | 0.5764 | 0.90 |
| LSTM | DL | 0.4006 | 0.3145 | 0.4976 | 0.4851 | 0.4450 | 0.91 |
| CNN_LSTM | DL | 0.3582 | 0.2927 | 0.5015 | 0.5091 | 0.5539 | 0.95 |

![Final Model Comparison](analysis-plots/final_comparison.png)


## 10.2 Inference and Recommendation

**Best model per hold-out split (by Macro-F1):**

| Split | Best Model | Type | MacroF1 | Acc | AUC |
| --- | --- | --- | --- | --- | --- |
| 70/15/15 | LogisticRegression | ML | 0.4540 | 0.7423 | 0.3627 |
| 60/20/20 | Ensemble | DL | 0.5127 | 0.7623 | 0.6198 |
| 80/10/10 | XGBoost | ML | 0.5201 | 0.9295 | 0.4156 |

**Mean Macro-F1 across all three splits (stability ranking):**

| Model | Mean MacroF1 |
| --- | --- |
| EEGNet | 0.4676 |
| LogisticRegression | 0.4665 |
| Ensemble | 0.4632 |
| XGBoost | 0.4622 |
| RandomForest | 0.4437 |
| LSTM | 0.3344 |
| CNN_LSTM | 0.3197 |

### Best Overall Model: **EEGNet**

Based on mean Macro-F1 across all three temporal hold-out splits, **EEGNet** achieves the highest average score of **0.4676**.

**Key Observations:**

- The last 15% of the recording is ~8% closed-eye, creating a ~45% distribution shift between training and test. This is the root cause of all metric paradoxes.
- Models with well-calibrated probabilities (LogReg, EEGNet) transfer thresholds across the distribution shift more reliably than uncalibrated models.
- All models struggle under severe concept drift; Macro-F1 values near 0.50 indicate performance only marginally above the balanced-accuracy baseline.

**Recommended Model Per Use Case:**

| Use Case | Model | Reason |
| --- | --- | --- |
| Balanced accuracy (research) | EEGNet | Highest mean Macro-F1 (0.4676) |
| Stable production ML | LogisticRegression | Most consistent ML model across splits, fast inference |
| Online/streaming BCI | EEGNet | ~1.1K params, fast inference, electrode-aware architecture |

---


### Appendix: Dataset Suitability for Neural Network Training

| Criterion | Verdict | Explanation |
| --- | --- | --- |
| Sample size | ⚠ Marginal | ~14 k total; DL typically needs >50 k sequences |
| Single subject | ✗ Poor generalisation | All 14,980 samples from one 117-second session |
| Temporal continuity | ⚠ Concept drift | Eye-state ratio shifts from 50% to 6% closed over recording |
| Preprocessing | ✓ Bandpass + IQR | Bandpass 0.5–45 Hz + IQR cleaning preserves EEG integrity |
| Class balance | ✓ Adequate globally | 55% open / 45% closed globally; drifts at end |
| Label quality | ✓ Camera-verified | Eye state labels added by manual video annotation |

---

