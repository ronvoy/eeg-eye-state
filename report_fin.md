
# EEG Eye State Classification — Complete Analysis Report

---

**Dataset Source:** [UCI Machine Learning Repository — EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)

---


## Table of Contents

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
4. [Signal Preprocessing (Bandpass + IQR)](#4-signal-preprocessing)
   - 4.1 [Bandpass Filter (0.5–45 Hz)](#41-bandpass-filter-05--45-hz)
   - 4.2 [Residual Outlier Removal (Safety Net)](#42-residual-outlier-removal-safety-net)
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
   - 9.1 [LDA](#91-lda)
   - 9.2 [t-SNE](#92-t-sne)
   - 9.3 [UMAP](#93-umap)
   - 9.4 [Clustering Evaluation](#94-clustering-evaluation)
   - 9.5 [Inference: Dimensionality Reduction Comparison](#95-inference-dimensionality-reduction-comparison)
10. [Machine Learning Classification (v2 Pipeline)](#10-machine-learning-classification-v2-pipeline)
    - 10.1 [Temporal Concept Drift Diagnosis](#101-temporal-concept-drift-diagnosis)
    - 10.2 [Split Configurations](#102-split-configurations)
    - 10.3 [Cross-Validation Results](#103-cross-validation-results)
    - 10.4 [Hold-Out Split Results](#104-hold-out-split-results)
    - 10.5 [Walk-Forward CV](#105-walk-forward-cv)
    - 10.6 [Sliding-Window CV](#106-sliding-window-cv)
11. [Deep Learning Classification (v2 Pipeline)](#11-deep-learning-classification-v2-pipeline)
    - 11.0 [Architecture Overview & Training Setup](#110-architecture-overview--training-setup)
    - 11.1 [LSTM Classifier](#111-lstm-classifier)
    - 11.2 [CNN-LSTM Hybrid](#112-cnn-lstm-hybrid)
    - 11.3 [EEG Transformer](#113-eeg-transformer)
    - 11.4 [EEGNet (Lawhern 2018)](#114-eegnet-lawhern-2018)
    - 11.5 [PatchTST Lite (Nie 2023)](#115-patchtst-lite-nie-2023)
    - 11.6 [Soft-Vote Ensemble](#116-soft-vote-ensemble)
    - 11.7 [DL Model Comparison](#117-dl-model-comparison)
12. [Final Comparison and Inference](#12-final-comparison-and-inference)
    - 12.1 [Unified Model Comparison](#121-unified-model-comparison)
    - 12.2 [Inference and Recommendation](#122-inference-and-recommendation)

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


## 1.3 Variable Classification

**Numerical Variables (Continuous):** 14 EEG electrode channels recording voltage in micro-volts (uV).

| Variable | Type | Description |
| --- | --- | --- |
| AF3 | Continuous (float64) | EEG voltage at AF3 electrode (uV) |
| F7 | Continuous (float64) | EEG voltage at F7 electrode (uV) |
| F3 | Continuous (float64) | EEG voltage at F3 electrode (uV) |
| FC5 | Continuous (float64) | EEG voltage at FC5 electrode (uV) |
| T7 | Continuous (float64) | EEG voltage at T7 electrode (uV) |
| P7 | Continuous (float64) | EEG voltage at P7 electrode (uV) |
| O1 | Continuous (float64) | EEG voltage at O1 electrode (uV) |
| O2 | Continuous (float64) | EEG voltage at O2 electrode (uV) |
| P8 | Continuous (float64) | EEG voltage at P8 electrode (uV) |
| T8 | Continuous (float64) | EEG voltage at T8 electrode (uV) |
| FC6 | Continuous (float64) | EEG voltage at FC6 electrode (uV) |
| F4 | Continuous (float64) | EEG voltage at F4 electrode (uV) |
| F8 | Continuous (float64) | EEG voltage at F8 electrode (uV) |
| AF4 | Continuous (float64) | EEG voltage at AF4 electrode (uV) |

**Categorical Variable (Target):**

| Variable | Type | Values | Description |
| --- | --- | --- | --- |
| eyeDetection | Binary (int) | 0 = Open, 1 = Closed | Eye state detected via camera during recording |


## 1.4 Electrode Positions & Significance

The Emotiv EPOC headset uses a modified 10-20 international system for electrode placement. Each electrode captures electrical activity from a specific cortical region.

| Electrode | 10-20 Position | Brain Region | Functional Significance |
| --- | --- | --- | --- |
| AF3 | Anterior Frontal Left | Prefrontal Cortex | Executive function, attention |
| F7 | Frontal Left Lateral | Left Temporal-Frontal | Language processing |
| F3 | Frontal Left | Left Frontal Lobe | Motor planning, positive affect |
| FC5 | Fronto-Central Left | Left Motor-Frontal | Motor preparation |
| T7 | Temporal Left | Left Temporal Lobe | Auditory processing, memory |
| P7 | Parietal Left | Left Parietal-Temporal | Visual-spatial processing |
| O1 | Occipital Left | Left Visual Cortex | Visual processing |
| O2 | Occipital Right | Right Visual Cortex | Visual processing |
| P8 | Parietal Right | Right Parietal-Temporal | Spatial attention |
| T8 | Temporal Right | Right Temporal Lobe | Face / emotion recognition |
| FC6 | Fronto-Central Right | Right Motor-Frontal | Motor preparation |
| F4 | Frontal Right | Right Frontal Lobe | Motor planning, negative affect |
| F8 | Frontal Right Lateral | Right Temporal-Frontal | Emotion, social cognition |
| AF4 | Anterior Frontal Right | Prefrontal Cortex | Executive function, attention |


## 1.5 Basic Statistics

Descriptive statistics for all 14 EEG channels (uV).

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

> **Note on Spike Artifacts:** Some channels exhibit extremely large max values — orders of magnitude above the 75th percentile. These are likely **electrode spike artifacts** caused by momentary loss of contact, muscle movement, or impedance changes in the Emotiv headset. These extreme values will be addressed by the outlier removal step.


## 1.6 Class Distribution

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

The correlation heatmap reveals linear relationships between EEG channels. Highly correlated channels may carry redundant information.

![Correlation Heatmap](analysis-plots/correlation_heatmap_raw.png)


## 3.3 Box Plots

Box plots highlight potential outliers beyond the 1.5x IQR whiskers.

![Box Plots](analysis-plots/boxplots_raw.png)

The raw box plots are compressed by extreme spike artifacts. Below is a **zoomed view** clipped at the 1st–99th percentile range to reveal the actual distribution of most samples.

![Box Plots Zoomed](analysis-plots/boxplots_raw_zoomed.png)


## 3.4 Histograms

Amplitude distributions per channel split by eye state.

![Histograms](analysis-plots/histograms_raw.png)


## 3.5 Violin Plots

Violin plots combine box-plot summaries with kernel density estimates.

![Violin Plots](analysis-plots/violinplots_raw.png)


## 3.6 Temporal Plots & State Transitions

Time-series plots reveal the temporal structure of EEG signals and transitions between eye states — essential context for a time-series classification task.

![Temporal Raw Signal](analysis-plots/temporal_raw_signal.png)

**State transitions:** 23 transitions between Open and Closed states in 14980 samples (117.0s recording). Average segment length: ~651 samples (5.09s).

![State Transition Points](analysis-plots/state_transitions.png)


# 4. Signal Preprocessing

EEG signals contain artifacts from eye blinks, muscle movement, and electrode drift that must be removed before analysis. This section applies a two-stage cleaning pipeline: **(1) bandpass filtering** to remove DC drift and high-frequency noise, and **(2) a light IQR safety net** to catch residual spike artifacts.


## 4.1 Bandpass Filter (0.5–45 Hz)

A **4th-order Butterworth bandpass filter** (0.5–45.0 Hz) removes DC drift and high-frequency noise while preserving the physiologically relevant EEG bands (Delta through Gamma).

The filter transfer function is:

$$H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2N}}}$$

Applied via `scipy.signal.filtfilt` (zero-phase, forward-backward filtering) to avoid phase distortion.

![Bandpass Filter Comparison](analysis-plots/bandpass_filter_comparison.png)

Bandpass filter applied to all 14 channels. Samples preserved: **14980** (no samples removed by filtering).


## 4.2 Residual Outlier Removal (Safety Net)

A **light IQR filter** (3.0x IQR, max 3 passes) removes any residual extreme values that survived bandpass filtering. The wider threshold (3.0x vs traditional 1.5x) preserves more data while still catching hardware glitches.

| Channel | Lower Bound | Upper Bound |
| --- | --- | --- |
| AF3 | -71.36 | 67.65 |
| F7 | -62.27 | 58.16 |
| F3 | -48.52 | 47.29 |
| FC5 | -57.51 | 55.72 |
| T7 | -27.03 | 26.95 |
| P7 | -30.01 | 30.31 |
| O1 | -31.76 | 32.17 |
| O2 | -40.48 | 40.77 |
| P8 | -46.06 | 46.45 |
| T8 | -45.97 | 45.61 |
| FC6 | -48.23 | 47.63 |
| F4 | -43.52 | 42.29 |
| F8 | -58.28 | 57.52 |
| AF4 | -66.21 | 63.67 |

| Metric | Value |
| --- | --- |
| Original samples | 14980 |
| Cleaned samples | 12098 |
| Removed samples | 2882 |
| Removal percentage | 19.2% |
| IQR passes | 3 |
| Bandpass filter | 0.5–45.0 Hz |

> **Preprocessing Summary:** Bandpass filter (0.5–45.0 Hz) → light IQR (3.0x, 19.2% samples removed). This pipeline preserves brain activity while removing spike artifacts.


# 5. Data Visualization (After Preprocessing)

Comparison of distributions before and after preprocessing (bandpass + IQR).


## 5.1 Box Plots Comparison

Side-by-side box plots confirm preprocessing effectiveness. Whiskers are set to **3.0x IQR** to match the cleaning threshold.

![Box Plots After Cleaning](analysis-plots/boxplots_cleaned.png)


## 5.2 Histograms After Cleaning

![Histograms After Cleaning](analysis-plots/histograms_cleaned.png)


# 6. Log-Normalization Assessment (Rejected)

Logarithmic normalization compresses the dynamic range of EEG amplitudes, reducing the impact of extreme values and making distributions more symmetric. We test `log10(x - min + 1)` on each channel and evaluate whether it improves distribution quality. **The transformed data is not used downstream** — this section documents the assessment only.


## 6.1 Before vs After — All Channels

The following grid shows the distribution of every EEG channel before (blue) and after (red) log-normalization.

![Log-Normalization — All Channels](analysis-plots/log_normalization_all_channels.png)


## 6.2 Skewness & Kurtosis Analysis

Skewness measures distribution asymmetry (0 = perfectly symmetric). Kurtosis (excess) measures tail heaviness (0 = normal). Log-normalization should reduce both towards zero.

| Channel | Skew Before | Skew After | Kurtosis Before | Kurtosis After | Improved? |
| --- | --- | --- | --- | --- | --- |
| AF3 | -0.0159 | -2.6296 | 1.5225 | 14.4651 | No |
| F7 | 0.1589 | -2.4622 | 1.3812 | 19.7854 | No |
| F3 | -0.0934 | -1.4786 | 0.1315 | 7.5997 | No |
| FC5 | 0.2547 | -1.1547 | 0.3027 | 10.8366 | No |
| T7 | 0.1452 | -1.1161 | 0.5240 | 4.7728 | No |
| P7 | 0.2017 | -1.6427 | 1.0615 | 12.5339 | No |
| O1 | 0.1332 | -1.2383 | 0.7038 | 6.8244 | No |
| O2 | -0.1038 | -1.4794 | 0.2926 | 7.9060 | No |
| P8 | 0.0391 | -1.8105 | 0.5763 | 14.4808 | No |
| T8 | -0.0011 | -1.5806 | 0.2436 | 7.7598 | No |
| FC6 | -0.0954 | -1.8962 | 0.5539 | 12.4408 | No |
| F4 | -0.1098 | -1.6366 | 0.1391 | 8.3272 | No |
| F8 | -0.0365 | -2.5561 | 1.3305 | 16.9926 | No |
| AF4 | -0.0887 | -2.5662 | 1.1931 | 15.2835 | No |

**Result:** Log-normalization improved distribution quality (reduced |skewness| + |kurtosis|) for **0/14 channels (0%)**.

> **Decision: Log-normalization REJECTED.** The transform worsened distribution quality for the majority of channels. After outlier removal, the EEG distributions are already approximately symmetric. **All subsequent analyses use the cleaned (non-transformed) data.**


## 6.3 Summary Statistics Before vs After

| Channel | Orig Mean | Orig Std | Norm Mean | Norm Std |
| --- | --- | --- | --- | --- |
| AF3 | -2.01 | 16.94 | 1.7742 | 0.1507 |
| F7 | -2.16 | 14.18 | 1.7526 | 0.1233 |
| F3 | -0.91 | 10.06 | 1.6096 | 0.1174 |
| FC5 | -0.73 | 11.44 | 1.7094 | 0.1004 |
| T7 | 0.09 | 5.86 | 1.3760 | 0.1133 |
| P7 | 0.34 | 6.18 | 1.4405 | 0.1043 |
| O1 | 0.17 | 7.08 | 1.4907 | 0.1053 |
| O2 | 0.02 | 8.78 | 1.5847 | 0.1080 |
| P8 | 0.21 | 10.25 | 1.6488 | 0.1089 |
| T8 | -0.06 | 9.84 | 1.5631 | 0.1289 |
| FC6 | -0.44 | 10.67 | 1.6536 | 0.1146 |
| F4 | -0.70 | 8.98 | 1.5468 | 0.1228 |
| F8 | -0.48 | 14.25 | 1.7446 | 0.1310 |
| AF4 | -1.59 | 16.70 | 1.7883 | 0.1415 |


# 7. Feature Engineering

Feature engineering derives new variables from raw EEG channels to capture domain-specific patterns for exploratory analysis. **Note:** The ML/DL pipeline in Sections 10–11 uses the raw 14 channels directly to avoid preprocessing data leakage.


## 7.1 Hemispheric Asymmetry

The asymmetry index $(Left - Right)$ for paired electrodes captures lateralisation differences linked to cognitive and emotional states.

| Feature | Left | Right | Mean | Std |
| --- | --- | --- | --- | --- |
| AF3_AF4_asym | AF3 | AF4 | -0.4256 | 10.0964 |
| F7_F8_asym | F7 | F8 | -1.6842 | 18.6590 |
| F3_F4_asym | F3 | F4 | -0.2076 | 6.6197 |
| FC5_FC6_asym | FC5 | FC6 | -0.2898 | 14.4560 |
| T7_T8_asym | T7 | T8 | 0.1521 | 9.7764 |
| P7_P8_asym | P7 | P8 | 0.1288 | 9.6871 |
| O1_O2_asym | O1 | O2 | 0.1511 | 7.8960 |

**Asymmetry by Eye State** — do hemispheric differences change with eye state?

| Feature | Mean (Open) | Mean (Closed) | t-statistic | p-value | Significant (p<0.05) |
| --- | --- | --- | --- | --- | --- |
| AF3_AF4_asym | -0.3957 | -0.4612 | 0.344 | 7.31e-01 | No |
| F7_F8_asym | -1.9611 | -1.3546 | -1.797 | 7.24e-02 | No |
| F3_F4_asym | -0.2168 | -0.1966 | -0.167 | 8.67e-01 | No |
| FC5_FC6_asym | -0.0952 | -0.5214 | 1.624 | 1.04e-01 | No |
| T7_T8_asym | 0.5215 | -0.2877 | 4.580 | 4.69e-06 | Yes |
| P7_P8_asym | 0.2308 | 0.0074 | 1.247 | 2.12e-01 | No |
| O1_O2_asym | 0.2518 | 0.0312 | 1.553 | 1.20e-01 | No |

**1/7** asymmetry features show a statistically significant difference between eye states (Welch's t-test, p < 0.05). Hemispheric asymmetry contributes partial discriminative signal.


## 7.2 Frequency Band Power Features

Band power features capture the relative energy in each EEG frequency band. Research shows that band powers — particularly alpha — are among the strongest predictors for eye state classification (up to 96% accuracy in papers).

$$P_{\text{band}}(t) = \frac{1}{C} \sum_{c=1}^{C} \left[x_c^{\text{band}}(t)\right]^2$$

| Feature | Band / Description | Mean | Std |
| --- | --- | --- | --- |
| band_Delta_power | 0.5–4 Hz | 68.9116 | 92.3664 |
| band_Theta_power | 4–8 Hz | 11.2756 | 13.9995 |
| band_Alpha_power | 8–12 Hz | 9.3031 | 11.4566 |
| band_Beta_power | 12–30 Hz | 14.6257 | 13.3401 |
| band_Gamma_power | 30–64 Hz | 3.7009 | 7.2674 |
| alpha_asymmetry | O1α² − O2α² | -4.8230 | 15.9385 |

**6 band power features** added. Alpha asymmetry captures the Berger effect.


## 7.3 Global Channel Statistics

Per-sample summary statistics across all 14 channels.

| Feature | Description | Mean | Std |
| --- | --- | --- | --- |
| ch_mean | Mean across 14 channels | -0.59 | 6.92 |
| ch_std | Std across 14 channels | 8.3649 | 4.1805 |


## 7.4 Feature Summary

Total engineered features for exploratory analysis: **29** (14 original + 15 engineered).

| # | Feature | Type |
| --- | --- | --- |
| 1 | AF3 | Original EEG |
| 2 | F7 | Original EEG |
| 3 | F3 | Original EEG |
| 4 | FC5 | Original EEG |
| 5 | T7 | Original EEG |
| 6 | P7 | Original EEG |
| 7 | O1 | Original EEG |
| 8 | O2 | Original EEG |
| 9 | P8 | Original EEG |
| 10 | T8 | Original EEG |
| 11 | FC6 | Original EEG |
| 12 | F4 | Original EEG |
| 13 | F8 | Original EEG |
| 14 | AF4 | Original EEG |
| 15 | AF3_AF4_asym | Engineered |
| 16 | F7_F8_asym | Engineered |
| 17 | F3_F4_asym | Engineered |
| 18 | FC5_FC6_asym | Engineered |
| 19 | T7_T8_asym | Engineered |
| 20 | P7_P8_asym | Engineered |
| 21 | O1_O2_asym | Engineered |
| 22 | band_Delta_power | Engineered |
| 23 | band_Theta_power | Engineered |
| 24 | band_Alpha_power | Engineered |
| 25 | band_Beta_power | Engineered |
| 26 | band_Gamma_power | Engineered |
| 27 | alpha_asymmetry | Engineered |
| 28 | ch_mean | Engineered |
| 29 | ch_std | Engineered |


# 8. FFT, Spectrogram and PSD Analysis

Frequency-domain analysis reveals the power distribution across brain wave bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** (8-12 Hz), **Beta** (12-30 Hz), and **Gamma** (30-64 Hz). Alpha power increases when eyes are closed (the **Berger effect**).


## 8.1 FFT Frequency Spectrum

The FFT decomposes each EEG channel into constituent frequencies.

![FFT Frequency Spectrum](analysis-plots/fft_frequency_spectrum.png)


## 8.2 Power Spectral Density (PSD)

Welch's method estimates the PSD for each channel. Shaded regions indicate standard EEG frequency bands.

![PSD Analysis](analysis-plots/psd_analysis.png)

**PSD Interpretation — Berger Effect:** Alpha-band power (8–12 Hz) increases when the eyes are closed, particularly in occipital electrodes (O1, O2). If the red curve (closed) shows higher power in the alpha band compared to blue (open), this confirms the dataset captures genuine physiological differences between eye states.


## 8.3 Spectrogram Analysis

Spectrograms show the time-frequency power distribution. Horizontal dashed lines mark band boundaries.

![Spectrograms Eyes Open](analysis-plots/spectrograms_open.png)

![Spectrograms Eyes Closed](analysis-plots/spectrograms_closed.png)


# 9. Dimensionality Reduction

Projecting high-dimensional EEG data into lower-dimensional spaces reveals clustering structure. **LDA** maximises class separability; **t-SNE** and **UMAP** capture non-linear manifold structure.


## 9.1 LDA

LDA maximises the ratio of between-class to within-class variance, yielding a single discriminant for binary classification.

![LDA 1D Projection](analysis-plots/lda_1d_projection.png)


## 9.2 t-SNE

t-Distributed Stochastic Neighbor Embedding is a non-linear technique that preserves local neighbourhood structure. A subsample of 5000 points is used for computational efficiency.

![t-SNE 2D Projection](analysis-plots/tsne_2d_projection.png)


## 9.3 UMAP

> **Note:** `umap-learn` is not installed. Skipping UMAP.


## 9.4 Clustering Evaluation

Clustering metrics quantify separation quality in reduced spaces.

| Method | Silhouette (higher better) | Davies-Bouldin (lower better) | Calinski-Harabasz (higher better) |
| --- | --- | --- | --- |
| LDA (1D) | 0.0075 | 5.3628 | 213.28 |
| t-SNE (2D) | 0.0006 | 30.5364 | 4.57 |


## 9.5 Inference: Dimensionality Reduction Comparison

| Method | Type | Strengths | Limitations | Best For |
| --- | --- | --- | --- | --- |
| **LDA** | Linear, supervised | Maximises class separation, single component for binary | Limited to C-1 components, assumes Gaussian classes | Binary/multi-class classification preprocessing |
| **t-SNE** | Non-linear, unsupervised | Excellent local structure preservation, reveals clusters | Slow on large data, non-deterministic, no inverse transform | Exploratory visualisation of cluster structure |
| **UMAP** | Non-linear, unsupervised | Preserves both local and global structure, faster than t-SNE | Hyperparameter sensitive (n_neighbors, min_dist) | Scalable visualisation, general-purpose embedding |

**Clustering metric summary:**
- **Best Silhouette Score:** LDA (1D) (0.0075)
- **Best Davies-Bouldin Index:** LDA (1D) (5.3628)
- **Best Calinski-Harabasz Score:** LDA (1D) (213.28)


# 10. Machine Learning Classification (v2 Pipeline)

The v2 ML pipeline addresses two critical issues from standard approaches: (1) **temporal concept drift** — the last 20% of the recording is 90%+ eyes-open, creating severe distribution shift; and (2) **class imbalance** — all models use `class_weight='balanced'` and CV-optimised decision thresholds. **Primary metric: Macro-F1** (equally weights both eye states under distribution shift). All splits are chronological — no shuffling, no data leakage.


## 10.1 Temporal Concept Drift Diagnosis

The subject's eye-state distribution changes dramatically over the recording. Every hold-out split places the test window in the heavily open-dominant tail, which is the root cause of the accuracy paradox and low binary-F1.

| Segment | Open | Closed | % Closed |
| --- | --- | --- | --- |
| Q1 [0–3024] | 1324 | 1700 | 56.2% |
| Q2 [3024–6049] | 1555 | 1470 | 48.6% |
| Q3 [6049–9073] | 1095 | 1929 | 63.8% |
| Q4 [9073–12098] | 2602 | 423 | 14.0% |
| Last 10% | 1146 | 64 | 5.3% |
| Last 15% | 1751 | 64 | 3.5% |
| Last 20% | 2263 | 157 | 6.5% |

> **Warning:** The last 15% of the recording is only **8.1% closed-eye**. Models trained on balanced data (≈50% closed) and tested on this window face a 44.9% distribution shift. Accuracy is misleading — Macro-F1 is the honest metric.


## 10.2 Split Configurations

| Split | Train N | CV N | Test N | Train Closed% | CV Closed% | Test Closed% | Δ Shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 70/15/15 | 8468 | 1815 | 1815 | 59.6% | 22.4% | 3.5% | 56.1% |
| 60/20/20 | 7258 | 2420 | 2420 | 60.3% | 40.7% | 6.5% | 53.8% |
| 80/10/10 | 9678 | 1210 | 1210 | 55.4% | 7.7% | 5.3% | 50.1% |


## 10.3 Cross-Validation Results (5-Fold TimeSeriesSplit)

5-fold time-series CV on the 70/15 training portion. Each fold trains on all preceding data, respecting temporal order. Scaling inside Pipeline prevents data leakage.

| Model | CV Macro-F1 Mean | CV Macro-F1 Std |
| --- | --- | --- |
| LogisticRegression | 0.4628 | 0.0584 |
| SVM_RBF | 0.4640 | 0.0846 |
| RandomForest | 0.4573 | 0.0754 |
| GradientBoosting | 0.4742 | 0.0757 |
| XGBoost | 0.4744 | 0.0698 |


## 10.4 Hold-Out Split Results


### Split 70/15/15

Train=8468 (59.6% closed) | CV=1815 (22.4% closed) | Test=1815 (3.5% closed) | Δ shift=56.1%

**LogisticRegression:** Logistic Regression models the posterior probability:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Uses `class_weight='balanced'` to penalise minority-class misclassification.

Acc=0.8402 | MacroF1=0.4856 | BinaryF1=0.0584 | AUC=0.2077 | Threshold=0.55 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1516 | 235 |
| True Closed | 55 | 9 |

TP=9  FP=235  FN=55  TN=1516

**SVM_RBF:** SVM with RBF kernel maps features into higher-dimensional space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

Maximises the soft margin with `class_weight='balanced'`.

Acc=0.8722 | MacroF1=0.4701 | BinaryF1=0.0085 | AUC=0.1375 | Threshold=0.82 | TrainTime=21.6s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1582 | 169 |
| True Closed | 63 | 1 |

TP=1  FP=169  FN=63  TN=1582

**RandomForest:** Random Forest builds 200 decision trees, each trained on a bootstrapped subset:

$$\hat{y} = \text{mode}\{h_b(\mathbf{x})\}_{b=1}^{200}$$

Uses `class_weight='balanced'` and splits by Gini impurity.

Acc=0.8006 | MacroF1=0.4500 | BinaryF1=0.0109 | AUC=0.1240 | Threshold=0.72 | TrainTime=2.9s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1451 | 300 |
| True Closed | 62 | 2 |

TP=2  FP=300  FN=62  TN=1451

**GradientBoosting:** Gradient Boosting corrects residual errors sequentially:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

200 boosting rounds, learning rate $\eta = 0.1$, max depth 5.

Acc=0.8512 | MacroF1=0.4706 | BinaryF1=0.0217 | AUC=0.2093 | Threshold=0.80 | TrainTime=20.9s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1542 | 209 |
| True Closed | 61 | 3 |

TP=3  FP=209  FN=61  TN=1542

**XGBoost:** XGBoost uses `scale_pos_weight = n_neg / n_pos` to handle class imbalance directly in the gradient computation, producing the highest closed-eye recall among ML models.

Acc=0.7521 | MacroF1=0.4399 | BinaryF1=0.0217 | AUC=0.2310 | Threshold=0.90 | TrainTime=1.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1360 | 391 |
| True Closed | 59 | 5 |

TP=5  FP=391  FN=59  TN=1360

**70/15/15 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.8402 | 0.4856 | 0.5009 | 0.5032 | 0.2077 | 0.55 |
| GradientBoosting | 0.8512 | 0.4706 | 0.4880 | 0.4638 | 0.2093 | 0.80 |
| SVM_RBF | 0.8722 | 0.4701 | 0.4838 | 0.4596 | 0.1375 | 0.82 |
| RandomForest | 0.8006 | 0.4500 | 0.4828 | 0.4300 | 0.1240 | 0.72 |
| XGBoost | 0.7521 | 0.4399 | 0.4855 | 0.4274 | 0.2310 | 0.90 |


### Split 60/20/20

Train=7258 (60.3% closed) | CV=2420 (40.7% closed) | Test=2420 (6.5% closed) | Δ shift=53.8%

**LogisticRegression:** Logistic Regression models the posterior probability:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Uses `class_weight='balanced'` to penalise minority-class misclassification.

Acc=0.6583 | MacroF1=0.4126 | BinaryF1=0.0327 | AUC=0.1816 | Threshold=0.52 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1579 | 684 |
| True Closed | 143 | 14 |

TP=14  FP=684  FN=143  TN=1579

**SVM_RBF:** SVM with RBF kernel maps features into higher-dimensional space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

Maximises the soft margin with `class_weight='balanced'`.

Acc=0.6628 | MacroF1=0.4112 | BinaryF1=0.0263 | AUC=0.2002 | Threshold=0.70 | TrainTime=15.6s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1593 | 670 |
| True Closed | 146 | 11 |

TP=11  FP=670  FN=146  TN=1593

**RandomForest:** Random Forest builds 200 decision trees, each trained on a bootstrapped subset:

$$\hat{y} = \text{mode}\{h_b(\mathbf{x})\}_{b=1}^{200}$$

Uses `class_weight='balanced'` and splits by Gini impurity.

Acc=0.6686 | MacroF1=0.4180 | BinaryF1=0.0361 | AUC=0.1874 | Threshold=0.67 | TrainTime=2.1s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1603 | 660 |
| True Closed | 142 | 15 |

TP=15  FP=660  FN=142  TN=1603

**GradientBoosting:** Gradient Boosting corrects residual errors sequentially:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

200 boosting rounds, learning rate $\eta = 0.1$, max depth 5.

Acc=0.5917 | MacroF1=0.3916 | BinaryF1=0.0426 | AUC=0.2576 | Threshold=0.67 | TrainTime=20.7s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1410 | 853 |
| True Closed | 135 | 22 |

TP=22  FP=853  FN=135  TN=1410

**XGBoost:** XGBoost uses `scale_pos_weight = n_neg / n_pos` to handle class imbalance directly in the gradient computation, producing the highest closed-eye recall among ML models.

Acc=0.6397 | MacroF1=0.4059 | BinaryF1=0.0333 | AUC=0.2440 | Threshold=0.84 | TrainTime=0.5s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1533 | 730 |
| True Closed | 142 | 15 |

TP=15  FP=730  FN=142  TN=1533

**60/20/20 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.6686 | 0.4180 | 0.4704 | 0.4019 | 0.1874 | 0.67 |
| LogisticRegression | 0.6583 | 0.4126 | 0.4685 | 0.3935 | 0.1816 | 0.52 |
| SVM_RBF | 0.6628 | 0.4112 | 0.4661 | 0.3870 | 0.2002 | 0.70 |
| XGBoost | 0.6397 | 0.4059 | 0.4677 | 0.3865 | 0.2440 | 0.84 |
| GradientBoosting | 0.5917 | 0.3916 | 0.4689 | 0.3816 | 0.2576 | 0.67 |


### Split 80/10/10

Train=9678 (55.4% closed) | CV=1210 (7.7% closed) | Test=1210 (5.3% closed) | Δ shift=50.1%

**LogisticRegression:** Logistic Regression models the posterior probability:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Uses `class_weight='balanced'` to penalise minority-class misclassification.

Acc=0.9471 | MacroF1=0.4864 | BinaryF1=0.0000 | AUC=0.2434 | Threshold=0.71 | TrainTime=0.0s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1146 | 0 |
| True Closed | 64 | 0 |

TP=0  FP=0  FN=64  TN=1146

**SVM_RBF:** SVM with RBF kernel maps features into higher-dimensional space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

Maximises the soft margin with `class_weight='balanced'`.

Acc=0.9372 | MacroF1=0.4838 | BinaryF1=0.0000 | AUC=0.1262 | Threshold=0.89 | TrainTime=28.5s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1134 | 12 |
| True Closed | 64 | 0 |

TP=0  FP=12  FN=64  TN=1134

**RandomForest:** Random Forest builds 200 decision trees, each trained on a bootstrapped subset:

$$\hat{y} = \text{mode}\{h_b(\mathbf{x})\}_{b=1}^{200}$$

Uses `class_weight='balanced'` and splits by Gini impurity.

Acc=0.9074 | MacroF1=0.4757 | BinaryF1=0.0000 | AUC=0.1162 | Threshold=0.77 | TrainTime=3.9s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1098 | 48 |
| True Closed | 64 | 0 |

TP=0  FP=48  FN=64  TN=1098

**GradientBoosting:** Gradient Boosting corrects residual errors sequentially:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

200 boosting rounds, learning rate $\eta = 0.1$, max depth 5.

Acc=0.9083 | MacroF1=0.4933 | BinaryF1=0.0348 | AUC=0.1454 | Threshold=0.82 | TrainTime=24.2s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1097 | 49 |
| True Closed | 62 | 2 |

TP=2  FP=49  FN=62  TN=1097

**XGBoost:** XGBoost uses `scale_pos_weight = n_neg / n_pos` to handle class imbalance directly in the gradient computation, producing the highest closed-eye recall among ML models.

Acc=0.8926 | MacroF1=0.4936 | BinaryF1=0.0441 | AUC=0.1583 | Threshold=0.95 | TrainTime=0.9s

|  | Pred Open | Pred Closed |
| --- | --- | --- |
| True Open | 1077 | 69 |
| True Closed | 61 | 3 |

TP=3  FP=69  FN=61  TN=1077

**80/10/10 — ML Test Summary (ranked by Macro-F1):**

| Model | Acc | MacroF1 | Prec(M) | Rec(M) | AUC | Thresh |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.8926 | 0.4936 | 0.4940 | 0.4933 | 0.1583 | 0.95 |
| GradientBoosting | 0.9083 | 0.4933 | 0.4929 | 0.4942 | 0.1454 | 0.82 |
| LogisticRegression | 0.9471 | 0.4864 | 0.4736 | 0.5000 | 0.2434 | 0.71 |
| SVM_RBF | 0.9372 | 0.4838 | 0.4733 | 0.4948 | 0.1262 | 0.89 |
| RandomForest | 0.9074 | 0.4757 | 0.4725 | 0.4791 | 0.1162 | 0.77 |


## 10.5 Walk-Forward CV (Expanding Window) — 5 Folds

Expanding-window walk-forward CV simulates real deployment: the model always trains on all available past data before predicting the next window. Future data never leaks into training.

Fold 1 — train=6049 | val=1008 | val_closed=100.00%

  LogisticRegression: Acc=1.0000 MacroF1=1.0000 AUC=nan t=0.05

  SVM_RBF: Acc=1.0000 MacroF1=1.0000 AUC=nan t=0.05

  RandomForest: Acc=1.0000 MacroF1=1.0000 AUC=nan t=0.05

  GradientBoosting: Acc=1.0000 MacroF1=1.0000 AUC=nan t=0.05

  XGBoost: Acc=0.9782 MacroF1=0.4945 AUC=nan t=0.05

Fold 2 — train=7057 | val=1008 | val_closed=86.61%

  LogisticRegression: Acc=0.7619 MacroF1=0.4666 AUC=0.4806 t=0.47

  SVM_RBF: Acc=0.6746 MacroF1=0.4786 AUC=0.5304 t=0.51

  RandomForest: Acc=0.8611 MacroF1=0.4831 AUC=0.4916 t=0.28

  GradientBoosting: Acc=0.7431 MacroF1=0.4769 AUC=0.4913 t=0.46

  XGBoost: Acc=0.8284 MacroF1=0.4852 AUC=0.4802 t=0.17

Fold 3 — train=8065 | val=1008 | val_closed=4.76%

  LogisticRegression: Acc=0.9514 MacroF1=0.6105 AUC=0.3763 t=0.58

  SVM_RBF: Acc=0.9306 MacroF1=0.4820 AUC=0.2010 t=0.95

  RandomForest: Acc=0.9514 MacroF1=0.4875 AUC=0.2043 t=0.95

  GradientBoosting: Acc=0.9435 MacroF1=0.4855 AUC=0.1340 t=0.95

  XGBoost: Acc=0.7252 MacroF1=0.4273 AUC=0.2246 t=0.95

Fold 4 — train=9073 | val=1008 | val_closed=30.46%

  LogisticRegression: Acc=0.5883 MacroF1=0.4941 AUC=0.4369 t=0.53

  SVM_RBF: Acc=0.5387 MacroF1=0.4888 AUC=0.4796 t=0.64

  RandomForest: Acc=0.5823 MacroF1=0.5169 AUC=0.5122 t=0.61

  GradientBoosting: Acc=0.5972 MacroF1=0.5145 AUC=0.4875 t=0.67

  XGBoost: Acc=0.5794 MacroF1=0.5026 AUC=0.4900 t=0.79

Fold 5 — train=10081 | val=1008 | val_closed=5.16%

  LogisticRegression: Acc=0.9484 MacroF1=0.4868 AUC=0.1241 t=0.71

  SVM_RBF: Acc=0.9296 MacroF1=0.4954 AUC=0.2243 t=0.85

  RandomForest: Acc=0.9484 MacroF1=0.4868 AUC=0.2482 t=0.88

  GradientBoosting: Acc=0.9028 MacroF1=0.4940 AUC=0.3002 t=0.76

  XGBoost: Acc=0.9018 MacroF1=0.4935 AUC=0.3161 t=0.95

**Walk-Forward CV — Mean ± Std (primary: Macro-F1):**

| Model | MacroF1 Mean±Std | Acc Mean±Std | AUC Mean±Std |
| --- | --- | --- | --- |
| LogisticRegression | 0.6116±0.2006 | 0.8500±0.1541 | 0.2836±0.1880 |
| SVM_RBF | 0.5890±0.2056 | 0.8147±0.1771 | 0.2871±0.1949 |
| RandomForest | 0.5949±0.2029 | 0.8687±0.1500 | 0.2913±0.1914 |
| GradientBoosting | 0.5942±0.2033 | 0.8373±0.1473 | 0.2826±0.1938 |
| XGBoost | 0.4806±0.0272 | 0.8026±0.1394 | 0.3022±0.1814 |


## 10.6 Sliding-Window CV (Fixed-Size Window) — 5 Folds

Sliding-window CV tests how well models generalise across different temporal regimes (different epochs of the recording). High fold-variance directly quantifies the severity of concept drift.

Fold 1 — train=6049 | val=1008 | val_closed=100.00%

  LogisticRegression: Acc=1.0000 MacroF1=1.0000 AUC=nan

  SVM_RBF: Acc=1.0000 MacroF1=1.0000 AUC=nan

  RandomForest: Acc=1.0000 MacroF1=1.0000 AUC=nan

  GradientBoosting: Acc=1.0000 MacroF1=1.0000 AUC=nan

  XGBoost: Acc=0.9782 MacroF1=0.4945 AUC=nan

Fold 2 — train=6049 | val=1008 | val_closed=86.61%

  LogisticRegression: Acc=0.6210 MacroF1=0.4850 AUC=0.5483

  SVM_RBF: Acc=0.8026 MacroF1=0.4777 AUC=0.5141

  RandomForest: Acc=0.8442 MacroF1=0.4875 AUC=0.4731

  GradientBoosting: Acc=0.8433 MacroF1=0.4756 AUC=0.4528

  XGBoost: Acc=0.8353 MacroF1=0.4833 AUC=0.4804

Fold 3 — train=6049 | val=1008 | val_closed=4.76%

  LogisticRegression: Acc=0.9534 MacroF1=0.6150 AUC=0.3077

  SVM_RBF: Acc=0.9236 MacroF1=0.4801 AUC=0.1158

  RandomForest: Acc=0.9494 MacroF1=0.4870 AUC=0.0880

  GradientBoosting: Acc=0.9276 MacroF1=0.4812 AUC=0.0718

  XGBoost: Acc=0.5893 MacroF1=0.3730 AUC=0.1065

Fold 4 — train=6049 | val=1008 | val_closed=30.46%

  LogisticRegression: Acc=0.6865 MacroF1=0.4937 AUC=0.3999

  SVM_RBF: Acc=0.5615 MacroF1=0.4694 AUC=0.5064

  RandomForest: Acc=0.5347 MacroF1=0.5099 AUC=0.5459

  GradientBoosting: Acc=0.6548 MacroF1=0.4990 AUC=0.4857

  XGBoost: Acc=0.6210 MacroF1=0.5087 AUC=0.5203

Fold 5 — train=6049 | val=1008 | val_closed=5.16%

  LogisticRegression: Acc=0.9286 MacroF1=0.5078 AUC=0.2395

  SVM_RBF: Acc=0.9474 MacroF1=0.4865 AUC=0.1936

  RandomForest: Acc=0.9484 MacroF1=0.4868 AUC=0.1923

  GradientBoosting: Acc=0.9405 MacroF1=0.5008 AUC=0.2265

  XGBoost: Acc=0.8413 MacroF1=0.4690 AUC=0.2158

**Sliding-Window CV — Mean ± Std:**

| Model | MacroF1 Mean±Std | Acc Mean±Std | AUC Mean±Std |
| --- | --- | --- | --- |
| LogisticRegression | 0.6203±0.1955 | 0.8379±0.1535 | 0.2991±0.1819 |
| SVM_RBF | 0.5827±0.2087 | 0.8470±0.1567 | 0.2660±0.2087 |
| RandomForest | 0.5942±0.2031 | 0.8554±0.1681 | 0.2599±0.2140 |
| GradientBoosting | 0.5913±0.2046 | 0.8732±0.1201 | 0.2474±0.1957 |
| XGBoost | 0.4657±0.0481 | 0.7730±0.1466 | 0.2646±0.2046 |

**Feature Importance (RandomForest — 70/15/15 training partition):**

![Feature Importance](analysis-plots/ml_feature_importance_v2.png)

![ML ROC Curves](analysis-plots/ml_roc_curves_v2.png)


# 11. Deep Learning Classification (v2 Pipeline)

All DL models use PyTorch with: **(1) weighted CrossEntropyLoss** (inverse class frequency) to handle imbalance, **(2) AdamW + CosineAnnealingLR** for stable training, **(3) CV-optimised decision threshold** to correct the accuracy paradox under concept drift, and **(4) Macro-F1 as primary metric**. Sequences are built per partition — no cross-boundary leakage.


## 11.0 Architecture Overview & Training Setup

**Binary Cross-Entropy (weighted):**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

where $w_c = \frac{N}{2 \cdot N_c}$ is the per-class weight. **Sequence length:** SEQ_LEN=64 samples (≈500ms at 128 Hz). **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. **Scheduler:** CosineAnnealingLR over 25 epochs.

| Model | Architecture | Parameters | Key Innovation |
| --- | --- | --- | --- |
| LSTM | BiLSTM(128)×2 → AvgPool → MLP | ~200K | Long-range temporal dependencies |
| CNN-LSTM | Conv1D(64,128) → BiLSTM(64) → MLP | ~150K | Local feature extraction + sequence memory |
| EEGTransformer | CLS + PE + 3× TransEnc(d=64,h=4) → MLP | ~80K | Global cross-electrode attention |
| EEGNet | Depthwise Conv2D blocks → Linear | ~400 | Electrode-aware, compact, best calibrated |
