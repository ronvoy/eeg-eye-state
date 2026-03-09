
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

![Class Balance](analysis-plots\class_balance_raw.png)


## 3.2 Correlation Heatmap

The correlation heatmap reveals linear relationships between EEG channels. Highly correlated channels may carry redundant information.

![Correlation Heatmap](analysis-plots\correlation_heatmap_raw.png)


## 3.3 Box Plots

Box plots highlight potential outliers beyond the 1.5x IQR whiskers.

![Box Plots](analysis-plots\boxplots_raw.png)


## 3.4 Histograms

Amplitude distributions per channel split by eye state.

![Histograms](analysis-plots\histograms_raw.png)


## 3.5 Violin Plots

Violin plots combine box-plot summaries with kernel density estimates.

![Violin Plots](analysis-plots\violinplots_raw.png)


# 4. Outlier Removal

Outliers in EEG data arise from muscle artifacts, electrode displacement, or external interference. An **iterative IQR method** (1.5x interquartile range) is applied in multiple passes until no further outliers remain, ensuring that new outliers exposed by earlier passes are also removed.

**Passes required:** 5 (converged when no further outliers found).

| Channel | Lower Bound (Pass 1) | Upper Bound (Pass 1) |
| --- | --- | --- |
| AF3 | 4233.59 | 4358.71 |
| F7 | 3945.89 | 4062.81 |
| F3 | 4224.88 | 4292.56 |
| FC5 | 4080.76 | 4152.57 |
| T7 | 4310.50 | 4365.91 |
| P7 | 4592.55 | 4643.86 |
| O1 | 4021.53 | 4115.90 |
| O2 | 4578.98 | 4648.71 |
| P8 | 4165.39 | 4233.07 |
| T8 | 4195.12 | 4262.83 |
| FC6 | 4163.34 | 4235.14 |
| F4 | 4243.33 | 4306.93 |
| F8 | 4557.97 | 4644.08 |
| AF4 | 4306.66 | 4401.03 |

| Metric | Value |
| --- | --- |
| Original samples | 14980 |
| Cleaned samples | 9695 |
| Removed samples | 5285 |
| Removal percentage | 35.3% |
| Passes | 5 |


# 5. Data Visualization (After Outlier Removal)

Comparison of distributions before and after outlier removal.


## 5.1 Box Plots Comparison

Side-by-side box plots confirm outlier removal effectiveness.

![Box Plots After Cleaning](analysis-plots\boxplots_cleaned.png)


## 5.2 Histograms After Cleaning

![Histograms After Cleaning](analysis-plots\histograms_cleaned.png)


# 6. Log-Normalization

Logarithmic normalization compresses the dynamic range of EEG amplitudes, reducing the impact of extreme values and making distributions more symmetric. We apply `log10(x - min + 1)` to each channel.

![Log-Normalization Effect](analysis-plots\log_normalization_comparison.png)

| Channel | Orig Mean | Orig Std | Norm Mean | Norm Std |
| --- | --- | --- | --- | --- |
| AF3 | 4291.08 | 14.98 | 1.5938 | 0.1840 |
| F7 | 4001.07 | 17.93 | 1.6432 | 0.2100 |
| F3 | 4259.13 | 11.08 | 1.4765 | 0.1957 |
| FC5 | 4116.31 | 12.21 | 1.4992 | 0.1928 |
| T7 | 4337.72 | 8.99 | 1.3916 | 0.1735 |
| P7 | 4617.74 | 9.00 | 1.3102 | 0.2350 |
| O1 | 4067.93 | 15.92 | 1.5967 | 0.1839 |
| O2 | 4613.88 | 11.50 | 1.4747 | 0.1908 |
| P8 | 4199.35 | 11.13 | 1.4657 | 0.1901 |
| T8 | 4229.07 | 11.28 | 1.4555 | 0.2005 |
| FC6 | 4199.03 | 12.16 | 1.4824 | 0.2158 |
| F4 | 4274.63 | 10.52 | 1.4518 | 0.1842 |
| F8 | 4600.62 | 14.20 | 1.5456 | 0.2126 |
| AF4 | 4352.87 | 15.73 | 1.6107 | 0.1879 |


# 7. Feature Engineering

Feature engineering derives new variables from raw EEG channels to capture domain-specific patterns that may improve classification performance.


## 7.1 Hemispheric Asymmetry

The asymmetry index $(Left - Right)$ for paired electrodes captures lateralisation differences linked to cognitive and emotional states. Research shows that hemispheric imbalance correlates with attentional shifts associated with eye opening and closing.

| Feature | Left | Right | Mean | Std |
| --- | --- | --- | --- | --- |
| AF3_AF4_asym | AF3 | AF4 | -61.7892 | 9.2975 |
| F7_F8_asym | F7 | F8 | -599.5502 | 20.7529 |
| F3_F4_asym | F3 | F4 | -15.4972 | 9.3525 |
| FC5_FC6_asym | FC5 | FC6 | -82.7151 | 15.6083 |
| T7_T8_asym | T7 | T8 | 108.6488 | 12.0318 |
| P7_P8_asym | P7 | P8 | 418.3872 | 11.0518 |
| O1_O2_asym | O1 | O2 | -545.9479 | 16.1104 |


## 7.2 Global Channel Statistics

Per-sample summary statistics across all 14 channels capture overall brain activity levels at each time point.

| Feature | Description | Mean | Std |
| --- | --- | --- | --- |
| ch_mean | Mean across 14 channels | 4297.17 | 7.72 |
| ch_std | Std across 14 channels | 196.1182 | 3.2813 |


## 7.3 Feature Summary

Total features for classification: **23** (14 original + 9 engineered).

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
| 22 | ch_mean | Engineered |
| 23 | ch_std | Engineered |


# 8. FFT, Spectrogram and PSD Analysis

Frequency-domain analysis reveals the power distribution across brain wave bands: **Delta** (0.5-4 Hz), **Theta** (4-8 Hz), **Alpha** (8-12 Hz), **Beta** (12-30 Hz), and **Gamma** (30-64 Hz). Alpha power increases when eyes are closed (the **Berger effect**).


## 8.1 FFT Frequency Spectrum

The FFT decomposes each EEG channel into constituent frequencies.

![FFT Frequency Spectrum](analysis-plots\fft_frequency_spectrum.png)


## 8.2 Power Spectral Density (PSD)

Welch's method estimates the PSD for each channel. Shaded regions and labels indicate standard EEG frequency bands.

![PSD Analysis](analysis-plots\psd_analysis.png)


## 8.3 Spectrogram Analysis

Spectrograms show the time-frequency power distribution. Horizontal dashed lines mark band boundaries (4, 8, 12, 30 Hz).

![Spectrograms Eyes Open](analysis-plots\spectrograms_open.png)

![Spectrograms Eyes Closed](analysis-plots\spectrograms_closed.png)


# 9. Dimensionality Reduction

Projecting high-dimensional EEG data into lower-dimensional spaces reveals clustering structure. **PCA** maximises variance; **LDA** maximises class separability; **t-SNE** and **UMAP** capture non-linear manifold structure.


## 9.1 PCA

PCA identifies orthogonal directions of maximum variance.

![PCA Variance](analysis-plots\pca_variance.png)

| Component | Variance (%) | Cumulative (%) |
| --- | --- | --- |
| PC1 | 29.78 | 29.78 |
| PC2 | 20.11 | 49.89 |
| PC3 | 9.40 | 59.28 |
| PC4 | 7.66 | 66.94 |
| PC5 | 7.11 | 74.05 |
| PC6 | 4.75 | 78.80 |
| PC7 | 4.06 | 82.87 |
| PC8 | 3.84 | 86.71 |
| PC9 | 3.79 | 90.50 |
| PC10 | 2.75 | 93.25 |
| PC11 | 2.51 | 95.76 |
| PC12 | 1.81 | 97.57 |
| PC13 | 1.36 | 98.93 |
| PC14 | 1.07 | 99.99 |
| PC15 | 0.01 | 100.00 |
| PC16 | 0.00 | 100.00 |
| PC17 | 0.00 | 100.00 |
| PC18 | 0.00 | 100.00 |
| PC19 | 0.00 | 100.00 |
| PC20 | 0.00 | 100.00 |
| PC21 | 0.00 | 100.00 |
| PC22 | 0.00 | 100.00 |
| PC23 | 0.00 | 100.00 |

**11 components** capture >= 95% of variance.

![PCA 2D Projection](analysis-plots\pca_2d_projection.png)


## 9.2 LDA

LDA maximises the ratio of between-class to within-class variance, yielding a single discriminant for binary classification.

![LDA 1D Projection](analysis-plots\lda_1d_projection.png)


## 9.3 t-SNE

t-Distributed Stochastic Neighbor Embedding is a non-linear technique that preserves local neighbourhood structure. A subsample of 5000 points is used for computational efficiency.

![t-SNE 2D Projection](analysis-plots\tsne_2d_projection.png)


## 9.4 UMAP

UMAP preserves both local and global structure, often producing cleaner clusters than t-SNE with faster computation.

![UMAP 2D Projection](analysis-plots\umap_2d_projection.png)


## 9.5 Clustering Evaluation

Clustering metrics quantify separation quality in reduced spaces.

| Method | Silhouette (higher better) | Davies-Bouldin (lower better) | Calinski-Harabasz (higher better) |
| --- | --- | --- | --- |
| PCA (2D) | 0.0035 | 23.0642 | 14.39 |
| LDA (1D) | 0.1561 | 1.6419 | 2148.12 |
| t-SNE (2D) | 0.0183 | 10.0752 | 42.44 |
| UMAP (2D) | 0.0297 | 16.6995 | 14.01 |


# 10. Machine Learning Classification

Five classical ML algorithms are evaluated using a 70/30 stratified train-test split. `StandardScaler` is fit **exclusively on training data** to prevent data leakage.


## 10.1 Train/Test Split & Class Balance

Stratified split: 70% train / 30% test, preserving class proportions.

| Split | Open (0) | Closed (1) | Total | Closed % |
| --- | --- | --- | --- | --- |
| Train | 3751 | 3035 | 6786 | 44.7% |
| Test | 1608 | 1301 | 2909 | 44.7% |


## 10.2 Cross-Validation Results (5-Fold Stratified)

5-fold stratified cross-validation on the training set.

| Model | CV F1 Mean | CV F1 Std |
| --- | --- | --- |
| Logistic Regression | 0.6313 | 0.0141 |
| K-Nearest Neighbors | 0.9353 | 0.0053 |
| Support Vector Machine | 0.9148 | 0.0069 |
| Random Forest | 0.8932 | 0.0065 |
| Gradient Boosting | 0.8551 | 0.0071 |


## 10.3 Logistic Regression

Logistic Regression models the posterior probability using the sigmoid function:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The model minimises binary cross-entropy loss with L2 regularisation:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)] + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

It serves as an interpretable linear baseline for binary classification.

| Metric | Value |
| --- | --- |
| Accuracy | 0.6978 |
| Precision | 0.6939 |
| Recall | 0.5803 |
| F1-Score | 0.6321 |
| AUC-ROC | 0.7510 |
| Training Time | 0.030s |


## 10.4 K-Nearest Neighbors

KNN classifies each sample by majority vote among its $k$ nearest neighbours using the Euclidean distance metric:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{m=1}^{M}(x_{im} - x_{jm})^2}$$

The predicted class is:

$$\hat{y} = \arg\max_c \sum_{i \in N_k(\mathbf{x})} \mathbb{1}(y_i = c)$$

KNN is non-parametric, making no distributional assumptions. With $k=5$ and standardised features, it captures local EEG decision boundaries.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9529 |
| Precision | 0.9491 |
| Recall | 0.9454 |
| F1-Score | 0.9472 |
| AUC-ROC | 0.9868 |
| Training Time | 0.000s |


## 10.5 Support Vector Machine

SVM finds the hyperplane that maximises the margin between classes. The RBF kernel maps features into higher-dimensional space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

The optimisation objective with soft margin is:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(\mathbf{w}^T\phi(\mathbf{x}_i) + b))$$

The RBF kernel captures non-linear decision boundaries between eye states.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9295 |
| Precision | 0.9322 |
| Recall | 0.9085 |
| F1-Score | 0.9202 |
| AUC-ROC | 0.9790 |
| Training Time | 10.131s |


## 10.6 Random Forest

Random Forest builds an ensemble of $B$ decision trees, each trained on a bootstrapped subset with random feature selection:

$$\hat{y} = \text{mode}\{h_b(\mathbf{x})\}_{b=1}^{B}$$

Each tree splits nodes using the Gini impurity criterion:

$$G = 1 - \sum_{c=1}^{C} p_c^2$$

Bagging reduces variance and random subspace selection decorrelates trees. 200 estimators are used.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9230 |
| Precision | 0.9339 |
| Recall | 0.8909 |
| F1-Score | 0.9119 |
| AUC-ROC | 0.9768 |
| Training Time | 0.923s |


## 10.7 Gradient Boosting

Gradient Boosting builds an additive ensemble where each tree corrects residual errors of the previous ensemble:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

Each tree $h_m$ is fit to the negative gradient of the loss function. The learning rate $\eta$ controls the contribution of each tree. 200 boosting rounds are used with default depth and $\eta = 0.1$.

| Metric | Value |
| --- | --- |
| Accuracy | 0.8707 |
| Precision | 0.8745 |
| Recall | 0.8301 |
| F1-Score | 0.8517 |
| AUC-ROC | 0.9458 |
| Training Time | 9.682s |


## 10.8 Feature Importance

Feature importance from Random Forest and Gradient Boosting.

![Feature Importance](analysis-plots\feature_importance.png)


## 10.9 ROC Curves

ROC curves plot True Positive Rate vs False Positive Rate.

![ML ROC Curves](analysis-plots\ml_roc_curves.png)


## 10.10 ML Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6978 | 0.6939 | 0.5803 | 0.6321 | 0.7510 | 0.030 |
| K-Nearest Neighbors | 0.9529 | 0.9491 | 0.9454 | 0.9472 | 0.9868 | 0.000 |
| Support Vector Machine | 0.9295 | 0.9322 | 0.9085 | 0.9202 | 0.9790 | 10.131 |
| Random Forest | 0.9230 | 0.9339 | 0.8909 | 0.9119 | 0.9768 | 0.923 |
| Gradient Boosting | 0.8707 | 0.8745 | 0.8301 | 0.8517 | 0.9458 | 9.682 |

![ML Confusion Matrices](analysis-plots\ml_confusion_matrices.png)

![ML Comparison Chart](analysis-plots\ml_comparison_chart.png)


# 11. Neural Network Classification

Deep-learning models learn hierarchical feature representations from raw EEG signals. This section evaluates a **1D CNN**, a **2D CNN on spectrograms**, and an **LSTM** network.

Window size = 64 samples, step = 16. Total windows: 602 (train 421, test 181).


## 11.1 1D CNN on Raw EEG

A 1D Convolutional Neural Network applies learnable filters across the temporal dimension of multi-channel EEG windows (64 samples x 14 channels). The convolution for filter $f$ at position $t$ is:

$$y_t^{(f)} = \text{ReLU}\left(\sum_{k=0}^{K-1} \sum_{c=1}^{C} w_{k,c}^{(f)} \cdot x_{t+k,c} + b^{(f)}\right)$$

where $K$ is the kernel size and $C$ the number of channels. Max-pooling reduces dimensionality and global average pooling aggregates features.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9669 |
| Precision | 0.9615 |
| Recall | 0.9615 |
| F1-Score | 0.9615 |
| AUC-ROC | 0.9964 |
| Training Time | 12.913s |

![1D CNN Training History](analysis-plots\cnn1d_training.png)


## 11.2 CNN on Spectrograms

A 2D CNN processes spectrogram representations of EEG windows as multi-channel images (frequency x time x EEG channels). The 2D convolution learns frequency-time patterns:

$$Y_{i,j}^{(f)} = \text{ReLU}\left(\sum_{m,n,c} W_{m,n,c}^{(f)} \cdot X_{i+m,j+n,c} + b^{(f)}\right)$$

**Improvements:** Smaller windows with more overlap generate more training samples. Class weights address label imbalance. A reduced learning rate and increased patience improve convergence.

Spectrogram window = 64, step = 8. Shape per sample: (17, 5, 14) (freq x time x channels). Total samples: 1204.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9088 |
| Precision | 0.9310 |
| Recall | 0.8544 |
| F1-Score | 0.8911 |
| AUC-ROC | 0.9715 |
| Training Time | 23.779s |

![CNN Spectrogram Training History](analysis-plots\cnn2d_spectrogram_training.png)


## 11.3 LSTM / RNN

Long Short-Term Memory networks capture temporal dependencies through gating mechanisms:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(candidate)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell state)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}$$

The forget gate controls what to discard, the input gate what to store, and the output gate what to expose as the hidden state.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9558 |
| Precision | 0.9487 |
| Recall | 0.9487 |
| F1-Score | 0.9487 |
| AUC-ROC | 0.9829 |
| Training Time | 16.321s |

![LSTM Training History](analysis-plots\lstm_training.png)


## 11.4 Neural Network Comparison

Side-by-side comparison of all neural-network architectures.

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Train Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1D CNN | 0.9669 | 0.9615 | 0.9615 | 0.9615 | 0.9964 | 12.913 |
| CNN (Spectrogram) | 0.9088 | 0.9310 | 0.8544 | 0.8911 | 0.9715 | 23.779 |
| LSTM | 0.9558 | 0.9487 | 0.9487 | 0.9487 | 0.9829 | 16.321 |

![Neural Network Comparison](analysis-plots\nn_comparison_chart.png)

![NN Confusion Matrices](analysis-plots\nn_confusion_matrices.png)


# 12. Final Comparison and Inference

This section unifies all models — classical ML and deep learning — ranked by F1-Score.


## 12.1 Unified Comparison Table

| Rank | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1D CNN | 0.9669 | 0.9615 | 0.9615 | 0.9615 | 0.9964 |
| 2 | LSTM | 0.9558 | 0.9487 | 0.9487 | 0.9487 | 0.9829 |
| 3 | K-Nearest Neighbors | 0.9529 | 0.9491 | 0.9454 | 0.9472 | 0.9868 |
| 4 | Support Vector Machine | 0.9295 | 0.9322 | 0.9085 | 0.9202 | 0.9790 |
| 5 | Random Forest | 0.9230 | 0.9339 | 0.8909 | 0.9119 | 0.9768 |
| 6 | CNN (Spectrogram) | 0.9088 | 0.9310 | 0.8544 | 0.8911 | 0.9715 |
| 7 | Gradient Boosting | 0.8707 | 0.8745 | 0.8301 | 0.8517 | 0.9458 |
| 8 | Logistic Regression | 0.6978 | 0.6939 | 0.5803 | 0.6321 | 0.7510 |

![Final Model Comparison](analysis-plots\final_comparison.png)


## 12.2 Inference and Recommendation

### Best Overall Model: **1D CNN**

Based on comprehensive evaluation, **1D CNN** achieves the highest F1-Score of **0.9615** with accuracy **0.9669** and AUC-ROC **0.9964**.

Runner-up: **LSTM** (F1 = 0.9487).

**Key Observations:**

- Deep learning (**1D CNN**) outperforms the best classical ML model (**K-Nearest Neighbors**) by **1.43** percentage points in F1-Score.

- **For production deployment**, **1D CNN** is recommended.

- **For low-latency applications**, **K-Nearest Neighbors** offers the fastest training (0.000s) with F1 = 0.9472.



---

