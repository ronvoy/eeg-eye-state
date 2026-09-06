# Slide deck — EEG eye state (Markdown outline)

Use each `##` as a slide title; body text is the speaker description. Export to PowerPoint / Beamer / Marp as needed.

---

## Title

**EEG-based eye-state classification (open vs closed)**  
Consolidated data-mining and machine-learning study on the UCI Emotiv recording. Final artefact: single notebook `eeg-eye-state.ipynb`, dataset `dataset/eeg_data_og.csv`.

---

## Problem and data

**Binary classification on a high-frequency physiological time series.**  
One continuous measurement at 128 Hz; 14 scalp channels; label from camera-aligned eye state. Adjacent samples are strongly correlated, so **random train/test splits inflate performance** — we show both stratified and chronological evaluation.

---

## Preprocessing: why cleaning matters

**Consumer EEG contains short spike artifacts.**  
Iterative **IQR (Tukey) filtering** per channel removes extreme samples that would dominate distance-based models and scaling. Trade-off: fewer rows, but more physically plausible amplitudes. Flow: see `process/preprocessing_pipeline.md`.

---

## Feature engineering

**From 14 raw channels to 23 tabular features.**  
Seven **hemispheric asymmetry** terms (left minus right pairs), plus **global mean** and **std** across channels capture gross activation and dispersion. These features align with neuroscience intuition (e.g., occipital alpha differences when eyes close).

---

## Classical models we keep

**Only strong, interpretable baselines.**  
**Logistic regression** (linear margin), **k-NN** (local decision surface), **RBF SVM** (kernelised margin), **random forest** (ensemble trees). Gradient boosting was dropped as consistently weaker than the forest in prior coursework runs. Individual flows: `process/logistic_regression.md`, `knn.md`, `svm.md`, `random_forest.md`.

---

## Validation strategy

**Two complementary splits.**  
(1) **Stratified 70/15/15** with `StandardScaler` fit on train — comparable to standard ML benchmarks. (2) **Chronological 70/30** on the same feature matrix — exposes temporal leakage if we only report (1). Deep learning uses **chronological 70/15/15** *before* windowing so windows never cross split boundaries.

---

## Deep learning: EEGNet

**Spatial–temporal CNN tailored to multi-channel EEG windows.**  
Input windows `[channels × time]`; temporal convolution, depthwise spatial mixing across electrodes, separable convolutions, pooling, linear head with norm constraints. Suited to short 0.5 s contexts at 128 Hz. Flow: `process/eegnet.md`.

---

## Deep learning: EEGFormer

**Transformer encoder over time.**  
Each time step is a token: linear map from 14 channel values to `d_model`, plus learned positional encoding, stacked self-attention blocks, mean pooling, classifier. Adds a modern **attention-based** baseline alongside CNN; trained end-to-end on our windows (no external pretrained weights). Flow: `process/eegformer.md`.

---

## Results interpretation

**Do not rank stratified k-NN against chronological EEGNet.**  
Tabular stratified scores are **optimistic upper bounds** under correlated samples; chronological and window-based tests measure **forward prediction** in time. Typically k-NN / SVM excel on engineered features under stratified split; DL metrics must be read on their **own** held-out segment.

---

## Conclusions

**Practical recommendation:** for **offline batch labelling** with engineered features, **k-NN or SVM** after IQR cleaning and scaling remain the coursework sweet spot. For **streaming / window-based** models with leakage-aware splits, compare **EEGNet vs EEGFormer** in `eeg-eye-state.ipynb`. Always document **which split** you report.

---

## References and appendix

**Dataset:** UCI EEG Eye State. **Code:** `eeg-eye-state.ipynb`, diagrams under `process/`. **Legacy full report** with FFT/PSD figures: root `report.md`.
