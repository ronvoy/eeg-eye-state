# EEG eye state classification — consolidated report (2026)

This document **replaces the narrative role** of the long-form [`report.md`](../report.md) in the project root for the **final pipeline** implemented in [`eeg-eye-state.ipynb`](../eeg-eye-state.ipynb). The original root `report.md` remains a useful archive of plots, tables, and exploratory sections (FFT, PSD, UMAP, legacy neural nets with failed spectrogram/CNN runs).

**Dataset:** [UCI EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state), file `dataset/eeg_data_og.csv` — 14 Emotiv channels, binary `eyeDetection` (0 = open, 1 = closed), 128 Hz, single continuous segment (~117 s).

---

## 1. Scope of the consolidated work

Older notebooks (`DataMiningGennaio.ipynb`, `dmml.ipynb`, `script.ipynb`, `eyestate-classification-on-eeg-data.ipynb`, `eeg_classification_leakage_free.ipynb`, `EEG_Eye_Predict_GodNotebook_v245_ExamGrade_FIXROC.ipynb`) mixed overlapping EDA, many model variants, and different validation philosophies. The **final notebook** keeps only what consistently worked best or adds clear value:

| Area | Kept in `eeg-eye-state.ipynb` | Dropped as redundant / weak |
| --- | --- | --- |
| Cleaning | Iterative per-channel IQR (Tukey) | Log-normalisation (rejected in original report) |
| Tabular features | 14 channels + 7 asymmetries + `ch_mean`, `ch_std` | Extra spectral pipelines for the *tabular* branch |
| Classical ML | Logistic regression (baseline), **k-NN**, **RBF SVM**, **Random forest** | Gradient boosting (worse than forest here), long model zoo |
| Validation | Stratified 70/15/15 + scaler on train only; **extra chronological split** for honesty | Random split without discussing temporal autocorrelation |
| Deep learning | **EEGNet** + **EEGFormer** on **chronological** windows | Legacy Keras 1D-CNN / LSTM / spectrogram CNN that collapsed on small windows; full TCFormer sweep (see God notebook if needed) |

**Process diagrams (Mermaid):** each algorithm’s flow is sketched under [`process/`](../process/) (preprocessing, LR, k-NN, SVM, RF, EEGNet, EEGFormer).

---

## 2. Methodology summary

1. **Load** CSV from `dataset/eeg_data_og.csv`.
2. **IQR cleaning:** repeat row filtering until stable — remove any sample where *any* channel lies outside 1.5×IQR; removes spike artifacts (often ~35% of rows, consistent with the original report).
3. **Feature engineering:** hemispheric difference features for paired electrodes; global mean and std across channels → **23-dimensional** tabular vector.
4. **Tabular models:** `StandardScaler` fit **only** on training data; stratified train/validation/test split for reporting metrics comparable to the coursework tables.
5. **Temporal reality check:** the same four sklearn models are evaluated on **chronological** 70% train / 30% test (no shuffle). Metrics typically **drop** versus stratified split because adjacent EEG samples are highly correlated and random splits leak temporal structure.
6. **Deep learning:** chronological **70/15/15** split on the *cleaned raw 14 channels*; z-score using train statistics; sliding windows (64 samples, stride 4); optional per-window channel z-score; **BCE-with-logits** with class-balanced `pos_weight`; early stopping on validation loss. Architectures:
   - **EEGNet:** compact Lawhern-style blocks with depthwise spatial conv and separable temporal conv (max-norm constraints on selected layers).
   - **EEGFormer:** temporal Transformer baseline — each time step embeds all channel amplitudes (`Linear(14 → d_model)`), learned positional bias, `TransformerEncoder`, mean pooling, linear head. This is a **from-scratch** Transformer encoder for EEG windows, aligned with “EEG + Transformer” lines of work; it is **not** a claim of reproducing any single external pretrained checkpoint.

---

## 3. Expected results (qualitative)

- **Stratified tabular evaluation:** **k-NN** and **RBF SVM** typically lead, with **random forest** close behind — matching the ordering in the original [`report.md`](../report.md) (k-NN often best F1 / AUC under i.i.d.-style sampling).
- **Chronological tabular evaluation:** all models degrade; the gap quantifies how optimistic random stratification is on this recording.
- **Deep models:** EEGNet and EEGFormer are evaluated on **held-out future time**; reported test F1/AUC are **comparable across DL models** and should be interpreted **only** against each other (not against tabular stratified scores, which use different splits and sample units).

Exact numbers should be taken from the executed [`eeg-eye-state.ipynb`](../eeg-eye-state.ipynb) outputs on your machine.

---

## 4. Artefacts and presentation

| Artefact | Role |
| --- | --- |
| [`eeg-eye-state.ipynb`](../eeg-eye-state.ipynb) | Single runnable pipeline |
| [`process/*.md`](../process/) | Mermaid flowcharts per method |
| [`docs/slide.md`](slide.md) | Slide titles + bullet descriptions for oral exam / deck export |
| [`report.md`](../report.md) | Legacy full analysis + static plot paths |

---

## 5. Ethics and limitations

- Single-subject continuous recording: conclusions apply to **this protocol / headset**, not universal brain–computer interfaces.
- Aggressive IQR removal trades **bias vs variance** (fewer but cleaner samples).
- Always report **both** stratified and chronological metrics when claiming generalisation.

---

## References (indicative)

- UCI EEG Eye State dataset page (citation there).
- Lawhern, V. J., et al. *EEGNet* (compact CNN for EEG).
- General Transformer encoder formulation (Vaswani et al.) as used in the **EEGFormer** baseline block.
