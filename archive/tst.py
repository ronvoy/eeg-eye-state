# pip install numpy pandas scikit-learn xgboost torch
"""
=============================================================================
EEG Eye-State Detection — ML / DL Pipeline  v2
=============================================================================
Dataset   : EEG Eye State  (14 electrodes, 14,980 samples ≈ 128 Hz)
Target    : eyeDetection   (0 = eyes open, 1 = eyes closed)
Nature    : Temporal / non-stationary  — NO shuffle, NO data leakage

────────────────────────────────────────────────────────────────────────────
ROOT CAUSE OF LOW F1 (diagnosed from data analysis)
────────────────────────────────────────────────────────────────────────────
The last 15 % of the recording is 91.9 % "eyes open" (class 0).
Models trained on the earlier, more balanced segments then face a severe
distribution shift at test time → the default 0.5 threshold produces too
many class-1 predictions → low binary-F1.

Fixes implemented in v2
  ① class_weight = 'balanced'          (all sklearn models)
  ② Weighted CrossEntropyLoss          (all DL models)
  ③ Threshold optimisation on CV set   (all models)
  ④ Macro-F1 as PRIMARY metric         (both classes weighted equally)
  ⑤ Precision, Recall + Confusion Matrix reported for every test evaluation

────────────────────────────────────────────────────────────────────────────
Split strategies
  S1  Temporal hold-out  70 / 15 / 15
  S2  Temporal hold-out  60 / 20 / 20
  S3  Temporal hold-out  80 / 10 / 10
  S4  Walk-forward (expanding-window) CV — 5 folds  (ML only)
  S5  Sliding-window CV                 — 5 folds  (ML only)

Models
  Classical  LogisticRegression · SVM-RBF · RandomForest
             GradientBoosting  · XGBoost
  Deep       LSTM · CNN-LSTM · EEGTransformer
             EEGNet (Lawhern 2018) · PatchTST_Lite (Nie 2023)
  Meta       Ensemble (random-weight soft-vote, CV-optimised)

Requirements:  pip install numpy pandas scikit-learn xgboost torch
=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import math, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.pipeline import Pipeline

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
print(f"[info] Device : {DEVICE}")
warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CSV_PATH     = "dataset/eeg_data_og.csv"   # ← adjust path if needed
SEQ_LEN      = 64                  # look-back window for DL (samples)
PATCH_SIZE   = 8                   # PatchTST patch width
PATCH_STRIDE = 4                   # PatchTST patch stride
DL_EPOCHS    = 25
DL_BATCH     = 128
DL_LR        = 1e-3
ENS_TRIALS   = 3_000              # random-search trials for ensemble weights
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  LOADING DATA")
print("="*72)

df          = pd.read_csv(CSV_PATH)
FEATURE_COLS = [c for c in df.columns if c != "eyeDetection"]
X_all       = df[FEATURE_COLS].values.astype(np.float32)
y_all       = df["eyeDetection"].values.astype(np.int64)
N, N_FEATURES = X_all.shape

print(f"  Shape : {df.shape}")
print(f"  Global target : {np.bincount(y_all)}  (open / closed)")

# ── Temporal distribution diagnosis ──────────────────────────────────────────
print("\n  ⚑  Temporal class distribution (closed-eye rate per quarter)")
print(f"  {'Segment':20s}  {'Open':>6}  {'Closed':>6}  {'%Closed':>8}")
for q in range(4):
    s, e = q * N // 4, (q + 1) * N // 4
    cnts = np.bincount(y_all[s:e])
    print(f"  Q{q+1} [{s:5d}:{e:5d}]       {cnts[0]:6d}  {cnts[1]:6d}  {cnts[1]/len(y_all[s:e])*100:7.1f}%")
print()
print("  ⚠  Last 10/15/20% are heavily open-dominant → severe distribution")
print("     shift at test time → accuracy paradox and low binary-F1.")
for label, s in [("last_10%", int(N * 0.90)),
                  ("last_15%", int(N * 0.85)),
                  ("last_20%", int(N * 0.80))]:
    cnts = np.bincount(y_all[s:])
    print(f"  {label}:  open={cnts[0]}  closed={cnts[1]}  "
          f"closed_rate={cnts[1]/len(y_all[s:])*100:.1f}%")

# ─── SPLIT UTILITIES  (strict temporal order) ────────────────────────────────

def temporal_three_way_split(X, y, train_frac, cv_frac):
    """Chronological 3-way split. Test = remainder at the end."""
    n  = len(X)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cv_frac))
    return (X[:i1], y[:i1],
            X[i1:i2], y[i1:i2],
            X[i2:],   y[i2:])


def walk_forward_cv_indices(n, n_folds=5, min_train_frac=0.50):
    """Expanding-window walk-forward CV. Future never leaks into training."""
    min_tr = int(n * min_train_frac)
    step   = (n - min_tr) // (n_folds + 1)
    return [(slice(0, min_tr + k * step),
             slice(min_tr + k * step, min_tr + (k + 1) * step))
            for k in range(n_folds)]


def sliding_window_cv_indices(n, n_folds=5, window_frac=0.50):
    """Fixed-size training window slides forward without overlap."""
    win  = int(n * window_frac)
    step = (n - win) // (n_folds + 1)
    return [(slice(k * step, k * step + win),
             slice(k * step + win, min(k * step + win + step, n)))
            for k in range(n_folds)]


# ─── SEQUENCE BUILDER FOR DL (overlapping windows, no leakage) ───────────────

def build_sequences(X_flat, y_flat, seq_len):
    """
    (N, F) → (M, seq_len, F) sliding-window sequences.
    Label for window i = y[i + seq_len]  (next-step prediction).
    Call per partition to prevent cross-boundary leakage.
    """
    Xs, ys = [], []
    for i in range(len(X_flat) - seq_len):
        Xs.append(X_flat[i: i + seq_len])
        ys.append(y_flat[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def make_dl_loaders(X_tr, y_tr, X_cv, y_cv, seq_len, batch):
    """
    Scale on training set only (no leakage).
    Returns loaders, scaler, and class-frequency weights for loss weighting.
    """
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_cv_s   = scaler.transform(X_cv)

    Xs_tr, ys_tr = build_sequences(X_tr_s, y_tr, seq_len)
    Xs_cv, ys_cv = build_sequences(X_cv_s, y_cv, seq_len)

    # Class weights: inverse frequency
    classes, counts = np.unique(ys_tr, return_counts=True)
    class_weights   = torch.tensor(
        [counts.sum() / (len(classes) * c) for c in counts],
        dtype=torch.float32)

    def to_loader(Xs, ys):
        return DataLoader(
            TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
            batch_size=batch, shuffle=False)   # NO SHUFFLE — temporal data

    return (to_loader(Xs_tr, ys_tr),
            to_loader(Xs_cv, ys_cv),
            scaler, class_weights, ys_cv)


# ─── METRICS ─────────────────────────────────────────────────────────────────

def _safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def print_confusion_matrix(y_true, y_pred, indent="    "):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
    print(f"{indent}Confusion Matrix:")
    print(f"{indent}               Pred Open   Pred Closed")
    print(f"{indent}True Open    {tn:9d}   {fp:11d}")
    print(f"{indent}True Closed  {fn:9d}   {tp:11d}")
    print(f"{indent}→ TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def evaluate(y_true, y_pred, y_prob=None, label="", show_cm=True):
    """
    Full metrics report.  Macro-F1 is the PRIMARY metric given temporal
    distribution shift (equally penalises errors on both classes).
    Binary-F1 (positive = eyes-closed) is reported for reference.
    """
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    bf1  = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    auc  = _safe_auc(y_true, y_prob) if y_prob is not None else float("nan")

    print(f"  [{label}]")
    print(f"    Acc={acc:.4f}  MacroF1={mf1:.4f}  BinaryF1={bf1:.4f}"
          f"  Prec(M)={prec:.4f}  Rec(M)={rec:.4f}  AUC={auc:.4f}")
    if show_cm:
        print_confusion_matrix(y_true, y_pred)

    return {"acc": acc, "macro_f1": mf1, "binary_f1": bf1,
            "precision": prec, "recall": rec, "auc": auc}


# ─── THRESHOLD OPTIMISATION (on CV set) ──────────────────────────────────────

def optimize_threshold(y_cv, probs_cv):
    """
    Grid-search over [0.05, 0.95] to find the probability threshold that
    maximises macro-F1 on the CV set.  Applied to test at inference time.
    Avoids the accuracy paradox caused by the default 0.5 threshold when
    the test window has a very different class distribution from training.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs_cv >= t).astype(int)
        score = f1_score(y_cv, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


# ─── ML MODEL FACTORY ────────────────────────────────────────────────────────

def get_ml_models():
    """All sklearn pipelines include class_weight='balanced' to handle imbalance."""
    models = {
        "LogisticRegression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        solver="lbfgs", random_state=RANDOM_SEED))]),
        "SVM_RBF": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=RANDOM_SEED))]),
        "RandomForest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200,
                                           class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_SEED))]),
        "GradientBoosting": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200,
                                                learning_rate=0.1,
                                                max_depth=5,
                                                random_state=RANDOM_SEED))]),
    }
    if HAS_XGB:
        neg = int((y_all == 0).sum())
        pos = int((y_all == 1).sum())
        models["XGBoost"] = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", XGBClassifier(n_estimators=200,
                                   scale_pos_weight=neg / pos,   # handles imbalance
                                   eval_metric="logloss",
                                   random_state=RANDOM_SEED, n_jobs=-1))])
    return models


def run_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # Optimise threshold on CV
    cv_prob   = model.predict_proba(X_cv)[:, 1]
    opt_t, _  = optimize_threshold(y_cv, cv_prob)

    results = {}
    for part_name, Xs, ys in [("CV", X_cv, y_cv), ("Test", X_te, y_te)]:
        prob   = model.predict_proba(Xs)[:, 1]
        thresh = opt_t if part_name == "Test" else 0.5
        pred   = (prob >= thresh).astype(int)
        results[part_name] = evaluate(ys, pred, prob,
                                       label=f"{name}|{part_name}(t={thresh:.2f})",
                                       show_cm=(part_name == "Test"))
        results[part_name]["raw_prob"] = prob

    print(f"    Train time: {train_time:.1f}s   Opt-threshold: {opt_t:.2f}")
    return results, opt_t


# ─── DEEP LEARNING ARCHITECTURES ─────────────────────────────────────────────

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
            nn.Conv1d(n_features, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):                          # x: (B, T, F)
        x = self.cnn(x.permute(0, 2, 1))          # (B, F, T) → conv → (B, 128, T/2)
        x = x.permute(0, 2, 1)                    # back to (B, T/2, 128)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class EEGTransformer(nn.Module):
    """
    EEG-specific Transformer classifier.
    CLS token + sinusoidal PE + pre-LN TransformerEncoder → MLP head.
    """
    def __init__(self, n_features, d_model=64, nhead=4, n_layers=3,
                 dim_ff=128, dropout=0.1, seq_len=64):
        super().__init__()
        self.proj      = nn.Linear(n_features, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=seq_len + 1,
                                             dropout=dropout)
        enc            = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.tf        = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.head      = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):                          # (B, T, F)
        B   = x.size(0)
        x   = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = self.pos_enc(torch.cat([cls, x], dim=1))
        return self.head(self.tf(x)[:, 0])         # CLS output


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018).
    Depthwise + separable 2-D convolutions designed for compact,
    channel-aware EEG classification.

    Input  : (B, T, C)  →  reshaped to (B, 1, C, T) for Conv2d.
    Block 1: Temporal filter across all channels
             + Depthwise spatial filter per electrode.
    Block 2: Separable convolution (channel-wise then point-wise).
    Head   : Linear(flat_size → 2).
    """
    def __init__(self, n_channels=14, T=64, F1=8, D=2, dropout=0.25):
        super().__init__()
        F2     = F1 * D
        kern_t = T // 2    # temporal kernel ≈ half the pseudo-sampling-rate

        self.block1 = nn.Sequential(
            # temporal filter
            nn.Conv2d(1,  F1, (1, kern_t), padding=(0, kern_t // 2), bias=False),
            nn.BatchNorm2d(F1),
            # depthwise spatial filter (one filter per temporal feature map)
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            # depthwise (channel-wise) separable convolution
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),                 # pointwise
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()

        # Dynamically infer classifier input dimension
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, n_channels, T)
            flat_dim = self.flatten(self.block2(self.block1(dummy))).shape[1]
        self.head = nn.Linear(flat_dim, 2)

    def forward(self, x):                          # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)       # → (B, 1, C, T)
        return self.head(self.flatten(self.block2(self.block1(x))))


class PatchTST_Lite(nn.Module):
    """
    Lightweight PatchTST (Nie et al. 2023).
    The time series is divided into overlapping fixed-size patches.
    Each patch is linearly embedded; a Transformer encoder with a
    learnable [CLS] token aggregates global context for classification.

    Advantages over whole-sequence Transformers:
      • Shorter sequence fed to attention (15 patches vs 64 timesteps)
      • Local temporal structure captured within each patch
      • Better positional inductive bias for time series
    """
    def __init__(self, n_features, seq_len=64, patch_size=8, stride=4,
                 d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        n_patches       = (seq_len - patch_size) // stride + 1    # = 15
        patch_dim       = patch_size * n_features                  # = 112

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.drop       = nn.Dropout(dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.tf   = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):                          # x: (B, T, F)
        B   = x.size(0)
        patches = [
            x[:, i: i + self.patch_size, :].reshape(B, -1)
            for i in range(0, x.size(1) - self.patch_size + 1, self.stride)
        ]
        x   = self.patch_proj(torch.stack(patches, dim=1))   # (B, n_p, d)
        cls = self.cls_token.expand(B, -1, -1)
        x   = self.drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x   = self.norm(self.tf(x))
        return self.head(x[:, 0])                 # CLS token


# ─── DL TRAINING UTILITIES ───────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion):
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
def predict_dl(model, loader):
    model.eval(); preds, probs = [], []
    for Xb, _ in loader:
        logits = model(Xb.to(DEVICE))
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_dl(name, model, tr_loader, cv_loader, y_cv_seq,
           X_te, y_te, scaler, seq_len, class_weights):
    """
    Train DL model with weighted loss (handles class imbalance).
    Optimise decision threshold on CV.  Evaluate on temporally isolated test.
    """
    model.to(DEVICE)
    cw        = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=DL_EPOCHS)

    for epoch in range(1, DL_EPOCHS + 1):
        loss = train_epoch(model, tr_loader, optimiser, criterion)
        scheduler.step()
        if epoch % 5 == 0:
            cv_preds, _ = predict_dl(model, cv_loader)
            mf1 = f1_score(y_cv_seq, cv_preds, average="macro", zero_division=0)
            print(f"    Epoch {epoch:3d}/{DL_EPOCHS}  "
                  f"loss={loss:.4f}  cv_macro_f1={mf1:.4f}")

    # ── CV evaluation + threshold optimisation ────────────────────────────────
    cv_preds, cv_probs = predict_dl(model, cv_loader)
    opt_t, _           = optimize_threshold(y_cv_seq, cv_probs)
    cv_pred_opt        = (cv_probs >= opt_t).astype(int)

    results = {}
    results["CV"] = evaluate(y_cv_seq, cv_pred_opt, cv_probs,
                              label=f"{name}|CV(t={opt_t:.2f})", show_cm=False)
    results["CV"]["raw_prob"] = cv_probs

    # ── Test evaluation (build sequences from test partition only) ────────────
    X_te_s         = scaler.transform(X_te)
    Xs_te, ys_te   = build_sequences(X_te_s, y_te, seq_len)
    te_loader      = DataLoader(
        TensorDataset(torch.tensor(Xs_te), torch.tensor(ys_te)),
        batch_size=DL_BATCH, shuffle=False)
    te_preds, te_probs = predict_dl(model, te_loader)
    te_pred_opt        = (te_probs >= opt_t).astype(int)

    results["Test"]          = evaluate(ys_te, te_pred_opt, te_probs,
                                         label=f"{name}|Test(t={opt_t:.2f})",
                                         show_cm=True)
    results["Test"]["raw_prob"] = te_probs
    results["y_te_seq"]         = ys_te

    print(f"    Opt-threshold: {opt_t:.2f}")
    return results, opt_t


# ─── ENSEMBLE OPTIMIZER ──────────────────────────────────────────────────────

class EnsembleOptimizer:
    """
    Weighted soft-voting ensemble across all ML + DL models.

    Strategy
    --------
    1. Collect P(y=1 | x) probability vectors from every base model on the
       CV partition (never touching the test set).
    2. Random-search over the simplex of non-negative, sum-to-1 weight
       vectors (ENS_TRIALS samples) to find the combination that maximises
       macro-F1 on CV.
    3. Apply the found weights to test-partition probabilities and report.

    This directly answers "which combo of algorithms yields the best result
    on each test split?" by showing the optimal weight per model.
    """
    def __init__(self):
        self.best_weights  = None
        self.model_names   = []

    def optimize(self, probs_cv_dict, y_cv, n_trials=ENS_TRIALS):
        self.model_names = list(probs_cv_dict.keys())
        k         = len(self.model_names)
        prob_mat  = np.column_stack([probs_cv_dict[n]
                                     for n in self.model_names])  # (m, k)

        def eval_w(w):
            ens_prob = prob_mat @ w
            preds    = (ens_prob >= 0.5).astype(int)
            return f1_score(y_cv, preds, average="macro", zero_division=0)

        rng = np.random.RandomState(RANDOM_SEED)
        best_w  = np.ones(k) / k
        best_f1 = eval_w(best_w)

        for _ in range(n_trials):
            w = rng.dirichlet(np.ones(k))
            s = eval_w(w)
            if s > best_f1:
                best_f1, best_w = s, w.copy()

        # Additional refinement: try single-model and top-3/5 uniform combos
        for combo_size in [1, 3, 5]:
            top_idx = np.argsort([eval_w(np.eye(k)[i]) for i in range(k)])[::-1]
            for indices in [top_idx[:combo_size]]:
                w = np.zeros(k)
                w[indices] = 1.0 / combo_size
                s = eval_w(w)
                if s > best_f1:
                    best_f1, best_w = s, w.copy()

        self.best_weights = best_w
        return dict(zip(self.model_names, best_w.tolist())), best_f1

    def predict(self, probs_test_dict, threshold=0.5):
        prob_mat = np.column_stack([probs_test_dict[n]
                                    for n in self.model_names])
        ens_prob = prob_mat @ self.best_weights
        return (ens_prob >= threshold).astype(int), ens_prob


# ─── MAIN EXPERIMENT ─────────────────────────────────────────────────────────
SPLIT_CONFIGS = [
    ("70/15/15", 0.70, 0.15),
    ("60/20/20", 0.60, 0.20),
    ("80/10/10", 0.80, 0.10),
]

summary_rows = []


for split_label, tr_frac, cv_frac in SPLIT_CONFIGS:
    print("\n" + "="*72)
    print(f"  HOLD-OUT SPLIT  {split_label}  "
          f"(train={tr_frac:.0%} | cv={cv_frac:.0%} | "
          f"test={1-tr_frac-cv_frac:.0%})")
    print("="*72)

    X_tr, y_tr, X_cv, y_cv, X_te, y_te = temporal_three_way_split(
        X_all, y_all, tr_frac, cv_frac)

    tr_closed  = y_tr.mean()
    cv_closed  = y_cv.mean()
    te_closed  = y_te.mean()
    print(f"  Sizes   → Train:{len(X_tr)}  CV:{len(X_cv)}  Test:{len(X_te)}")
    print(f"  Closed% → Train:{tr_closed:.1%}  CV:{cv_closed:.1%}  "
          f"Test:{te_closed:.1%}  "
          f"← distribution shift = {abs(tr_closed-te_closed):.1%}")

    # Storage for ensemble
    ml_cv_probs  = {}
    ml_te_probs  = {}
    dl_cv_probs  = {}
    dl_te_probs  = {}
    y_te_seq_ref = None

    # ── Classical ML ─────────────────────────────────────────────────────────
    print("\n  ── Classical ML ──")
    for name, model in get_ml_models().items():
        print(f"\n  {name}")
        res, opt_t = run_ml(name, model,
                             X_tr, y_tr, X_cv, y_cv, X_te, y_te)
        ml_cv_probs[name] = res["CV"]["raw_prob"]
        ml_te_probs[name] = res["Test"]["raw_prob"]

        for part in ["CV", "Test"]:
            row = {"split": split_label, "model": name, "partition": part,
                   "threshold": opt_t if part == "Test" else 0.5}
            row.update({k: v for k, v in res[part].items()
                        if k != "raw_prob"})
            summary_rows.append(row)

    # ── Deep Learning ─────────────────────────────────────────────────────────
    print("\n  ── Deep Learning ──")

    dl_model_factories = {
        "LSTM":          lambda: LSTMClassifier(N_FEATURES),
        "CNN_LSTM":      lambda: CNNLSTMClassifier(N_FEATURES),
        "EEGTransformer":lambda: EEGTransformer(N_FEATURES, d_model=64,
                                                  nhead=4, n_layers=3,
                                                  seq_len=SEQ_LEN),
        "EEGNet":        lambda: EEGNet(n_channels=N_FEATURES, T=SEQ_LEN),
        "PatchTST_Lite": lambda: PatchTST_Lite(N_FEATURES, seq_len=SEQ_LEN,
                                                 patch_size=PATCH_SIZE,
                                                 stride=PATCH_STRIDE),
    }

    for name, factory in dl_model_factories.items():
        print(f"\n  {name}")
        (tr_loader, cv_loader,
         scaler, cw, y_cv_seq) = make_dl_loaders(
            X_tr, y_tr, X_cv, y_cv, SEQ_LEN, DL_BATCH)

        model = factory()
        res, opt_t = run_dl(name, model, tr_loader, cv_loader, y_cv_seq,
                             X_te, y_te, scaler, SEQ_LEN, cw)

        dl_cv_probs[name] = res["CV"]["raw_prob"]
        dl_te_probs[name] = res["Test"]["raw_prob"]
        if y_te_seq_ref is None:
            y_te_seq_ref = res["y_te_seq"]

        for part in ["CV", "Test"]:
            row = {"split": split_label, "model": name, "partition": part,
                   "threshold": opt_t if part == "Test" else 0.5}
            row.update({k: v for k, v in res[part].items()
                        if k not in ("raw_prob", "y_te_seq")})
            summary_rows.append(row)

    # ── Align probabilities for ensemble ─────────────────────────────────────
    # DL predictions cover indices [SEQ_LEN:] of y_cv / y_te.
    # ML predictions cover [0:n] of y_cv / y_te.
    # Align by trimming ML to [SEQ_LEN:] so all vectors are the same length.
    y_cv_ens = y_cv[SEQ_LEN:]
    y_te_ens = y_te_seq_ref             # already y_te[SEQ_LEN:] from build_sequences

    combined_cv = {**{k: v[SEQ_LEN:] for k, v in ml_cv_probs.items()},
                   **dl_cv_probs}
    combined_te = {**{k: v[SEQ_LEN:] for k, v in ml_te_probs.items()},
                   **dl_te_probs}

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print(f"\n  ── Ensemble (random-weight soft-vote, {ENS_TRIALS} trials) ──")
    ens = EnsembleOptimizer()
    best_weights, cv_ens_f1 = ens.optimize(combined_cv, y_cv_ens)

    print(f"\n  Optimal weights  (CV macro-F1 = {cv_ens_f1:.4f})")
    print(f"  {'Model':30s}  {'Weight':>8}  {'Contribution':>13}")
    for mname, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 30)
        print(f"  {mname:30s}  {w:8.4f}  {bar}")

    # Optimise ensemble threshold on CV
    ens_cv_prob_vec = (np.column_stack([combined_cv[n] for n in ens.model_names])
                       @ ens.best_weights)
    opt_t_ens, _ = optimize_threshold(y_cv_ens, ens_cv_prob_vec)

    te_ens_pred, te_ens_prob = ens.predict(combined_te, threshold=opt_t_ens)

    print(f"\n  ENSEMBLE TEST RESULT  (split={split_label}, t={opt_t_ens:.2f})")
    ens_res = evaluate(y_te_ens, te_ens_pred, te_ens_prob,
                        label=f"Ensemble|Test", show_cm=True)
    row = {"split": split_label, "model": "Ensemble", "partition": "Test",
           "threshold": opt_t_ens}
    row.update(ens_res)
    summary_rows.append(row)


# ─── TEMPORAL CV (ML only — DL per-fold training is expensive) ───────────────

print("\n" + "="*72)
print("  WALK-FORWARD CV (Expanding Window) — 5 Folds")
print("="*72)

wf_agg = defaultdict(list)
for fi, (tr_sl, val_sl) in enumerate(walk_forward_cv_indices(N)):
    X_tr, y_tr   = X_all[tr_sl], y_all[tr_sl]
    X_val, y_val = X_all[val_sl], y_all[val_sl]
    print(f"\n  Fold {fi+1}  train={len(X_tr)}  val={len(X_val)}"
          f"  val_closed_rate={y_val.mean():.2%}")
    for name, model in get_ml_models().items():
        model.fit(X_tr, y_tr)
        prob  = model.predict_proba(X_val)[:, 1]
        opt_t, _ = optimize_threshold(y_val, prob)
        pred  = (prob >= opt_t).astype(int)
        m     = evaluate(y_val, pred, prob,
                          label=f"{name}|fold{fi+1}(t={opt_t:.2f})",
                          show_cm=False)
        m["opt_t"] = opt_t
        wf_agg[name].append(m)

print("\n  Walk-Forward CV — Mean ± Std  (primary: MacroF1)")
print(f"  {'Model':30s}  {'MacroF1':>14}  {'Acc':>14}  {'AUC':>14}")
for name, folds in wf_agg.items():
    mf1s = [f["macro_f1"] for f in folds]
    accs = [f["acc"]      for f in folds]
    aucs = [f["auc"] if not math.isnan(f["auc"]) else 0.0 for f in folds]
    print(f"  {name:30s}  "
          f"{np.mean(mf1s):.4f}±{np.std(mf1s):.4f}  "
          f"{np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}")


print("\n" + "="*72)
print("  SLIDING-WINDOW CV (Fixed-Size Window) — 5 Folds")
print("="*72)

sw_agg = defaultdict(list)
for fi, (tr_sl, val_sl) in enumerate(sliding_window_cv_indices(N)):
    X_tr, y_tr   = X_all[tr_sl], y_all[tr_sl]
    X_val, y_val = X_all[val_sl], y_all[val_sl]
    print(f"\n  Fold {fi+1}  train={len(X_tr)}  val={len(X_val)}"
          f"  val_closed_rate={y_val.mean():.2%}")
    for name, model in get_ml_models().items():
        model.fit(X_tr, y_tr)
        prob  = model.predict_proba(X_val)[:, 1]
        opt_t, _ = optimize_threshold(y_val, prob)
        pred  = (prob >= opt_t).astype(int)
        m     = evaluate(y_val, pred, prob,
                          label=f"{name}|fold{fi+1}(t={opt_t:.2f})",
                          show_cm=False)
        sw_agg[name].append(m)

print("\n  Sliding-Window CV — Mean ± Std")
print(f"  {'Model':30s}  {'MacroF1':>14}  {'Acc':>14}  {'AUC':>14}")
for name, folds in sw_agg.items():
    mf1s = [f["macro_f1"] for f in folds]
    accs = [f["acc"]      for f in folds]
    aucs = [f["auc"] if not math.isnan(f["auc"]) else 0.0 for f in folds]
    print(f"  {name:30s}  "
          f"{np.mean(mf1s):.4f}±{np.std(mf1s):.4f}  "
          f"{np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}")


# ─── FINAL SUMMARY TABLE ──────────────────────────────────────────────────────
print("\n" + "="*72)
print("  FINAL SUMMARY — Test Partition, All Hold-Out Splits")
print("  Primary metric: Macro-F1  (equally weights both eye states)")
print("="*72)

cols = ["split", "model", "acc", "macro_f1", "binary_f1",
        "precision", "recall", "auc", "threshold"]
df_sum = pd.DataFrame(summary_rows)
df_sum = df_sum[df_sum["partition"] == "Test"].copy()
for c in cols:
    if c not in df_sum.columns:
        df_sum[c] = float("nan")

df_sum = df_sum.sort_values(["split", "macro_f1"], ascending=[True, False])
print(df_sum[cols].to_string(index=False,
                              float_format=lambda x: f"{x:.4f}"))

print("\n✓  Pipeline complete.")