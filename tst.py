# pip install numpy pandas scikit-learn xgboost torch
"""
=============================================================================
EEG Eye-State Detection — Full ML / DL Pipeline
=============================================================================
Dataset  : EEG Eye State (14 electrodes, 14,979 samples @ ~128 Hz)
Target   : eyeDetection  (0 = eyes open, 1 = eyes closed)
Nature   : Temporal / time-series  — NO shuffling, NO data leakage

Split strategies implemented
────────────────────────────
  S1  Temporal hold-out  70 / 15 / 15  (train / CV / test)
  S2  Temporal hold-out  60 / 20 / 20
  S3  Temporal hold-out  80 / 10 / 10
  S4  Walk-forward (expanding-window) cross-validation  — 5 folds
  S5  Sliding-window cross-validation                   — 5 folds

Models implemented
──────────────────
  ML  1. Logistic Regression
  ML  2. SVM (RBF kernel)
  ML  3. Random Forest
  ML  4. Gradient Boosting (sklearn)
  ML  5. XGBoost
  DL  6. LSTM
  DL  7. CNN-LSTM
  DL  8. EEG Time-Series Transformer

Requirements
────────────
  pip install numpy pandas scikit-learn xgboost torch
=============================================================================
"""

import math
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not found — XGBoost model will be skipped.")

# ── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {DEVICE}")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH    = "dataset/eeg_data_og.csv"     # adjust path as needed
SEQ_LEN     = 64                    # look-back window for DL models (samples)
DL_EPOCHS   = 20
DL_BATCH    = 128
DL_LR       = 1e-3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  LOADING DATA")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"Shape : {df.shape}")
print(f"Target distribution:\n{df['eyeDetection'].value_counts()}")

FEATURE_COLS = [c for c in df.columns if c != "eyeDetection"]
TARGET_COL   = "eyeDetection"

X_all = df[FEATURE_COLS].values.astype(np.float32)   # (N, 14)
y_all = df[TARGET_COL].values.astype(np.int64)        # (N,)
N     = len(X_all)


# ─────────────────────────────────────────────────────────────────────────────
#  2. SPLIT UTILITIES  (NO SHUFFLE — temporal order preserved)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_three_way_split(X, y, train_frac, cv_frac):
    """
    Splits data preserving temporal order into train / cv / test.
    test_frac = 1 - train_frac - cv_frac  (remainder at the end).
    """
    n = len(X)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cv_frac))
    return (X[:i1],  y[:i1],
            X[i1:i2], y[i1:i2],
            X[i2:],   y[i2:])


def walk_forward_cv_indices(n, n_folds=5, min_train_frac=0.5):
    """
    Expanding-window walk-forward CV.
    Fold k uses indices [0 .. cutk) for training and [cutk .. cutk+step) for val.
    min_train_frac ensures the initial training window is meaningful.
    """
    min_train = int(n * min_train_frac)
    remaining  = n - min_train
    step       = remaining // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * step
        val_end   = train_end + step
        folds.append((slice(0, train_end), slice(train_end, val_end)))
    return folds


def sliding_window_cv_indices(n, n_folds=5, window_frac=0.5):
    """
    Sliding-window CV: training window of fixed size slides forward.
    No data from the future leaks into each training window.
    """
    win  = int(n * window_frac)
    step = (n - win) // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        start     = k * step
        train_end = start + win
        val_end   = min(train_end + step, n)
        folds.append((slice(start, train_end), slice(train_end, val_end)))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
#  3. SEQUENCE BUILDING FOR DL (sliding window, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(X_flat, y_flat, seq_len):
    """
    Converts flat (N, F) array into (M, seq_len, F) overlapping windows.
    Label for window i is y[i + seq_len]  (predict the *next* sample).
    Windows never cross the boundary — caller is responsible for passing
    only the relevant partition to avoid leakage.
    """
    Xs, ys = [], []
    for i in range(len(X_flat) - seq_len):
        Xs.append(X_flat[i: i + seq_len])
        ys.append(y_flat[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def make_dl_loaders(X_tr, y_tr, X_cv, y_cv, seq_len, batch):
    """Scale on train, apply to cv/test; build sequences; return DataLoaders."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cv_s = scaler.transform(X_cv)

    Xs_tr, ys_tr = build_sequences(X_tr_s, y_tr, seq_len)
    Xs_cv, ys_cv = build_sequences(X_cv_s, y_cv, seq_len)

    def to_loader(Xs, ys, shuffle=False):
        ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    # shuffle=False — temporal order intact even during DL training batches
    return (to_loader(Xs_tr, ys_tr, shuffle=False),
            to_loader(Xs_cv, ys_cv, shuffle=False),
            scaler)


# ─────────────────────────────────────────────────────────────────────────────
#  4. METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob=None, label=""):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="binary")
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print(f"  [{label}]  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return {"acc": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
#  5. ML MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_models():
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000,
                                          solver="lbfgs",
                                          random_state=RANDOM_SEED))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True,
                           random_state=RANDOM_SEED))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200,
                                              n_jobs=-1,
                                              random_state=RANDOM_SEED))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200,
                                                   learning_rate=0.1,
                                                   max_depth=5,
                                                   random_state=RANDOM_SEED))
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(n_estimators=200,
                                     use_label_encoder=False,
                                     eval_metric="logloss",
                                     random_state=RANDOM_SEED,
                                     n_jobs=-1))
        ])
    return models


def run_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    results = {}
    for split_name, Xs, ys in [("CV",   X_cv, y_cv),
                                ("Test", X_te, y_te)]:
        pred = model.predict(Xs)
        prob = model.predict_proba(Xs)[:, 1]
        results[split_name] = evaluate(ys, pred, prob,
                                        label=f"{name} | {split_name}")
    print(f"    Train time: {train_time:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  6. DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Stacked bidirectional LSTM → global average pool → classifier head."""
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):                     # x: (B, T, F)
        out, _ = self.lstm(x)                 # (B, T, 2H)
        pooled = out.mean(dim=1)              # (B, 2H)
        return self.head(pooled)              # (B, 2)


class CNNLSTMClassifier(nn.Module):
    """1-D conv feature extractor followed by a single LSTM layer."""
    def __init__(self, n_features, seq_len, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):                     # x: (B, T, F)
        x = x.permute(0, 2, 1)               # (B, F, T) for Conv1d
        x = self.cnn(x)                       # (B, 128, T/2)
        x = x.permute(0, 2, 1)               # (B, T/2, 128)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)              # (B, 128)
        return self.head(pooled)


# ── Transformer ───────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):                    # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class EEGTransformer(nn.Module):
    """
    EEG Time-Series Transformer.
    Architecture:
      Input projection  → Positional Encoding
      → N × TransformerEncoderLayer
      → Global Average Pooling
      → MLP classifier head
    """
    def __init__(self, n_features, d_model=64, nhead=4,
                 n_layers=3, dim_ff=128, dropout=0.1, seq_len=64):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 1,
                                              dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True     # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Learnable [CLS] token — aggregate global sequence information
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):                    # x: (B, T, F)
        B  = x.size(0)
        x  = self.input_proj(x)              # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1)      # (B, T+1, d_model)
        x  = self.pos_enc(x)
        x  = self.transformer(x)             # (B, T+1, d_model)
        cls_out = x[:, 0]                    # take [CLS] representation
        return self.head(cls_out)            # (B, 2)


# ── Training / inference loop ─────────────────────────────────────────────────

def train_dl_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_dl(model, loader):
    model.eval()
    all_preds, all_probs = [], []
    for Xb, _ in loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs)
    return np.concatenate(all_preds), np.concatenate(all_probs)


def run_dl(name, model, train_loader, cv_loader, X_te, y_te, scaler, seq_len):
    """Train DL model, evaluate on CV; build test sequences and evaluate."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=DL_LR,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=DL_EPOCHS)

    best_cv_f1, best_state = 0.0, None
    for epoch in range(1, DL_EPOCHS + 1):
        loss = train_dl_epoch(model, train_loader, optimiser, criterion)
        scheduler.step()
        if epoch % 5 == 0:
            preds, probs = predict_dl(model, cv_loader)
            # recover cv labels from loader
            cv_labels = torch.cat([yb for _, yb in cv_loader]).numpy()
            f1 = f1_score(cv_labels, preds, average="binary")
            print(f"    Epoch {epoch:3d}/{DL_EPOCHS}  loss={loss:.4f}  "
                  f"cv_f1={f1:.4f}")
            if f1 > best_cv_f1:
                best_cv_f1   = f1
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    results = {}
    # CV evaluation
    cv_preds, cv_probs = predict_dl(model, cv_loader)
    cv_labels = torch.cat([yb for _, yb in cv_loader]).numpy()
    results["CV"] = evaluate(cv_labels, cv_preds, cv_probs,
                              label=f"{name} | CV")

    # Test evaluation — build sequences from test partition (no leakage)
    X_te_s  = scaler.transform(X_te)
    Xs_te, ys_te = build_sequences(X_te_s, y_te, seq_len)
    te_ds   = TensorDataset(torch.tensor(Xs_te), torch.tensor(ys_te))
    te_loader = DataLoader(te_ds, batch_size=DL_BATCH, shuffle=False)
    te_preds, te_probs = predict_dl(model, te_loader)
    results["Test"] = evaluate(ys_te, te_preds, te_probs,
                                label=f"{name} | Test")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  7. MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES = X_all.shape[1]          # 14

# ════════════════════════════════════════════════════════════════════════════
#  PART A — THREE-WAY HOLD-OUT SPLITS
# ════════════════════════════════════════════════════════════════════════════

SPLIT_CONFIGS = [
    ("70/15/15", 0.70, 0.15),
    ("60/20/20", 0.60, 0.20),
    ("80/10/10", 0.80, 0.10),
]

all_results = defaultdict(dict)      # {split_label: {model_name: results}}

for split_label, tr_frac, cv_frac in SPLIT_CONFIGS:
    print("\n" + "="*70)
    print(f"  HOLD-OUT SPLIT  {split_label}  (train={tr_frac:.0%}  "
          f"cv={cv_frac:.0%}  test={1-tr_frac-cv_frac:.0%})")
    print("="*70)

    X_tr, y_tr, X_cv, y_cv, X_te, y_te = temporal_three_way_split(
        X_all, y_all, tr_frac, cv_frac)

    print(f"  Train:{len(X_tr)}  CV:{len(X_cv)}  Test:{len(X_te)}")

    # ── ML models ────────────────────────────────────────────────────────
    print("\n  ── Classical ML ──")
    for name, model in get_ml_models().items():
        print(f"\n  {name}")
        res = run_ml(name, model, X_tr, y_tr, X_cv, y_cv, X_te, y_te)
        all_results[split_label][name] = res

    # ── DL models ────────────────────────────────────────────────────────
    print("\n  ── Deep Learning ──")

    dl_models = {
        "LSTM":        LSTMClassifier(N_FEATURES),
        "CNN_LSTM":    CNNLSTMClassifier(N_FEATURES, SEQ_LEN),
        "EEGTransformer": EEGTransformer(N_FEATURES, d_model=64, nhead=4,
                                          n_layers=3, seq_len=SEQ_LEN),
    }

    for name, model in dl_models.items():
        print(f"\n  {name}")
        train_loader, cv_loader, scaler = make_dl_loaders(
            X_tr, y_tr, X_cv, y_cv, SEQ_LEN, DL_BATCH)
        res = run_dl(name, model, train_loader, cv_loader,
                     X_te, y_te, scaler, SEQ_LEN)
        all_results[split_label][name] = res


# ════════════════════════════════════════════════════════════════════════════
#  PART B — TEMPORAL CROSS-VALIDATION (Walk-Forward & Sliding-Window)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  TEMPORAL CROSS-VALIDATION — Walk-Forward (Expanding Window)")
print("="*70)

wf_folds  = walk_forward_cv_indices(N, n_folds=5, min_train_frac=0.50)
wf_results = defaultdict(list)      # {model_name: [fold_metrics, ...]}

for fold_i, (tr_sl, val_sl) in enumerate(wf_folds):
    X_tr, y_tr = X_all[tr_sl], y_all[tr_sl]
    X_val, y_val = X_all[val_sl], y_all[val_sl]
    print(f"\n  Fold {fold_i+1}  train={len(X_tr)}  val={len(X_val)}")

    for name, model in get_ml_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, pred, prob, label=f"{name}|fold{fold_i+1}")
        wf_results[name].append(m)

print("\n  Walk-Forward CV — Mean ± Std across folds")
for name, folds in wf_results.items():
    accs = [f["acc"] for f in folds]
    f1s  = [f["f1"]  for f in folds]
    aucs = [f["auc"] for f in folds]
    print(f"  {name:30s}  Acc={np.mean(accs):.4f}±{np.std(accs):.4f}"
          f"  F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"
          f"  AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")


print("\n" + "="*70)
print("  TEMPORAL CROSS-VALIDATION — Sliding Window")
print("="*70)

sw_folds   = sliding_window_cv_indices(N, n_folds=5, window_frac=0.50)
sw_results = defaultdict(list)

for fold_i, (tr_sl, val_sl) in enumerate(sw_folds):
    X_tr, y_tr   = X_all[tr_sl], y_all[tr_sl]
    X_val, y_val = X_all[val_sl], y_all[val_sl]
    print(f"\n  Fold {fold_i+1}  train={len(X_tr)}  val={len(X_val)}")

    for name, model in get_ml_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, pred, prob, label=f"{name}|fold{fold_i+1}")
        sw_results[name].append(m)

print("\n  Sliding-Window CV — Mean ± Std across folds")
for name, folds in sw_results.items():
    accs = [f["acc"] for f in folds]
    f1s  = [f["f1"]  for f in folds]
    aucs = [f["auc"] for f in folds]
    print(f"  {name:30s}  Acc={np.mean(accs):.4f}±{np.std(accs):.4f}"
          f"  F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"
          f"  AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  PART C — FINAL SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  FINAL SUMMARY — Hold-Out Splits (Test partition)")
print("="*70)
print(f"  {'Model':30s}  {'Split':10s}  {'Acc':>8}  {'F1':>8}  {'AUC':>8}")
print("  " + "-"*65)

for split_label, models in all_results.items():
    for name, res in models.items():
        te = res.get("Test", {})
        print(f"  {name:30s}  {split_label:10s}"
              f"  {te.get('acc', float('nan')):8.4f}"
              f"  {te.get('f1',  float('nan')):8.4f}"
              f"  {te.get('auc', float('nan')):8.4f}")

print("\nDone.\n")