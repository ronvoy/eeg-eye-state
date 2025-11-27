#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG dimensionality‑reduction benchmark:
PCA, ICA, LDA, CSP, t‑SNE and (if available) UMAP.
All results are saved as plots in the ./fig folder and a summary
table is printed on the console.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# --------------------------------------------------------------
# Optional imports (UMAP). If missing we skip the method gracefully.
# --------------------------------------------------------------
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError as e:
    UMAP_AVAILABLE = False
    warnings.warn(f"UMAP could not be imported – skipping UMAP embeddings.\n{e}")

# --------------------------------------------------------------
# CSP implementation (binary classification only)
# --------------------------------------------------------------
def csp_transform(X, y):
    """
    Simple CSP (Common Spatial Patterns) implementation for binary labels.
    Returns a 2‑dimensional projection (first & last CSP filters).
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Zero‑mean, preferably unit‑variance data.
    y : ndarray of shape (n_samples,)
        Binary class labels (two unique values).
    Returns
    -------
    Z : ndarray of shape (n_samples, 2)
        CSP projection (most discriminative + least discriminative).
    """
    # Ensure exactly two classes
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("CSP requires exactly two classes.")
    # Split data
    mask0 = (y == classes[0])
    mask1 = ~mask0
    X0 = X[mask0]
    X1 = X[mask1]

    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("At least one class has zero samples.")

    # Class covariances (unbiased estimator)
    C0 = X0.T @ X0 / X0.shape[0]
    C1 = X1.T @ X1 / X1.shape[0]

    # Composite covariance
    C = C0 + C1

    # Whitening transform via eigen‑decomposition of C
    vals, vecs = np.linalg.eigh(C)
    # Protect against tiny/negative eigenvalues
    vals = np.maximum(vals, 1e-12)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    # Whiten class‑0 covariance
    C0_w = W.T @ C0 @ W

    # Eigenvectors of whitened class‑0 covariance
    eig_vals, eig_vecs = np.linalg.eigh(C0_w)
    # Sort descending (largest variance for class‑0 first)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # Spatial filters
    filters = W @ eig_vecs

    # Project original (zero‑mean) data
    Z = X @ filters

    # Return the most discriminative (first) and least discriminative (last) components
    return np.column_stack((Z[:, 0], Z[:, -1]))


# --------------------------------------------------------------
# Helper: correlation heatmap (matplotlib only – no seaborn)
# --------------------------------------------------------------
def plot_correlation_heatmap(corr_df, save_path):
    """Draw and save a correlation matrix heatmap using pure matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_df.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(corr_df.shape[1]))
    ax.set_yticks(np.arange(corr_df.shape[0]))
    ax.set_xticklabels(corr_df.columns, rotation=90, ha='right')
    ax.set_yticklabels(corr_df.index)
    plt.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------
def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    data_path = 'dataset/eeg_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please place the CSV file in the "
            "dataset/ folder or adjust the path."
        )
    df = pd.read_csv(data_path)

    # Quick sanity checks
    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")
    target_col = 'eyeDetection'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from the CSV.")

    # Drop rows with any NaN (EEG data should be clean)
    df = df.dropna()

    # -------------------------
    # 2. Correlation analysis & feature selection
    # -------------------------
    feature_cols = [c for c in df.columns if c != target_col]
    corr_with_target = df[feature_cols].apply(
        lambda x: x.corr(df[target_col]), axis=0
    )
    THRESHOLD = 0.30  # absolute correlation cut‑off
    selected_features = [
        feat for feat, corr in zip(feature_cols, corr_with_target)
        if abs(corr) >= THRESHOLD
    ]

    if len(selected_features) < 2:
        print(
            f"Only {len(selected_features)} feature(s) survived the correlation "
            f"threshold ({THRESHOLD}). Falling back to all {len(feature_cols)} "
            "features."
        )
        selected_features = feature_cols

    print(f"Selected features after correlation thresholding: {selected_features}")

    # Plot full correlation matrix (including target) – saved for inspection
    os.makedirs('fig', exist_ok=True)
    corr_full = df.corr()
    plot_correlation_heatmap(corr_full, 'fig/correlation_heatmap.png')

    # -------------------------
    # 3. Prepare X, y and scale
    # -------------------------
    X = df[selected_features].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure we have at least two classes for supervised methods
    uniq_y = np.unique(y)
    if len(uniq_y) < 2:
        raise ValueError(
            "The dataset contains only one class label. Supervised methods "
            "(LDA, CSP) and classification metrics require at least two classes."
        )

    # -------------------------
    # 4. Define embedding methods
    # -------------------------
    methods = {
        "PCA": lambda X, y: PCA(n_components=2, random_state=42).fit_transform(X),
        "ICA": lambda X, y: FastICA(
            n_components=2, random_state=42, whiten='unit-variance'
        ).fit_transform(X),
        # LDA for binary data yields a single component; we keep it 1‑D
        "LDA": lambda X, y: LinearDiscriminantAnalysis(
            n_components=1
        ).fit_transform(X, y).reshape(-1, 1),
        "CSP": lambda X, y: csp_transform(X, y),
        "tSNE": lambda X, y: TSNE(
            n_components=2, random_state=42, perplexity=30, n_iter=1000
        ).fit_transform(X),
    }
    if UMAP_AVAILABLE:
        methods["UMAP"] = lambda X, y: umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        ).fit_transform(X)

    # -------------------------
    # 5. Run each method, compute metrics, plot
    # -------------------------
    results = []
    knn = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, func in methods.items():
        print(f"\nRunning {name} ...")
        start = time.time()

        try:
            if name in ("LDA", "CSP"):
                emb = func(X_scaled, y)
            else:
                emb = func(X_scaled, None)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            continue

        elapsed = time.time() - start
        print(f"  -> completed in {elapsed:.2f}s, shape={emb.shape}")

        # ------------------------------------------------------------------
        # 5.1 Quantitative scores (silhouette, DB, CH, KNN‑accuracy)
        # ------------------------------------------------------------------
        # For LDA we have only 1 dimension – silhouette etc. still work.
        try:
            sil = silhouette_score(emb, y, metric='euclidean')
        except Exception as e:
            sil = np.nan
            warnings.warn(f"Silhouette failed for {name}: {e}")

        try:
            db = davies_bouldin_score(emb, y)
        except Exception as e:
            db = np.nan
            warnings.warn(f"Davies‑Bouldin failed for {name}: {e}")

        try:
            ch = calinski_harabasz_score(emb, y)
        except Exception as e:
            ch = np.nan
            warnings.warn(f"Calinski‑Harabasz failed for {name}: {e}")

        # KNN cross‑validated accuracy
        try:
            acc = cross_val_score(knn, emb, y, cv=cv, scoring='accuracy').mean()
        except Exception as e:
            acc = np.nan
            warnings.warn(f"KNN CV failed for {name}: {e}")

        # ------------------------------------------------------------------
        # 5.2 Plot 2‑D scatter (or 1‑D strip for LDA)
        # ------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        if emb.shape[1] >= 2:
            plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap='coolwarm', alpha=0.6,
                        edgecolors='k', s=40)
            plt.xlabel(f'{name} Component 1')
            plt.ylabel(f'{name} Component 2')
        else:  # LDA – 1‑D case
            plt.scatter(emb[:, 0], np.zeros_like(emb[:, 0]), c=y,
                        cmap='coolwarm', alpha=0.6, edgecolors='k', s=40)
            plt.yticks([])
            plt.xlabel(f'{name} Component 1 (1‑D projection)')

        plt.title(f'{name} – 2‑D embedding (colored by eyeDetection)')
        plt.colorbar(label='eyeDetection')
        plt.tight_layout()
        plt.savefig(f'fig/{name}_embedding.png', dpi=150)
        plt.close()

        # ------------------------------------------------------------------
        # 5.3 Store results
        # ------------------------------------------------------------------
        results.append({
            "Method": name,
            "Time (s)": elapsed,
            "Silhouette": sil,
            "Davies‑Bouldin": db,
            "Calinski‑Harabasz": ch,
            "KNN‑Acc (5‑CV)": acc,
            "Embedding shape": emb.shape,
        })

    # -------------------------
    # 6. Print summary table
    # -------------------------
    results_df = pd.DataFrame(results)
    print("\n=== Summary of embeddings ===")
    print(results_df.to_string(index=False))

    # Optionally save the table to CSV
    results_df.to_csv('fig/embedding_summary.csv', index=False)
    print("\nAll plots saved to the 'fig' folder.")


if __name__ == "__main__":
    main()