"""
EEG dimensionality-reduction & evaluation pipeline
Saves plots to ./fig and a summary CSV at fig/analysis_summary.csv

Requirements:
  pip install numpy pandas matplotlib scikit-learn umap-learn   # umap-learn optional
Run: python eeg_dimred_analysis.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional supervised methods (only used if >=2 classes)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

# ---------------------------
# Helper functions
# ---------------------------
def ensure_dataset(csv_path, sample_text):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(sample_text)

def drop_highly_correlated(df_features, threshold=0.95):
    """Return selected_features (keeps first when correlated) and dropped list."""
    corr = df_features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [c for c in df_features.columns if c not in to_drop]
    return selected, to_drop, corr

def compute_csp(X_arr, y_arr, n_components=2, reg=1e-10):
    """Simple 2-class CSP implementation. Returns (X_proj, W)
       X_arr: (n_samples, n_features), y_arr: label array with exactly 2 unique labels
    """
    classes = np.unique(y_arr)
    if classes.shape[0] != 2:
        raise ValueError("CSP requires exactly 2 classes.")
    X0 = X_arr[y_arr == classes[0]]
    X1 = X_arr[y_arr == classes[1]]
    def cov_norm(X):
        C = np.cov(X, rowvar=False)
        return C / np.trace(C)
    C0 = cov_norm(X0) + reg * np.eye(X_arr.shape[1])
    C1 = cov_norm(X1) + reg * np.eye(X_arr.shape[1])
    Csum = C0 + C1
    M = np.linalg.pinv(Csum).dot(C0)
    eigvals, eigvecs = np.linalg.eig(M)
    ix = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, ix]
    if n_components % 2 == 0:
        left = list(range(n_components // 2))
        right = list(range(-n_components // 2, 0))
        pick = left + right
    else:
        pick = list(range(n_components))
    W = eigvecs[:, pick]
    Xcsp = X_arr.dot(W)
    return Xcsp, W

def evaluate_embedding(X_emb, y_true):
    """Return dictionary of metrics (or NaN if insufficient classes)."""
    metrics = {}
    if len(np.unique(y_true)) >= 2:
        try:
            metrics['silhouette'] = float(silhouette_score(X_emb, y_true))
            metrics['davies_bouldin'] = float(davies_bouldin_score(X_emb, y_true))
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_emb, y_true))
        except Exception:
            metrics['silhouette'] = metrics['davies_bouldin'] = metrics['calinski_harabasz'] = np.nan
        # KNN CV
        try:
            knn = KNeighborsClassifier(n_neighbors=5)
            cv = StratifiedKFold(n_splits=min(5, len(y_true)))
            scores = cross_val_score(knn, X_emb, y_true, cv=cv)
            metrics['knn_cv_mean'] = float(np.mean(scores))
            metrics['knn_cv_std'] = float(np.std(scores))
        except Exception:
            metrics['knn_cv_mean'] = metrics['knn_cv_std'] = np.nan
    else:
        metrics['silhouette'] = metrics['davies_bouldin'] = metrics['calinski_harabasz'] = np.nan
        metrics['knn_cv_mean'] = metrics['knn_cv_std'] = np.nan
    return metrics

# ---------------------------
# Main procedure
# ---------------------------
def main(csv_path="dataset/eeg_data.csv", corr_threshold=0.95, random_state=42):
    # sample text (used only if file missing)
    sample_csv = """AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4,eyeDetection
4316.41,4013.85,4267.69,4124.62,4342.56,4589.74,4089.23,4616.41,4200.51,4232.82,4213.33,4281.03,4630.26,4375.9,0
4305.13,4008.72,4259.49,4120.0,4341.03,4595.9,4092.82,4612.31,4199.49,4219.49,4198.46,4261.03,4611.79,4357.95,0
4293.33,3994.87,4254.36,4116.41,4337.44,4596.41,4092.31,4609.74,4193.33,4211.79,4186.67,4252.82,4597.95,4347.69,0
4297.44,3994.36,4258.46,4118.97,4336.92,4594.36,4096.92,4614.36,4193.33,4209.74,4192.31,4260.51,4602.05,4350.77,0
4308.21,4007.18,4268.21,4126.15,4344.62,4595.38,4102.05,4622.56,4205.13,4221.54,4205.13,4271.79,4614.36,4374.87,0
4315.9,4021.03,4277.95,4134.36,4346.15,4591.28,4095.9,4620.0,4208.72,4235.38,4212.31,4280.51,4625.64,4391.79,0
4335.9,4024.62,4281.03,4144.1,4336.41,4591.28,4088.72,4616.41,4202.05,4233.33,4211.79,4274.87,4631.28,4385.64,0
"""
    ensure_dataset(csv_path, sample_csv)
    df = pd.read_csv(csv_path)
    label_col = 'eyeDetection'
    feature_cols = [c for c in df.columns if c != label_col]
    print("Loaded data shape:", df.shape)
    print("Label distribution:\n", df[label_col].value_counts(dropna=False))

    # Drop constant features
    nunique = df[feature_cols].nunique()
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        print("Dropping constant columns:", const_cols)
        feature_cols = [c for c in feature_cols if c not in const_cols]

    X = df[feature_cols].astype(float)
    y = df[label_col].astype(int)

    # Scale
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # Correlation heatmap
    selected_features, dropped, corr = None, None, None
    selected_features, dropped, corr = drop_highly_correlated(Xs, threshold=corr_threshold)
    print("Correlation threshold:", corr_threshold)
    print("Dropped (highly correlated):", dropped)
    print("Selected features:", selected_features)

    os.makedirs("fig", exist_ok=True)
    # plot heatmap
    plt.figure(figsize=(8,6))
    plt.imshow(corr.values, interpolation='nearest', aspect='auto')
    plt.title("Feature correlation matrix (heatmap)")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.index)
    plt.tight_layout()
    plt.savefig("fig/correlation_heatmap.png", dpi=150)
    plt.close()

    X_sel = Xs[selected_features].values

    results = []

    # PCA
    t0 = time.perf_counter()
    pca = PCA(n_components=2, random_state=random_state)
    Xpca = pca.fit_transform(X_sel)
    t_pca = time.perf_counter() - t0
    metrics_pca = evaluate_embedding(Xpca, y)
    pca_loadings = pd.DataFrame(pca.components_.T, index=selected_features, columns=['PC1','PC2'])
    pca_top = {c: list(pca_loadings[c].abs().sort_values(ascending=False).head(2).index) for c in pca_loadings.columns}
    plt.figure(figsize=(6,5))
    plt.scatter(Xpca[:,0], Xpca[:,1])
    plt.title("PCA (2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig("fig/pca_2d.png"); plt.close()
    results.append({'method':'PCA','time_s':t_pca,'n_components':2, **metrics_pca,
                    'top_features_dim1':pca_top['PC1'],'top_features_dim2':pca_top['PC2']})

    # ICA
    t0 = time.perf_counter()
    ica = FastICA(n_components=2, random_state=random_state, max_iter=2000)
    Xica = ica.fit_transform(X_sel)
    t_ica = time.perf_counter() - t0
    metrics_ica = evaluate_embedding(Xica, y)
    try:
        ica_mixing = pd.DataFrame(ica.mixing_, index=selected_features, columns=['IC1','IC2'])
        ica_top = {c: list(ica_mixing[c].abs().sort_values(ascending=False).head(2).index) for c in ica_mixing.columns}
    except Exception:
        ica_top = {'IC1': None, 'IC2': None}
    plt.figure(figsize=(6,5))
    plt.scatter(Xica[:,0], Xica[:,1])
    plt.title("ICA (2D)"); plt.xlabel("IC1"); plt.ylabel("IC2")
    plt.tight_layout(); plt.savefig("fig/ica_2d.png"); plt.close()
    results.append({'method':'ICA','time_s':t_ica,'n_components':2, **metrics_ica,
                    'top_features_dim1':ica_top.get('IC1'),'top_features_dim2':ica_top.get('IC2')})

    # LDA (supervised) - only if >=2 classes
    n_classes = len(np.unique(y))
    if n_classes >= 2:
        t0 = time.perf_counter()
        lda = SklearnLDA(n_components=None)
        Xlda = lda.fit_transform(X_sel, y)
        t_lda = time.perf_counter() - t0
        metrics_lda = evaluate_embedding(Xlda, y)
        lda_coefs = pd.Series(lda.coef_.ravel(), index=selected_features)
        lda_top = list(lda_coefs.abs().sort_values(ascending=False).head(2).index)
        plt.figure(figsize=(6,2.5))
        plt.scatter(Xlda[:,0], np.zeros_like(Xlda[:,0]))
        plt.title("LDA (1D)"); plt.xlabel("LD1"); plt.yticks([])
        plt.tight_layout(); plt.savefig("fig/lda_1d.png"); plt.close()
        results.append({'method':'LDA','time_s':t_lda,'n_components':Xlda.shape[1], **metrics_lda,
                        'top_features_dim1':lda_top,'top_features_dim2':None})
    else:
        print("Only one label present -> skipping LDA (and skipping label-based metrics).")
        results.append({'method':'LDA','time_s':0,'n_components':0,'silhouette':np.nan,'davies_bouldin':np.nan,'calinski_harabasz':np.nan,'knn_cv_mean':np.nan,'knn_cv_std':np.nan,'top_features_dim1':None,'top_features_dim2':None,'note':'skipped_single_label'})

    # CSP (2-class only)
    if n_classes == 2:
        t0 = time.perf_counter()
        Xcsp, W = compute_csp(X_sel, y.values, n_components=2)
        t_csp = time.perf_counter() - t0
        metrics_csp = evaluate_embedding(Xcsp, y)
        Wdf = pd.DataFrame(W, index=selected_features, columns=['CSP1','CSP2'])
        csp_top = {c:list(Wdf[c].abs().sort_values(ascending=False).head(2).index) for c in Wdf.columns}
        plt.figure(figsize=(6,5))
        plt.scatter(Xcsp[:,0], Xcsp[:,1])
        plt.title("CSP (2D)"); plt.xlabel("CSP1"); plt.ylabel("CSP2")
        plt.tight_layout(); plt.savefig("fig/csp_2d.png"); plt.close()
        results.append({'method':'CSP','time_s':t_csp,'n_components':2, **metrics_csp,
                        'top_features_dim1':csp_top.get('CSP1'),'top_features_dim2':csp_top.get('CSP2')})
    else:
        print("CSP skipped (requires exactly 2 classes).")
        results.append({'method':'CSP','time_s':0,'n_components':0,'silhouette':np.nan,'davies_bouldin':np.nan,'calinski_harabasz':np.nan,'knn_cv_mean':np.nan,'knn_cv_std':np.nan,'top_features_dim1':None,'top_features_dim2':None,'note':'skipped_single_or_not_binary'})

    # t-SNE
    t0 = time.perf_counter()
    tsne = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto', perplexity=5)
    Xtsne = tsne.fit_transform(X_sel)
    t_tsne = time.perf_counter() - t0
    metrics_tsne = evaluate_embedding(Xtsne, y)
    plt.figure(figsize=(6,5)); plt.scatter(Xtsne[:,0], Xtsne[:,1]); plt.title("t-SNE (2D)"); plt.xlabel("tSNE1"); plt.ylabel("tSNE2")
    plt.tight_layout(); plt.savefig("fig/tsne_2d.png"); plt.close()
    results.append({'method':'t-SNE','time_s':t_tsne,'n_components':2, **metrics_tsne, 'top_features_dim1':None,'top_features_dim2':None})

    # UMAP (if installed)
    try:
        from umap import UMAP
        have_umap = True
    except Exception:
        have_umap = False
    if have_umap:
        t0 = time.perf_counter()
        umap = UMAP(n_components=2, random_state=random_state)
        Xumap = umap.fit_transform(X_sel)
        t_umap = time.perf_counter() - t0
        metrics_umap = evaluate_embedding(Xumap, y)
        plt.figure(figsize=(6,5)); plt.scatter(Xumap[:,0], Xumap[:,1]); plt.title("UMAP (2D)"); plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
        plt.tight_layout(); plt.savefig("fig/umap_2d.png"); plt.close()
        results.append({'method':'UMAP','time_s':t_umap,'n_components':2, **metrics_umap, 'top_features_dim1':None,'top_features_dim2':None})
    else:
        print("UMAP not installed — skipping UMAP. To enable, install umap-learn: pip install umap-learn")

    # Summary CSV
    resdf = pd.DataFrame(results)
    resdf.to_csv("fig/analysis_summary.csv", index=False)
    print("Saved figs into ./fig/ and summary CSV at fig/analysis_summary.csv")
    print(resdf)
    return resdf

if __name__ == "__main__":
    main()
