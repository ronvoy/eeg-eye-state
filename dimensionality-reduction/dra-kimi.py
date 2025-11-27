"""
eeg_reduce.py
Complete dimensionality-reduction benchmark on EEG data.
Author : you
"""

import os, time, warnings, pathlib, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import accuracy_score
from sklearn.cluster         import KMeans
from sklearn.pipeline        import Pipeline

from sklearn.metrics import (
    silhouette_score , davies_bouldin_score, calinski_harabasz_score
)

from scipy.linalg import eigh
from umap import UMAP
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (6,4)

DATA_FILE = "dataset/eeg_data.csv"
CORR_TH   = 0.80               # <-- change correlation threshold here
RANDOM_STATE = 42
N_COMPS   = 2                  # 2-D visualisation
KNN_K     = 5

# ------------------------------------------------------------------
# 1. IO
# ------------------------------------------------------------------
os.makedirs("fig", exist_ok=True)
df = pd.read_csv(DATA_FILE)
channels = [c for c in df.columns if c not in ("eyeDetection",)]
X_full = df[channels].values
y      = df["eyeDetection"].values

print("Original channels :", channels)

# ------------------------------------------------------------------
# 2. Correlation filter
# ------------------------------------------------------------------
corr = df[channels].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_TH)]
keep    = [c for c in channels if c not in to_drop]
print(f"Correlation threshold = {CORR_TH}")
print("Dropped (high corr)   :", to_drop)
print("Kept channels         :", keep)

X = df[keep].values
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 3. Correlation heat-map of kept channels
# ------------------------------------------------------------------
plt.figure()
sns.heatmap(pd.DataFrame(X_std, columns=keep).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation matrix – retained channels")
plt.tight_layout()
plt.savefig("fig/corr_kept.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------------
def bench(name, proj, X_tr, y_tr, X_te, y_te):
    """Return dict with metrics and wall time."""
    t0 = time.time()
    X_low = proj.fit_transform(X_tr) if hasattr(proj,"fit_transform") else proj.transform(X_tr)
    elapsed = time.time() - t0

    # clustering quality on the 2-D projection
    km = KMeans(n_clusters=len(np.unique(y_tr)), random_state=RANDOM_STATE, n_init='auto')
    pred_clust = km.fit_predict(X_low)

    sil  = silhouette_score(X_low, pred_clust)
    db   = davies_bouldin_score(X_low, pred_clust)
    ch   = calinski_harabasz_score(X_low, pred_clust)

    # knn accuracy on the 2-D projection
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    knn.fit(X_low, y_tr)
    acc = accuracy_score(y_te, knn.predict(proj.transform(X_te) if hasattr(proj,"transform") else proj.fit_transform(X_te)))

    return dict(method=name, sil=sil, db=db, ch=ch, knn_acc=acc, time=elapsed)

# ------------------------------------------------------------------
# 5. Algorithms
# ------------------------------------------------------------------
results = []

# ---- PCA ------------------------------------------------------------
pca = PCA(n_components=N_COMPS, random_state=RANDOM_STATE)
X_tr, X_te, y_tr, y_te = train_test_split(X_std, y, test_size=.2, random_state=RANDOM_STATE, stratify=y)
results.append(bench("PCA", pca, X_tr, y_tr, X_te, y_te))

# ---- ICA ------------------------------------------------------------
ica = FastICA(n_components=N_COMPS, random_state=RANDOM_STATE, whiten='unit-variance')
results.append(bench("ICA", ica, X_tr, y_tr, X_te, y_te))

# ---- LDA ------------------------------------------------------------
lda = LDA(n_components=min(N_COMPS, len(np.unique(y))-1))
results.append(bench("LDA", lda, X_tr, y_tr, X_te, y_te))

# ---- CSP (Common Spatial Patterns) ----------------------------------
# needs covariance estimation → works on raw trials, here we fake trials
# by simple sliding window (len 50, step 25) just to show the idea
WINDOW, STEP = 50, 25
trials, labels = [], []
for start in range(0, X_std.shape[0]-WINDOW, STEP):
    trials.append(X_std[start:start+WINDOW])
    labels.append(y[start])
trials = np.array(trials)          # (n_trials, window, channels)
labels = np.array(labels)

def csp_fit_transform(X_trials, y_labels, n_comp=N_COMPS):
    """Hand-written CSP for 2-class problem."""
    cl = np.unique(y_labels)
    if len(cl)!=2: raise ValueError("CSP needs binary problem")
    # class-wise covariance
    cov0 = np.zeros((X_trials.shape[2], X_trials.shape[2]))
    cov1 = np.zeros_like(cov0)
    for t, lab in zip(X_trials, y_labels):
        t = t.T
        if lab==cl[0]: cov0 += t @ t.T
        else:          cov1 += t @ t.T
    cov0 /= np.sum(y_labels==cl[0])
    cov1 /= np.sum(y_labels==cl[1])
    # GEVD
    D, V = eigh(cov0, cov0 + cov1)
    idx = np.argsort(D)[::-1]
    W = V[:, idx[:n_comp]]
    # project
    feats = np.array([W.T @ trial.T for trial in X_trials])
    return np.array([np.log(np.var(f, axis=1)) for f in feats])  # log-variance

t0 = time.time()
csp_feats = csp_fit_transform(trials, labels)
elapsed = time.time() - t0
# clustering & knn on CSP features
km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init='auto')
pred_clust = km.fit_predict(csp_feats)
sil  = silhouette_score(csp_feats, pred_clust)
db   = davies_bouldin_score(csp_feats, pred_clust)
ch   = calinski_harabasz_score(csp_feats, pred_clust)
X_csp_tr, X_csp_te, y_csp_tr, y_csp_te = train_test_split(csp_feats, labels, test_size=.2, random_state=RANDOM_STATE, stratify=labels)
knn = KNeighborsClassifier(n_neighbors=KNN_K).fit(X_csp_tr, y_csp_tr)
acc = accuracy_score(y_csp_te, knn.predict(X_csp_te))
results.append(dict(method="CSP", sil=sil, db=db, ch=ch, knn_acc=acc, time=elapsed))

# ---- t-SNE ----------------------------------------------------------
tsne = TSNE(n_components=N_COMPS, random_state=RANDOM_STATE, perplexity=30)
t0 = time.time()
X_tsne = tsne.fit_transform(X_std)
elapsed = time.time() - t0
km = KMeans(n_clusters=len(np.unique(y)), random_state=RANDOM_STATE, n_init='auto')
pred_clust = km.fit_predict(X_tsne)
sil  = silhouette_score(X_tsne, pred_clust)
db   = davies_bouldin_score(X_tsne, pred_clust)
ch   = calinski_harabasz_score(X_tsne, pred_clust)
# knn on whole tsne (no train/test split needed because tsne is unsupervised)
knn = KNeighborsClassifier(n_neighbors=KNN_K).fit(X_tsne, y)
acc = accuracy_score(y, knn.predict(X_tsne))
results.append(dict(method="t-SNE", sil=sil, db=db, ch=ch, knn_acc=acc, time=elapsed))

# ---- UMAP -----------------------------------------------------------
umap = UMAP(n_components=N_COMPS, random_state=RANDOM_STATE)
t0 = time.time()
X_umap = umap.fit_transform(X_std)
elapsed = time.time() - t0
km = KMeans(n_clusters=len(np.unique(y)), random_state=RANDOM_STATE, n_init='auto')
pred_clust = km.fit_predict(X_umap)
sil  = silhouette_score(X_umap, pred_clust)
db   = davies_bouldin_score(X_umap, pred_clust)
ch   = calinski_harabasz_score(X_umap, pred_clust)
knn = KNeighborsClassifier(n_neighbors=KNN_K).fit(X_umap, y)
acc = accuracy_score(y, knn.predict(X_umap))
results.append(dict(method="UMAP", sil=sil, db=db, ch=ch, knn_acc=acc, time=elapsed))

# ------------------------------------------------------------------
# 6. Visualisations
# ------------------------------------------------------------------
algorithms = {
    "PCA"  : pca.transform(X_std),
    "ICA"  : ica.transform(X_std),
    "LDA"  : lda.transform(X_std),
    "CSP"  : csp_feats,
    "t-SNE": X_tsne,
    "UMAP" : X_umap
}

for name, emb in algorithms.items():
    plt.figure()
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=y, palette="Set2", s=60)
    plt.title(f"{name} – 2-D projection")
    plt.tight_layout()
    plt.savefig(f"fig/{name.lower()}_2d.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------
# 7. Report
# ------------------------------------------------------------------
report = pd.DataFrame(results).sort_values("sil", ascending=False)
print("\n===========  SUMMARY  ===========")
print(report.to_string(index=False))
print("\n(↑ higher Sil/CH/KNN-acc better; ↓ lower DB better)")