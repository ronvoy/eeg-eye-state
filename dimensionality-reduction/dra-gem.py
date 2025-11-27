import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from scipy import linalg

# Sklearn Imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score

# Try importing UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: 'umap-learn' not installed. Skipping UMAP analysis.")

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------

DATA_DIR = 'dataset'
FILE_PATH = os.path.join(DATA_DIR, 'eeg_data.csv')
IMG_DIR = 'fig'

# Create output directory
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Create dummy data if file doesn't exist (for demonstration purposes)
if not os.path.exists(FILE_PATH):
    print(f"File {FILE_PATH} not found. Creating dummy data based on user snippet...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Generate synthetic EEG-like data
    np.random.seed(42)
    n_samples = 500
    # Columns from user request
    cols = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    
    data_matrix = np.random.normal(4200, 100, (n_samples, 14))
    
    # Inject some pattern for 'eyeDetection' (0 or 1)
    labels = np.random.randint(0, 2, n_samples)
    
    # Make class 1 slightly different in Frontal channels (AF3, AF4, F7, F8) to simulate artifacts
    data_matrix[labels == 1, 0] += 300 # AF3
    data_matrix[labels == 1, 13] += 300 # AF4
    
    df = pd.DataFrame(data_matrix, columns=cols)
    df['eyeDetection'] = labels
    df.to_csv(FILE_PATH, index=False)
else:
    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)

# ---------------------------------------------------------
# 2. FEATURE SELECTION & CORRELATION
# ---------------------------------------------------------

# Separate Features and Target
X_raw = df.drop(columns=['eyeDetection'])
y = df['eyeDetection']

# 1. Standardize Data (Crucial for EEG/PCA/LDA)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# 2. Correlation Analysis
corr_matrix = X_scaled.corr().abs()

# Plot Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X_raw.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of EEG Features')
plt.savefig(os.path.join(IMG_DIR, 'correlation_heatmap.png'))
plt.close()

# 3. Drop Highly Correlated Features (> 0.95)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

X_selected = X_scaled.drop(columns=to_drop)

print("\n" + "="*50)
print("FEATURE SELECTION REPORT")
print("="*50)
print(f"Original Features ({len(X_raw.columns)}): {list(X_raw.columns)}")
print(f"Features Removed (>0.95 corr): {to_drop}")
print(f"Selected Features ({len(X_selected.columns)}): {list(X_selected.columns)}")
print("="*50 + "\n")

# Use X_final for all subsequent analysis
X_final = X_selected.values
feature_names = X_selected.columns.tolist()

# ---------------------------------------------------------
# 3. ALGORITHM IMPLEMENTATIONS
# ---------------------------------------------------------

# Custom CSP Implementation (Common Spatial Patterns)
# Standard CSP maximizes variance of class 1 while minimizing class 2
class SimpleCSP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        # Expects X as (n_samples, n_features)
        # Calculate covariance matrices for each class
        class_labels = np.unique(y)
        covs = []
        for label in class_labels:
            X_class = X[y == label]
            # Covariance: (features x features)
            cov = np.cov(X_class, rowvar=False)
            covs.append(cov)
        
        # Sigmas
        R1 = covs[0]
        R2 = covs[1]
        
        # Generalized Eigenvalue Problem: R1 * w = lambda * (R1 + R2) * w
        # Scipy solves A x = lambda B x
        eigenvalues, eigenvectors = linalg.eig(R1, R1 + R2)
        
        # Sort by eigenvalues (descending)
        ix = np.argsort(np.abs(eigenvalues))[::-1]
        sorted_vectors = eigenvectors[:, ix]
        
        # Pick first and last vectors (most discriminative for Class 0 vs Class 1)
        # If n_components is 2, we take the very first and very last
        filters = np.zeros((self.n_components, X.shape[1]))
        filters[0] = sorted_vectors[:, 0]
        filters[1] = sorted_vectors[:, -1]
        
        self.filters_ = filters
        return self

    def transform(self, X):
        return np.dot(X, self.filters_.T)

# ---------------------------------------------------------
# 4. EVALUATION & EXECUTION
# ---------------------------------------------------------

results_log = []

def evaluate_and_plot(name, X_transformed, y_labels, time_taken):
    print(f"Processing {name}...")
    
    # 1. Metrics
    # Clustering Metrics (Unsupervised quality)
    try:
        sil = silhouette_score(X_transformed, y_labels)
        db_score = davies_bouldin_score(X_transformed, y_labels)
        ch_score = calinski_harabasz_score(X_transformed, y_labels)
    except:
        # Fails if only 1 cluster or errors
        sil, db_score, ch_score = 0, 0, 0

    # Classification Metric (Supervised utility)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_labels, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))

    results_log.append({
        'Method': name,
        'Time (s)': round(time_taken, 4),
        'Silhouette (High is good)': round(sil, 4),
        'Davies-Bouldin (Low is good)': round(db_score, 4),
        'Calinski-Harabasz (High is good)': round(ch_score, 4),
        'KNN Accuracy': round(acc, 4)
    })

    # 2. Plotting
    plt.figure(figsize=(8, 6))
    
    # Handle 1D case (LDA often returns 1 component for binary classes)
    if X_transformed.shape[1] == 1:
        sns.scatterplot(x=X_transformed[:, 0], y=np.zeros_like(X_transformed[:, 0]), 
                        hue=y_labels, palette='viridis', alpha=0.7)
        plt.title(f'{name} Projection (1D)')
    else:
        sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], 
                        hue=y_labels, palette='viridis', alpha=0.7)
        plt.title(f'{name} Projection (2D)')
        
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='EyeDetection')
    plt.savefig(os.path.join(IMG_DIR, f'{name.lower()}_projection.png'))
    plt.close()

# --- Execution ---

# 1. PCA
start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_final)
evaluate_and_plot('PCA', X_pca, y, time.time() - start)

# 2. LDA
# Note: For binary class, LDA max components = 1. We cannot force 2.
start = time.time()
lda = LDA(n_components=1) 
X_lda_1d = lda.fit_transform(X_final, y)
# Trick for consistency: If 1D, make 2nd dimension zeros or run PCA on result (impossible for 1D).
# We will just pass it as is, plotter handles 1D.
evaluate_and_plot('LDA', X_lda_1d, y, time.time() - start)

# 3. ICA
start = time.time()
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_final)
evaluate_and_plot('ICA', X_ica, y, time.time() - start)

# 4. CSP
start = time.time()
csp = SimpleCSP(n_components=2)
csp.fit(X_final, y)
X_csp = csp.transform(X_final)
evaluate_and_plot('CSP', X_csp, y, time.time() - start)

# 5. t-SNE
start = time.time()
# Perplexity should be < n_samples. Default is 30.
p_val = min(30, len(X_final) - 1)
tsne = TSNE(n_components=2, perplexity=p_val, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_final)
evaluate_and_plot('t-SNE', X_tsne, y, time.time() - start)

# 6. UMAP
if HAS_UMAP:
    start = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_final)
    evaluate_and_plot('UMAP', X_umap, y, time.time() - start)

# ---------------------------------------------------------
# 5. FINAL REPORT
# ---------------------------------------------------------

results_df = pd.DataFrame(results_log)
print("\n" + "="*80)
print("PERFORMANCE COMPARISON REPORT")
print("="*80)
# Sort by KNN Accuracy (usually most relevant for labeled EEG)
print(results_df.sort_values(by='KNN Accuracy', ascending=False).to_string(index=False))

# Suggestion logic
best_acc = results_df.sort_values(by='KNN Accuracy', ascending=False).iloc[0]
best_sep = results_df.sort_values(by='Silhouette (High is good)', ascending=False).iloc[0]

print("\n" + "-"*30)
print("CONCLUSION")
print("-"*30)
print(f"Most Efficient for Classification: {best_acc['Method']} (Acc: {best_acc['KNN Accuracy']})")
print(f"Best Visual Separation: {best_sep['Method']} (Silhouette: {best_sep['Silhouette (High is good)']})")