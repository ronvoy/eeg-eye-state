import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score, accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Create output folder
os.makedirs('fig', exist_ok=True)

# Load data
print("="*80)
print("LOADING DATA")
print("="*80)
data = pd.read_csv('dataset/eeg_data.csv')
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nFirst few rows:")
print(data.head())

# Separate features and target
feature_cols = [col for col in data.columns if col != 'eyeDetection']
X = data[feature_cols]
y = data['eyeDetection']

print(f"\nFeatures: {feature_cols}")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================================================
# STEP 1: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CORRELATION ANALYSIS")
print("="*80)

# Compute correlation matrix
corr_matrix = X.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap of EEG Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

# Identify highly correlated features (threshold = 0.9)
print("\nHighly correlated feature pairs (|correlation| > 0.9):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], 
                                   corr_matrix.columns[j], 
                                   corr_matrix.iloc[i, j]))
            print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

if not high_corr_pairs:
    print("  No feature pairs with |correlation| > 0.9")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# DIMENSIONALITY REDUCTION RULES
# ============================================================================
print("\n" + "="*80)
print("DIMENSIONALITY REDUCTION TECHNIQUE RULES")
print("="*80)

rules = """
1. PCA (Principal Component Analysis)
   - Assumes: Linear relationships, data should be centered and scaled
   - Requirements: No strict correlation threshold, works best with correlated features
   - Features selected: All features used, linear combinations created
   - Best for: Capturing maximum variance in data

2. LDA (Linear Discriminant Analysis)
   - Assumes: Normal distribution, equal covariance matrices across classes
   - Requirements: Needs labeled data, max components = min(n_features, n_classes-1)
   - Features selected: All features used
   - Best for: Supervised dimensionality reduction, maximizing class separability

3. ICA (Independent Component Analysis)
   - Assumes: Features are linear mixtures of independent sources
   - Requirements: Non-Gaussian distributions, data should be centered
   - Features selected: All features used
   - Best for: Signal separation, removing artifacts

4. CSP (Common Spatial Patterns)
   - Assumes: Multi-channel signal data, binary classification
   - Requirements: Labeled data, designed for EEG/MEG, needs multiple trials
   - Features selected: All spatial features (channels)
   - Best for: EEG feature extraction for classification

5. t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - Assumes: Local structure preservation
   - Requirements: Computationally expensive, non-deterministic
   - Features selected: All features used
   - Best for: Visualization, capturing non-linear relationships

6. UMAP (Uniform Manifold Approximation and Projection)
   - Assumes: Local and global structure preservation
   - Requirements: Faster than t-SNE, more deterministic
   - Features selected: All features used
   - Best for: Both visualization and general dimensionality reduction
"""
print(rules)

# ============================================================================
# FEATURE SELECTION FOR EACH METHOD
# ============================================================================
print("\n" + "="*80)
print("FEATURES SELECTED FOR EACH METHOD")
print("="*80)
print(f"All methods will use all {len(feature_cols)} features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# ============================================================================
# PERFORM DIMENSIONALITY REDUCTION
# ============================================================================
results = {}

# Function to evaluate dimensionality reduction
def evaluate_reduction(X_reduced, y, method_name):
    metrics = {}
    
    # Silhouette Score (higher is better, range: -1 to 1)
    sil_score = silhouette_score(X_reduced, y)
    metrics['Silhouette Score'] = sil_score
    
    # Davies-Bouldin Index (lower is better, min: 0)
    db_score = davies_bouldin_score(X_reduced, y)
    metrics['Davies-Bouldin Index'] = db_score
    
    # Calinski-Harabasz Score (higher is better)
    ch_score = calinski_harabasz_score(X_reduced, y)
    metrics['Calinski-Harabasz Score'] = ch_score
    
    # KNN Classification Accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.3, random_state=42, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics['KNN Accuracy'] = acc
    
    # Reconstruction error (for linear methods)
    if method_name in ['PCA', 'ICA']:
        metrics['Variance Explained'] = 'N/A'
    
    return metrics

# Function to plot 2D reduction
def plot_2d_reduction(X_reduced, y, method_name, time_taken):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                         c=y, cmap='viridis', alpha=0.6, s=50, edgecolors='black')
    plt.colorbar(scatter, label='eyeDetection')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(f'{method_name} - 2D Projection\nTime: {time_taken:.4f}s', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"fig/02_{method_name.lower().replace(' ', '_')}_2d.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {method_name} plot saved")

# ============================================================================
# 1. PCA
# ============================================================================
print("\n" + "="*80)
print("PERFORMING PCA")
print("="*80)
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - start_time

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
print(f"Time taken: {pca_time:.4f}s")

pca_metrics = evaluate_reduction(X_pca, y, 'PCA')
pca_metrics['Time (s)'] = pca_time
pca_metrics['Variance Explained'] = sum(pca.explained_variance_ratio_)
results['PCA'] = pca_metrics

plot_2d_reduction(X_pca, y, 'PCA', pca_time)

# ============================================================================
# 2. LDA
# ============================================================================
print("\n" + "="*80)
print("PERFORMING LDA")
print("="*80)
start_time = time.time()
lda = LDA(n_components=1)  # Max 1 component for binary classification
X_lda = lda.fit_transform(X_scaled, y)
# Add second dimension as zeros for visualization
X_lda_2d = np.column_stack([X_lda, np.zeros(len(X_lda))])
lda_time = time.time() - start_time

print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"Time taken: {lda_time:.4f}s")

lda_metrics = evaluate_reduction(X_lda_2d, y, 'LDA')
lda_metrics['Time (s)'] = lda_time
lda_metrics['Variance Explained'] = sum(lda.explained_variance_ratio_)
results['LDA'] = lda_metrics

plot_2d_reduction(X_lda_2d, y, 'LDA', lda_time)

# ============================================================================
# 3. ICA
# ============================================================================
print("\n" + "="*80)
print("PERFORMING ICA")
print("="*80)
start_time = time.time()
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)
ica_time = time.time() - start_time

print(f"Time taken: {ica_time:.4f}s")

ica_metrics = evaluate_reduction(X_ica, y, 'ICA')
ica_metrics['Time (s)'] = ica_time
results['ICA'] = ica_metrics

plot_2d_reduction(X_ica, y, 'ICA', ica_time)

# ============================================================================
# 4. CSP
# ============================================================================
print("\n" + "="*80)
print("PERFORMING CSP")
print("="*80)
start_time = time.time()

# Reshape data for CSP (n_epochs, n_channels, n_times)
# CSP expects 3D data, so we'll treat each sample as an epoch with 1 time point
n_samples = X_scaled.shape[0]
n_channels = X_scaled.shape[1]
X_csp_input = X_scaled.reshape(n_samples, n_channels, 1)

csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X_csp_input, y)
csp_time = time.time() - start_time

print(f"Time taken: {csp_time:.4f}s")

csp_metrics = evaluate_reduction(X_csp, y, 'CSP')
csp_metrics['Time (s)'] = csp_time
results['CSP'] = csp_metrics

plot_2d_reduction(X_csp, y, 'CSP', csp_time)

# ============================================================================
# 5. t-SNE
# ============================================================================
print("\n" + "="*80)
print("PERFORMING t-SNE")
print("="*80)
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
tsne_time = time.time() - start_time

print(f"Time taken: {tsne_time:.4f}s")

tsne_metrics = evaluate_reduction(X_tsne, y, 't-SNE')
tsne_metrics['Time (s)'] = tsne_time
results['t-SNE'] = tsne_metrics

plot_2d_reduction(X_tsne, y, 't-SNE', tsne_time)

# ============================================================================
# 6. UMAP
# ============================================================================
print("\n" + "="*80)
print("PERFORMING UMAP")
print("="*80)
start_time = time.time()
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
X_umap = umap_reducer.fit_transform(X_scaled)
umap_time = time.time() - start_time

print(f"Time taken: {umap_time:.4f}s")

umap_metrics = evaluate_reduction(X_umap, y, 'UMAP')
umap_metrics['Time (s)'] = umap_time
results['UMAP'] = umap_metrics

plot_2d_reduction(X_umap, y, 'UMAP', umap_time)

# ============================================================================
# COMPARISON OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).T
print("\n", results_df.to_string())

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Performance Metrics Comparison', fontsize=18, fontweight='bold')

metrics_to_plot = ['Silhouette Score', 'Davies-Bouldin Index', 
                   'Calinski-Harabasz Score', 'KNN Accuracy', 'Time (s)']

for idx, metric in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    methods = results_df.index
    values = results_df[metric].values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=9)

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('fig/03_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Metrics comparison plot saved")

# ============================================================================
# RANKING AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RANKING AND RECOMMENDATIONS")
print("="*80)

# Normalize metrics for ranking (higher is better)
ranking_df = results_df.copy()
ranking_df['Davies-Bouldin Index'] = 1 / (ranking_df['Davies-Bouldin Index'] + 1e-10)
ranking_df['Time (s)'] = 1 / (ranking_df['Time (s)'] + 1e-10)

# Normalize to 0-1 scale
for col in ['Silhouette Score', 'Davies-Bouldin Index', 
            'Calinski-Harabasz Score', 'KNN Accuracy', 'Time (s)']:
    min_val = ranking_df[col].min()
    max_val = ranking_df[col].max()
    if max_val - min_val > 0:
        ranking_df[col] = (ranking_df[col] - min_val) / (max_val - min_val)

# Calculate overall score
ranking_df['Overall Score'] = ranking_df[['Silhouette Score', 'Davies-Bouldin Index',
                                          'Calinski-Harabasz Score', 'KNN Accuracy']].mean(axis=1)

ranking_df = ranking_df.sort_values('Overall Score', ascending=False)

print("\nOverall Ranking (by normalized score):")
for i, (method, row) in enumerate(ranking_df.iterrows(), 1):
    print(f"{i}. {method}: {row['Overall Score']:.4f}")

print("\n" + "="*80)
print("BEST METHOD FOR EACH METRIC")
print("="*80)

for metric in ['Silhouette Score', 'Davies-Bouldin Index', 
               'Calinski-Harabasz Score', 'KNN Accuracy', 'Time (s)']:
    if metric == 'Davies-Bouldin Index':
        best_method = results_df[metric].idxmin()
        best_value = results_df[metric].min()
        print(f"{metric}: {best_method} ({best_value:.4f}) - Lower is better")
    else:
        best_method = results_df[metric].idxmax()
        best_value = results_df[metric].max()
        print(f"{metric}: {best_method} ({best_value:.4f}) - Higher is better")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
best_overall = ranking_df.index[0]
print(f"\n🏆 MOST EFFICIENT METHOD: {best_overall}")
print(f"   Overall Score: {ranking_df.loc[best_overall, 'Overall Score']:.4f}")
print(f"\n   Original Metrics:")
for metric in ['Silhouette Score', 'Davies-Bouldin Index', 
               'Calinski-Harabasz Score', 'KNN Accuracy', 'Time (s)']:
    print(f"   - {metric}: {results_df.loc[best_overall, metric]:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll plots saved in 'fig/' folder:")
print("  1. 01_correlation_heatmap.png")
print("  2. 02_pca_2d.png")
print("  3. 02_lda_2d.png")
print("  4. 02_ica_2d.png")
print("  5. 02_csp_2d.png")
print("  6. 02_t-sne_2d.png")
print("  7. 02_umap_2d.png")
print("  8. 03_metrics_comparison.png")

# Save results to CSV
results_df.to_csv('fig/results_summary.csv')
print("\n  9. results_summary.csv")
print("\n✓ All analyses completed successfully!")