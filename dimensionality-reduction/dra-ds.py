import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
import os

# Create figures directory
os.makedirs('fig', exist_ok=True)

# Load the data
data = pd.read_csv('dataset/eeg_data.csv')

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nClass distribution:")
print(data['eyeDetection'].value_counts())

# Separate features and target
X = data.drop('eyeDetection', axis=1)
y = data['eyeDetection']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Correlation Analysis and Heatmap
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

# Calculate correlation matrix
correlation_matrix = X.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('EEG Channels Correlation Heatmap')
plt.tight_layout()
plt.savefig('fig/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Correlation heatmap saved to 'fig/correlation_heatmap.png'")

# Find highly correlated features (threshold = 0.8)
high_corr_threshold = 0.8
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"\nHighly correlated pairs (|r| > {high_corr_threshold}):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

# Feature selection based on correlation
# Remove one feature from each highly correlated pair
features_to_keep = list(X.columns)
for pair in high_corr_pairs:
    if pair[1] in features_to_keep:
        features_to_keep.remove(pair[1])

print(f"\nSelected features after correlation analysis: {features_to_keep}")

# Prepare data with selected features
X_selected = X[features_to_keep]
X_selected_scaled = scaler.fit_transform(X_selected)

# 2. Define Analysis Rules and Perform Dimensionality Reduction
print("\n" + "="*50)
print("DIMENSIONALITY REDUCTION ANALYSIS RULES")
print("="*50)

analysis_rules = {
    'PCA': {
        'description': 'Principal Component Analysis - No correlation threshold needed as it handles multicollinearity',
        'requires_correlation_threshold': False,
        'selected_features': 'All features (handles correlation automatically)',
        'data': X_scaled
    },
    'LDA': {
        'description': 'Linear Discriminant Analysis - Requires class labels, no correlation threshold',
        'requires_correlation_threshold': False,
        'selected_features': 'All features (supervised method)',
        'data': X_scaled
    },
    'ICA': {
        'description': 'Independent Component Analysis - Assumes independent sources, moderate correlation acceptable',
        'requires_correlation_threshold': False,
        'selected_features': 'All features (seeks independent components)',
        'data': X_scaled
    },
    'CSP': {
        'description': 'Common Spatial Patterns - EEG specific, handles correlated channels',
        'requires_correlation_threshold': False,
        'selected_features': 'All EEG channels',
        'data': X_scaled
    },
    't-SNE': {
        'description': 't-Distributed Stochastic Neighbor Embedding - Non-linear, correlation handled implicitly',
        'requires_correlation_threshold': False,
        'selected_features': 'All features',
        'data': X_scaled
    },
    'UMAP': {
        'description': 'Uniform Manifold Approximation and Projection - Non-linear, robust to correlation',
        'requires_correlation_threshold': False,
        'selected_features': 'All features',
        'data': X_scaled
    }
}

# Custom CSP implementation for EEG data
def common_spatial_patterns(X, y, n_components=2):
    """
    Common Spatial Patterns for EEG feature extraction
    """
    # Separate classes
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    # Calculate covariance matrices
    cov_0 = np.cov(class_0.T)
    cov_1 = np.cov(class_1.T)
    
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(cov_0) @ cov_1)
    
    # Sort eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Select top and bottom components
    components = np.hstack([eigenvectors_sorted[:, :n_components//2], 
                           eigenvectors_sorted[:, -n_components//2:]])
    
    return X @ components

# Perform all analyses and collect results
results = []

def evaluate_embedding(embedding, y, method_name, X_original=None):
    """Evaluate embedding using multiple metrics"""
    metrics = {}
    
    # Clustering metrics
    if len(np.unique(y)) > 1:
        metrics['silhouette_score'] = silhouette_score(embedding, y)
        metrics['davies_bouldin_score'] = davies_bouldin_score(embedding, y)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embedding, y)
    else:
        metrics['silhouette_score'] = np.nan
        metrics['davies_bouldin_score'] = np.nan
        metrics['calinski_harabasz_score'] = np.nan
    
    # KNN Classification Accuracy
    if X_original is not None and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_original, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train on original features
        knn_original = KNeighborsClassifier(n_neighbors=5)
        knn_original.fit(X_train, y_train)
        original_accuracy = knn_original.score(X_test, y_test)
        
        # Train on embedded features
        embedding_train, embedding_test, y_train_emb, y_test_emb = train_test_split(
            embedding, y, test_size=0.3, random_state=42, stratify=y
        )
        knn_embedded = KNeighborsClassifier(n_neighbors=5)
        knn_embedded.fit(embedding_train, y_train_emb)
        embedded_accuracy = knn_embedded.score(embedding_test, y_test_emb)
        
        metrics['knn_accuracy_original'] = original_accuracy
        metrics['knn_accuracy_embedded'] = embedded_accuracy
        metrics['accuracy_preservation'] = embedded_accuracy / original_accuracy
    else:
        metrics['knn_accuracy_original'] = np.nan
        metrics['knn_accuracy_embedded'] = np.nan
        metrics['accuracy_preservation'] = np.nan
    
    return metrics

# Perform each analysis
methods = ['PCA', 'LDA', 'ICA', 'CSP', 't-SNE', 'UMAP']

for method in methods:
    print(f"\nPerforming {method}...")
    start_time = time.time()
    
    try:
        if method == 'PCA':
            pca = PCA(n_components=2, random_state=42)
            embedding = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            
        elif method == 'LDA':
            if len(np.unique(y)) > 1:  # LDA requires at least 2 classes
                lda = LinearDiscriminantAnalysis(n_components=min(2, len(np.unique(y))-1))
                embedding = lda.fit_transform(X_scaled, y)
            else:
                print("Skipping LDA - only one class present")
                continue
                
        elif method == 'ICA':
            ica = FastICA(n_components=2, random_state=42, max_iter=1000)
            embedding = ica.fit_transform(X_scaled)
            
        elif method == 'CSP':
            if len(np.unique(y)) > 1:  # CSP requires at least 2 classes
                embedding = common_spatial_patterns(X_scaled, y, n_components=2)
            else:
                print("Skipping CSP - only one class present")
                continue
                
        elif method == 't-SNE':
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
            embedding = tsne.fit_transform(X_scaled)
            
        elif method == 'UMAP':
            umap_model = umap.UMAP(n_components=2, random_state=42)
            embedding = umap_model.fit_transform(X_scaled)
        
        execution_time = time.time() - start_time
        
        # Evaluate the embedding
        metrics = evaluate_embedding(embedding, y, method, X_scaled)
        
        # Store results
        result = {
            'method': method,
            'execution_time': execution_time,
            'embedding': embedding,
            'metrics': metrics
        }
        
        if method == 'PCA':
            result['explained_variance'] = explained_variance
        
        results.append(result)
        
        print(f"{method} completed in {execution_time:.4f} seconds")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if len(np.unique(y)) > 1:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='eyeDetection')
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        
        plt.title(f'{method} Projection\n'
                 f'Silhouette: {metrics["silhouette_score"]:.3f} | '
                 f'DB Index: {metrics["davies_bouldin_score"]:.3f}')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'fig/{method.lower()}_projection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{method} plot saved to 'fig/{method.lower()}_projection.png'")
        
    except Exception as e:
        print(f"Error in {method}: {str(e)}")
        continue

# 3. Comparative Analysis
print("\n" + "="*50)
print("COMPARATIVE ANALYSIS RESULTS")
print("="*50)

# Create results dataframe
results_df = pd.DataFrame([{
    'Method': r['method'],
    'Time (s)': f"{r['execution_time']:.4f}",
    'Silhouette Score': f"{r['metrics']['silhouette_score']:.4f}",
    'Davies-Bouldin Index': f"{r['metrics']['davies_bouldin_score']:.4f}",
    'Calinski-Harabasz Score': f"{r['metrics']['calinski_harabasz_score']:.4f}",
    'KNN Accuracy (Embedded)': f"{r['metrics']['knn_accuracy_embedded']:.4f}" if not np.isnan(r['metrics']['knn_accuracy_embedded']) else 'N/A',
    'Accuracy Preservation': f"{r['metrics']['accuracy_preservation']:.3f}" if not np.isnan(r['metrics']['accuracy_preservation']) else 'N/A'
} for r in results])

print("\nPerformance Comparison:")
print(results_df.to_string(index=False))

# Create comparison visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, result in enumerate(results):
    if i < len(axes):
        embedding = result['embedding']
        method = result['method']
        
        if len(np.unique(y)) > 1:
            scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[i].set_title(f'{method}\nSilhouette: {result["metrics"]["silhouette_score"]:.3f}')
        else:
            scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
            axes[i].set_title(f'{method}')
        
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig/all_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nComparison plot saved to 'fig/all_methods_comparison.png'")

# 4. Determine Most Efficient Method
print("\n" + "="*50)
print("EFFICIENCY ANALYSIS")
print("="*50)

if results:
    # Normalize metrics for comparison (higher is better for most metrics)
    normalized_scores = []
    
    for result in results:
        method = result['method']
        metrics = result['metrics']
        
        if not np.isnan(metrics['silhouette_score']):
            # For silhouette and Calinski-Harabasz: higher is better
            # For Davies-Bouldin: lower is better
            # For time: lower is better
            
            # Calculate composite score (you can adjust weights as needed)
            silhouette_norm = metrics['silhouette_score']
            db_index_norm = 1 / (1 + metrics['davies_bouldin_score'])  # invert DB index
            ch_score_norm = metrics['calinski_harabasz_score'] / 1000 if not np.isnan(metrics['calinski_harabasz_score']) else 0
            time_norm = 1 / (1 + result['execution_time'])
            
            composite_score = (silhouette_norm + db_index_norm + ch_score_norm + time_norm) / 4
            
            normalized_scores.append({
                'method': method,
                'composite_score': composite_score,
                'silhouette': metrics['silhouette_score'],
                'db_index': metrics['davies_bouldin_score'],
                'execution_time': result['execution_time']
            })
    
    if normalized_scores:
        # Find best method
        best_method = max(normalized_scores, key=lambda x: x['composite_score'])
        
        print(f"\nMOST EFFICIENT METHOD: {best_method['method']}")
        print(f"Composite Score: {best_method['composite_score']:.4f}")
        print(f"Silhouette Score: {best_method['silhouette']:.4f}")
        print(f"Davies-Bouldin Index: {best_method['db_index']:.4f}")
        print(f"Execution Time: {best_method['execution_time']:.4f}s")

# 5. Feature Importance Analysis (for PCA)
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# PCA feature importance
pca_result = next((r for r in results if r['method'] == 'PCA'), None)
if pca_result:
    pca = PCA(n_components=2).fit(X_scaled)
    
    # Get absolute loadings for interpretation
    loadings = np.abs(pca.components_)
    
    # Feature importance based on PCA loadings
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'PC1_Loading': loadings[0],
        'PC2_Loading': loadings[1],
        'Total_Loading': loadings[0] + loadings[1]
    }).sort_values('Total_Loading', ascending=False)
    
    print("\nTop 5 Most Important Features (PCA Loadings):")
    print(feature_importance.head().to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos, top_features['Total_Loading'], align='center', alpha=0.7)
    plt.yticks(y_pos, top_features['Feature'])
    plt.xlabel('Total PCA Loading (Absolute Value)')
    plt.title('Top 10 Most Important EEG Channels (PCA)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('fig/feature_importance_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature importance plot saved to 'fig/feature_importance_pca.png'")

print("\nAnalysis complete! All plots saved to 'fig/' folder.")