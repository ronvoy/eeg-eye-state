import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import umap

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("eeg_data_og.csv")

# Separate features and target
X = df.drop("eyeDetection", axis=1)
y = df["eyeDetection"]

# ================================
# 2. IQR OUTLIER REMOVAL
# ================================
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

X = X[mask]
y = y[mask]

print(f"After IQR filtering: {X.shape}")

# ================================
# 3. STANDARDIZATION
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 4. FEATURE ENGINEERING
# ================================

# Rolling features
window = 10

X_df = pd.DataFrame(X_scaled)

X_roll_mean = X_df.rolling(window=window).mean().fillna(0)
X_roll_std = X_df.rolling(window=window).std().fillna(0)

# FFT features
X_fft = np.abs(np.fft.fft(X_scaled, axis=0))

# Combine all features
X_features = np.hstack([
    X_scaled,
    X_roll_mean.values,
    X_roll_std.values,
    X_fft
])

print(f"Final feature shape: {X_features.shape}")

# ================================
# 5. LDA (SUPERVISED)
# ================================
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_features, y)

plt.figure()
plt.hist(X_lda[y == 0], bins=50, alpha=0.6, label="Open")
plt.hist(X_lda[y == 1], bins=50, alpha=0.6, label="Closed")
plt.title("LDA Projection")
plt.legend()
plt.show()

# ================================
# 6. t-SNE
# ================================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_features)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=5, cmap="coolwarm")
plt.title("t-SNE Projection")
plt.colorbar(label="Eye State (0=open, 1=closed)")
plt.show()

# ================================
# 7. UMAP
# ================================
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_features)

plt.figure()
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, s=5, cmap="coolwarm")
plt.title("UMAP Projection")
plt.colorbar(label="Eye State (0=open, 1=closed)")
plt.show()

# ================================
# 8. CLASSIFICATION (OPTIONAL BUT IMPORTANT)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))