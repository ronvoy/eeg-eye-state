import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report

# Define file and target
INPUT_FILE = 'eeg_data_pca.csv'
TARGET_COLUMN = 'eyeDetection'

print(f"Starting LDA classification using features from {INPUT_FILE}...")

# --- 1. LOAD THE DATA ---
try:
    df_pca = pd.read_csv(INPUT_FILE)
    print(f"✅ Data loaded successfully. Shape: {df_pca.shape}")
except FileNotFoundError:
    print(f"❌ Error: '{INPUT_FILE}' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. DEFINE FEATURES (X) AND TARGET (y) ---

# All columns except the target are now features (including the AF_PCA_Component)
X = df_pca.drop(columns=[TARGET_COLUMN])
y = df_pca[TARGET_COLUMN]

# List of features being used for LDA (12 original + 1 PCA component)
feature_columns = X.columns.tolist()
print(f"\nFeatures used in LDA ({len(feature_columns)} total):")
print(feature_columns)

# --- 3. SPLIT DATA INTO TRAINING AND TESTING SETS ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # stratify ensures balanced classes
)

# --- 4. SCALE THE FEATURES ---
# Scaling is essential before applying LDA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use fit_transform on train, only transform on test

# --- 5. APPLY LDA AND TRAIN THE MODEL ---
# For a binary classification problem (like eyeDetection), LDA projects data onto a single dimension.
lda = LDA(n_components=1)

# Fit and transform the training data
X_train_lda = lda.fit_transform(X_train_scaled, y_train)

# Transform the test data using the fitted LDA model
X_test_lda = lda.transform(X_test_scaled)

print("\n--- LDA Feature Transformation ---")
print(f"Original feature space (X_train): {X_train_scaled.shape}")
print(f"Reduced feature space (X_train_lda): {X_train_lda.shape} (Data projected onto 1 dimension)")

# --- 6. CLASSIFICATION AND EVALUATION ---
# Since LDA is primarily a classification method, we use the fitted LDA model to predict
y_pred = lda.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Eyes Closed', 'Eyes Open']))