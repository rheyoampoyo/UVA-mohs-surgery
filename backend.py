# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    StandardScaler
)

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
mohs = pd.read_excel('Mohs Scheduling Mock Data 062925.xlsx')
mohs = mohs.drop(['Number of Lesions (1-4)','lesion_id'], axis=1)
for i in mohs.dtypes.index:
    if mohs.dtypes[i] == 'object':
        mohs[i] = mohs[i].astype('category')

# %%
features = ['Duration of Visit (min)', 'Anesthetic Amount (ml)', 'Number of stages']
X = mohs[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute WCSS for different k
wcss = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# %%
### Create Complexity Scores from KMeans Clustering
mohs_test = mohs.copy()
features = ['Duration of Visit (min)', 'Anesthetic Amount (ml)', 'Number of stages']
X = mohs_test[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_centers = kmeans.cluster_centers_

distances = []
for i in range(3):
    cluster_points = X_scaled[cluster_labels == i]
    centroid = cluster_centers[i]
    avg_dist = np.linalg.norm(cluster_points - centroid, axis=1).mean()
    distances.append((i, avg_dist))

cluster_ranking = sorted(distances, key=lambda x: x[1])
cluster_order = {cluster_id: rank for rank, (cluster_id, _) in enumerate(cluster_ranking)}
complexity_labels = [cluster_order[label] for label in cluster_labels]

mohs_test['Visit Complexity KMeans'] = complexity_labels

y = mohs_test['Visit Complexity KMeans']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(eval_metric='rmse', random_state=42)
model.fit(X_train, y_train)

y_full_pred = model.predict(X)
y_full_pred_normalized = (y_full_pred - y_full_pred.min()) / (y_full_pred.max() - y_full_pred.min())
mohs_test['Normalized Visit Complexity Score KMeans'] = y_full_pred_normalized


mohs_test['Visit Complexity Category KMeans'] = pd.cut(
    mohs_test['Normalized Visit Complexity Score KMeans'],
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=['Low', 'Medium', 'High']
)

mohs_test_knn = mohs_test.copy()

mohs_test

# %%
mohs_test_knn['Visit Complexity Category KMeans'].value_counts()

# %%
features = ['Duration of Visit (min)', 'Anesthetic Amount (ml)', 'Number of stages']
X = mohs_test[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
pca.fit(X_scaled)

# Explained variance by each component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', label='Cumulative Variance')
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual Variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.legend()
plt.grid(True)
plt.show()

# %%
### Create Complexity Scores from PCA Clustering
X = mohs_test[['Duration of Visit (min)', 'Anesthetic Amount (ml)', 'Number of stages']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
reconstruction_error_normalized = (reconstruction_error - reconstruction_error.min()) / \
                                   (reconstruction_error.max() - reconstruction_error.min())

mohs_test['Normalized Visit Complexity Score PCA'] = reconstruction_error_normalized
mohs_test['Visit Complexity Category PCA'] = pd.cut(
    mohs_test['Normalized Visit Complexity Score PCA'],
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=['Low', 'Medium', 'High']
)

mohs_test_pca = mohs_test.copy()
mohs_test_pca

# %%
mohs_test_pca['Visit Complexity Category PCA'].value_counts()

# %%
X = mohs_test.drop(columns=['Duration of Visit (min)', 'Anesthetic Amount (ml)', 'Number of stages'])

X_encoded = pd.get_dummies(X.astype(str), drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Compute WCSS for different k
wcss = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# %% [markdown]
# Likely means that the data is too high dimensional so lets turn to PCA

# %%
numeric_cols = ['Lesion  Size (cm)', 'Treatment Delay (days)', 'Age (years)']
categorical_cols = [
    'Recurrent Tumor (Y/N)', 'Aggressive Histology (Y/N)', 'Wound Management (H/M/L)',
    'Location (H/M/L)', 'Immunosuppressed (Y/N)', 'Bleeding Risk (Y/N)', 'Greater Average Time (Y/N)'
]

X = mohs[numeric_cols + categorical_cols].copy() 
X[categorical_cols] = X[categorical_cols].astype(str)

X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# %%
pca = PCA()
pca.fit(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))

plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual Variance')

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', label='Cumulative Variance')

plt.title('Scree Plot: Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
### Create Patient Complexity Scores from PCA Clustering
pca = PCA(n_components=11)
X_pca = pca.fit_transform(X_scaled)

X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

reconstruction_error_normalized = (reconstruction_error - reconstruction_error.min()) / \
                                  (reconstruction_error.max() - reconstruction_error.min())

mohs_test['Normalized Patient Complexity Score (PCA)'] = reconstruction_error_normalized
mohs_test['Patient Complexity Category (PCA)'] = pd.cut(
    reconstruction_error_normalized,
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=['Low', 'Medium', 'High']
)

mohs_test

# %% [markdown]
# From the plots and the values I choose to use KMeans Clustering to characterize the Visit Complexity (This looks at `Duration of Visit`, `Anesthetic Amount` and `Number of Stages`). The Elbow plot for KMeans suggests that the data is too high dimensional to characterize the Patient Complexity, therefore I used PCA for patient complexity which looks at everything else. I plan to compare and contrast the continuous values and the categorical values to see how we can predict the complexity of a patient using XGBoost.

# %% [markdown]
# ### Using features to predict `Visit Complexity Category KMeans` with XGBoost

# %%
features = mohs_test.iloc[:, 3:12]

target = mohs_test['Visit Complexity Category KMeans']
le = LabelEncoder()
target_encoded = le.fit_transform(target)
features_encoded = pd.get_dummies(features)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target_encoded, test_size=0.4, random_state=42
)

xgb_model = XGBClassifier(
    objective='multi:softprob',  # for multiclass classification
    eval_metric='mlogloss',      # evaluation metric
    use_label_encoder=True,     # suppress warning for label encoding
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

y_proba = xgb_model.predict_proba(X_test)

lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)

roc_auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
print(f"Multiclass ROC AUC (macro average): {roc_auc:.4f}")

# %% [markdown]
# ### Using features to predict `Visit Duration` with XGBoost

# %%
features = mohs_test.iloc[:, 3:13].copy()
target = mohs_test['Duration of Visit (min)']

for col in features.select_dtypes(include='category').columns:
    features[col] = features[col].cat.codes

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)


# %%
target

# %%
xgb_model_reg = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Fit the regressor on full dataset before saving
xgb_model_reg.fit(features, target)

# Save the trained regressor
joblib.dump(xgb_model_reg, 'xgb_model_regressor.joblib')
cv_scores = cross_val_score(
    xgb_model_reg, 
    features, 
    target, 
    cv=10, 
    scoring='r2'
)

print(f"Mean RMSE: {cv_scores.mean():.2f}")

# %%
# Save the model and feature names
joblib.dump(xgb_model, "xgb_visit_complexity_model.pkl")
joblib.dump(xgb_model_reg, 'xgb_model_regressor.joblib')
joblib.dump(features_encoded.columns.tolist(), "model_features.pkl")
joblib.dump(le, "label_encoder.pkl")

# %% [markdown]
# According to the NIH (https://pubmed.ncbi.nlm.nih.gov/38024184/#:~:text=ROC%20analysis%20is%20a%20powerful,between%20diseased%20and%20nondiseased%20individuals.)
# 
# ROC AUC scores above .80 are considered clinically useful. Therefore using XGBoost to classify Low, Medium and High Complexity is clinically useful. We might have all the models use the predicted values from KMeans and whatever has the highest ROC AUC will be the model we use as the backend for a Streamlit dashboard

# %% [markdown]
# 

# %% [markdown]
# 


