import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Prod/mohs_test_with_complexity_scores.csv")

X = data.drop(columns=[
    "Duration of Visit (min)",
    "Number of stages",
    "Anesthetic Amount (ml)",
    "Visit Complexity KMeans",
    "Normalized Visit Complexity Score KMeans"
])
X_encoded = pd.get_dummies(X)

y_duration = data["Duration of Visit (min)"]
y_stages_raw = data["Number of stages"]
y_anesthetic = data["Anesthetic Amount (ml)"]

X_train, X_test, y_d_train, y_d_test, y_s_train_raw, y_s_test_raw, y_a_train, y_a_test = train_test_split(
    X_encoded, y_duration, y_stages_raw, y_anesthetic, test_size=0.2, random_state=42
)

duration_regressor = XGBRegressor(n_estimators=100, random_state=42)
duration_regressor.fit(X_train, y_d_train)

anesthetic_regressor = XGBRegressor(n_estimators=100, random_state=42)
anesthetic_regressor.fit(X_train, y_a_train)

stages_regressor = XGBRegressor(n_estimators=100, random_state=42)
stages_regressor.fit(X_train, y_s_train_raw)

le_stages = LabelEncoder()
y_s_train_cls = le_stages.fit_transform(y_s_train_raw)
y_s_test_cls = le_stages.transform(y_s_test_raw)

stages_classifier = XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=42)
stages_classifier.fit(X_train, y_s_train_cls)

output_dir = "Prod/models"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(duration_regressor, os.path.join(output_dir, "xgb_regressor_visit_duration.pkl"))
joblib.dump(anesthetic_regressor, os.path.join(output_dir, "xgb_regressor_anesthetic_amount.pkl"))
joblib.dump(stages_regressor, os.path.join(output_dir, "xgb_regressor_number_of_stages.pkl"))
joblib.dump(stages_classifier, os.path.join(output_dir, "xgb_classifier_number_of_stages.pkl"))
joblib.dump(le_stages, os.path.join(output_dir, "label_encoder_number_of_stages.pkl"))
joblib.dump(X_encoded.columns.tolist(), os.path.join(output_dir, "model1_features.pkl"))

def print_regressor_metrics(y_train, y_test, y_train_pred, y_test_pred, label):
    print(f"\n--- {label} (Regressor) ---")
    print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("Train R^2:", r2_score(y_train, y_train_pred))
    print("Test R^2:", r2_score(y_test, y_test_pred))

def print_classifier_metrics(y_train_true, y_test_true, y_train_pred, y_test_pred, label):
    print(f"\n--- {label} (Classifier) ---")
    print("Train Accuracy:", accuracy_score(y_train_true, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test_true, y_test_pred))
    print("Train Precision (weighted):", precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0))
    print("Test Precision (weighted):", precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0))
    print("Train Recall (weighted):", recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0))
    print("Test Recall (weighted):", recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0))

duration_train_pred = duration_regressor.predict(X_train)
duration_test_pred = duration_regressor.predict(X_test)

anesthetic_train_pred = anesthetic_regressor.predict(X_train)
anesthetic_test_pred = anesthetic_regressor.predict(X_test)

stages_train_pred = stages_regressor.predict(X_train)
stages_test_pred = stages_regressor.predict(X_test)

stages_train_cls_pred = stages_classifier.predict(X_train)
stages_test_cls_pred = stages_classifier.predict(X_test)

print_regressor_metrics(y_d_train, y_d_test, duration_train_pred, duration_test_pred, "Visit Duration")
print_regressor_metrics(y_a_train, y_a_test, anesthetic_train_pred, anesthetic_test_pred, "Anesthetic Amount")
print_regressor_metrics(y_s_train_raw, y_s_test_raw, stages_train_pred, stages_test_pred, "Number of Stages")

print_classifier_metrics(y_s_train_cls, y_s_test_cls, stages_train_cls_pred, stages_test_cls_pred, "Number of Stages")