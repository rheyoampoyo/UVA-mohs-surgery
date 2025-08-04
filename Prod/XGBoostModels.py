import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ========== Load and preprocess data ==========
data = pd.read_csv("Prod/mohs_test_with_complexity_scores.csv")

# Drop target columns
X = data.drop(columns=[
    "Duration of Visit (min)",
    "Number of stages",
    "Anesthetic Amount (ml)",
    "Visit Complexity KMeans",
    "Normalized Visit Complexity Score KMeans"
])

X_encoded = pd.get_dummies(X)

# Targets
y_duration = data["Duration of Visit (min)"]
y_stages_raw = data["Number of stages"]
y_anesthetic = data["Anesthetic Amount (ml)"]

# ========== Train/test split (shared) ==========
X_train, X_test, y_d_train, y_d_test, y_s_train_raw, y_s_test_raw, y_a_train, y_a_test = train_test_split(
    X_encoded, y_duration, y_stages_raw, y_anesthetic, test_size=0.2, random_state=42
)

# ========== Train Regressors ==========
duration_regressor = XGBRegressor(n_estimators=100, random_state=42)
duration_regressor.fit(X_train, y_d_train)

anesthetic_regressor = XGBRegressor(n_estimators=100, random_state=42)
anesthetic_regressor.fit(X_train, y_a_train)

# ========== Encode and Train Classifier ==========
le_stages = LabelEncoder()
y_s_train = le_stages.fit_transform(y_s_train_raw)
y_s_test = le_stages.transform(y_s_test_raw)

stages_regressor = XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=42)
stages_regressor.fit(X_train, y_s_train)

# ========== Save models and encoder ==========
output_dir = "Prod/models"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(duration_regressor, os.path.join(output_dir, "xgb_regressor_visit_duration.pkl"))
joblib.dump(anesthetic_regressor, os.path.join(output_dir, "xgb_regressor_anesthetic_amount.pkl"))
joblib.dump(stages_regressor, os.path.join(output_dir, "xgb_classifier_number_of_stages.pkl"))
joblib.dump(le_stages, os.path.join(output_dir, "label_encoder_number_of_stages.pkl"))
joblib.dump(X_encoded.columns.tolist(), os.path.join(output_dir, "model1_features.pkl"))

# ========== Print metrics ==========
def print_metrics(y_true, y_pred, label):
    print(f"--- {label} ---")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R^2:", r2_score(y_true, y_pred))

print_metrics(y_d_test, duration_regressor.predict(X_test), "Visit Duration")
print_metrics(y_a_test, anesthetic_regressor.predict(X_test), "Anesthetic Amount")

print("--- Number of Stages ---")
stage_preds = stages_regressor.predict(X_test)
decoded_preds = le_stages.inverse_transform(stage_preds)
print("Accuracy:", accuracy_score(y_s_test_raw, decoded_preds))