# shap_validation_after_rfe.py
# Validates feature importance using SHAP after RFE selection.
# Now also saves the top N most important features as a new dataset for modeling.

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# === Config ===
DATA_FILE = "./final_datasets/Feature_Selected_Step1_new2.csv"  # Full feature set (before RFE pruning)
TARGET = "Sales"
TOP_SHAP = 25   # Number of top features to display
TOP_EXPORT = 20  # Number of top features to save for modeling
OUTPUT_FILE = "./final_datasets/Feature_Selected_SHAP_V2.csv"

# Load dataset
print("Loading dataset for SHAP analysis...")
df = pd.read_csv(DATA_FILE)
df["date_only"] = pd.to_datetime(df["date_only"], errors='coerce')

# Focus on Train/Validation (2015–2016)
df_train = df[df["date_only"].dt.year.isin([2015, 2016])].copy()

# Identify features
features = [col for col in df_train.columns if col not in ["date_only", "latitude", "longitude", TARGET]]
X = df_train[features]
y = df_train[TARGET]

# Train a Random Forest for SHAP analysis
print("Training Random Forest for SHAP analysis...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Compute SHAP values
print("Computing SHAP values...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Rank features by mean absolute SHAP value
shap_importance = pd.DataFrame({
    "Feature": features,
    "Mean_SHAP": np.abs(shap_values).mean(axis=0)
}).sort_values(by="Mean_SHAP", ascending=False)

# Display top features
print(f"\n=== Top {TOP_SHAP} Features by SHAP (Full Set) ===")
print(shap_importance.head(TOP_SHAP))

# Check where 'temp' ranks
if "temp" in shap_importance["Feature"].values:
    rank_temp = shap_importance.reset_index(drop=True).index[shap_importance["Feature"] == "temp"].tolist()[0] + 1
    print(f"\nTemperature ('temp') SHAP rank: {rank_temp} out of {len(shap_importance)} features.")
else:
    print("\nTemperature ('temp') not found as a feature (may have been renamed or removed earlier).")

# === Save top N features (with target + date for downstream models) ===
top_features = shap_importance["Feature"].head(TOP_EXPORT).tolist()
export_cols = ["date_only"] + top_features + [TARGET]  # Keep target and date
df_export = df[export_cols].copy()

df_export.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved dataset with Top {TOP_EXPORT} SHAP features to: {OUTPUT_FILE}")