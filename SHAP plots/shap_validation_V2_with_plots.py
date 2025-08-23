# shap_validation_V2_with_plots.py
# Validates feature importance using SHAP after RFE selection.
# Saves top features dataset and also generates SHAP plots.

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# === Config ===
DATA_FILE = "./final_datasets/Feature_Selected_Step1_new2.csv"
TARGET = "Sales"
TOP_SHAP = 25
TOP_EXPORT = 20
OUTPUT_FILE = "./final_datasets/Feature_Selected_SHAP_V2.csv"
PLOT_DIR = "./results/shap_plots"

# Create directory for plots
os.makedirs(PLOT_DIR, exist_ok=True)

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

# Train a Random Forest
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

# Save dataset with Top SHAP features
top_features = shap_importance["Feature"].head(TOP_EXPORT).tolist()
export_cols = ["date_only"] + top_features + [TARGET]
df_export = df[export_cols].copy()
df_export.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved dataset with Top {TOP_EXPORT} SHAP features to: {OUTPUT_FILE}")

# === Generate SHAP plots ===
print("\nGenerating SHAP plots...")

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"), bbox_inches="tight")
plt.close()

# Bar plot of mean SHAP values
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig(os.path.join(PLOT_DIR, "shap_bar.png"), bbox_inches="tight")
plt.close()

print(f"SHAP plots saved in: {PLOT_DIR}")
