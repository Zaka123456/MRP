# 🌦️ Retail Sales Forecasting with Weather-Enhanced Machine Learning

**Toronto Metropolitan University – Major Research Project (MRP)**

## 📄 Description

This project investigates how **external weather features** (temperature, precipitation, wind, solar radiation, etc.) influence **retail e-commerce sales forecasting** using the **DataCo Supply Chain dataset** (2015–2017) enriched with **Visual Crossing weather data**.

We implement a two-stage pipeline:

1. **Forecasting** – Benchmarking statistical (ARIMA) and machine learning models (Random Forest, XGBoost, CatBoost) against deep learning methods (LSTM, CNN-LSTM, GRU).
2. **Feature Interpretation** – Using SHAP analysis after RFE-based feature selection to identify the most important predictors of sales.

The best-performing model, **Stacked LSTM**, achieved **R² = 0.974**, **sMAPE = 5.09%**, **MAE = 25.9**, and **RMSE = 54.6**, demonstrating strong predictive capability for daily retail sales.

---

## 📊 Dataset Description

**1. DataCo Supply Chain Dataset (Kaggle, 2019)**

* \~180,000 rows, 53 columns (2015–2017 transactions)
* Key features: order details, shipment times, customer info, product categories, location (lat/lon)

**2. Visual Crossing Weather Data (2015–2017)**

* Features: tempmax, tempmin, temp, humidity, precipitation, precipitation type, snow, windspeed, cloudcover, solar radiation
* Extracted for \~11,000 unique store coordinates via batch API queries

---

## ❓ Research Questions

1. Which forecasting techniques most accurately predict retail sales?
2. Which features (transactional, weather, lag) are most important for retail sales forecasting?
3. Do weather variables (e.g., temperature, precipitation) significantly impact retail sales predictions?
4. Do lag features improve forecasting accuracy in retail sales forecasting?

---

## 🔄 Workflow & Code Files



### **Data Preparation**

* `07_aggregate_to_daily.py` – Aggregates raw transactional data into **daily-location-level sales** with pivoted categorical features.
* `06_Weather_features_fetching.ipynb` – Fetches weather features (temperature, humidity, precipitation, etc.) from **Visual Crossing API**.
* `08_Data_Cleaning_Weather.ipynb` – Cleans and imputes missing weather data.
* `09_Feature_Preparation.ipynb` – Creates lag features (1, 7, 30 days), rolling means, and merges sales + weather datasets.

### **Exploratory Data Analysis (EDA)**

* `05_EDA.ipynb` – Initial exploratory analysis (distributions, seasonal decomposition, correlation heatmaps).
* `10_EDA_V2.ipynb` – Refined exploratory analysis (category-level trends, weather-sales sensitivity).
* `profiling_report.py` – Generates an **automated data profiling report** (`DataCo_Profiling_Report.html`) using `ydata-profiling`.

### **Forecasting Models**

* `01_RF.ipynb` – Random Forest with MinMax scaling.
* `02_CatBoost.ipynb` – CatBoost regressor with categorical feature handling.
* `03_XGB.ipynb` – XGBoost with hyperparameter tuning.
* `04_LSTM.ipynb` – LSTM and Stacked LSTM with Keras (time-series reshaping, feature scaling).

### **Feature Importance**

* `11_shap_validation_V2.py` – Runs **SHAP analysis** after RFE, saves top N most informative features for downstream modeling.

---

## 🧰 Tools & Technologies

* **Python**: Pandas, NumPy, Scikit-learn
* **Deep Learning**: TensorFlow / Keras
* **Machine Learning**: Random Forest, XGBoost, CatBoost
* **Visualization**: Matplotlib, Seaborn
* **Explainability**: SHAP
* **Data Profiling**: ydata-profiling (pandas-profiling successor)
* **Data APIs**: Visual Crossing
* **Environment**: Jupyter Notebooks, VS Code

---

## 📦 Usage Instructions

### 1. Prepare Environment

```bash
git clone <repo_link>
cd Retail-Sales-Forecasting
pip install -r requirements.txt
```

### 2. Run Data Preparation

```bash
# Aggregate transactional data to daily-level
python 07_aggregate_to_daily.py
```

Output: `Data_Co_Daily_By_Location.csv`

```bash
# Run notebooks sequentially for weather fetching, cleaning, and merging
jupyter notebook 06_Weather_features_fetching.ipynb
jupyter notebook 08_Data_Cleaning_Weather.ipynb
jupyter notebook 09_Feature_Preparation.ipynb
```

### 3. Exploratory Data Analysis

```bash
jupyter notebook 05_EDA.ipynb
jupyter notebook 10_EDA_V2.ipynb
```

Generate profiling report (optional but recommended):

```bash
python profiling_report.py
```

Output: `reports/DataCo_Profiling_Report.html`

### 4. Train Forecasting Models

Run each notebook:

* `01_RF.ipynb` → Random Forest
* `02_CatBoost.ipynb` → CatBoost
* `03_XGB.ipynb` → XGBoost
* `04_LSTM.ipynb` → LSTM & Stacked LSTM

### 5. Feature Importance with SHAP

```bash
python 11_shap_validation_V2.py
```

Output: `Feature_Selected_SHAP_V2.csv`

---

## 💡 Key Findings

* **Stacked LSTM** consistently outperformed classical and ensemble models (R² = 0.974).
* **Weather features** improved forecast accuracy, confirming their predictive role.
* **Lag features** (1-day, 7-day) provided significant boost in temporal learning.
* Product-category-specific models capture heterogeneous weather sensitivity.

---

## ⚠️ Limitations

* Missing values in weather features required imputation (e.g., solar radiation gaps).
* Transactional data lacked 2018 beyond January, limiting horizon testing.


---

## 🚀 Future Work

* Extend forecasting to monthly granularity for operational decision-making.
* Implement **Prophet** and **Temporal Fusion Transformer (TFT)** for comparison.
* Perform **SHAP analysis** on deep learning models for interpretability.
* Develop full **RL-based inventory optimization** pipeline (SARSA, DQN, PPO).

---

## 📚 References

(Full list in report, key examples below)

* Makridakis et al. (2022). *M5 Competition: Findings & Conclusions*
* Taghizadeh (2017). *ANN for weather-sensitive retail demand*
* Sarker (2021). *Machine Learning: Algorithms & Research Directions*
* Sarker (2021). *Deep Learning: Techniques, Taxonomy, Applications*

---

## 📜 License

GNU General Public License v3.0

---

