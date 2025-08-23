# 🌦️ Retail Sales Forecasting with Weather-Enhanced Machine Learning

**Toronto Metropolitan University – Major Research Project (MRP)**

## 📄 Description

This project investigates how **external weather features** (temperature, precipitation, wind, solar radiation, etc.) influence **retail e-commerce sales forecasting** using the **DataCo Supply Chain dataset** (2015–2017) enriched with **Visual Crossing weather data**.

Project is implemented a two-stage pipeline:

1. **Forecasting** – Benchmarking (Random Forest) and machine learning models (XGBoost, CatBoost) against deep learning methods (LSTM).
2. **Feature Interpretation** – Using SHAP analysis to identify the most important predictors of sales.

The best-performing model, **Stacked LSTM**, achieved **R² = 0.972**, **sMAPE = 5.82%**, **MAE = 28.50**, and **RMSE = 57.04**, demonstrating strong predictive capability for daily retail sales.

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

1. Which forecasting techniques most accurately predict retail sales? Also, which features (transactional, weather, lag) are most important for retail sales forecasting?
2. Do weather variables (e.g., temperature, precipitation) significantly impact retail sales predictions?
3. Do lag features improve forecasting accuracy in retail sales forecasting?

---

## 🔄 Workflow & Code Files

### **Data Documentation**

* `01_Data_Co_Data_Dictionary.ipynb` – Provides definitions of DataCo dataset fields (53 columns: transactional, customer, product, and shipping attributes).
* `02_Weather_Features_Data_Dictionary.ipynb` – Describes weather attributes fetched from Visual Crossing API (temperature, humidity, precipitation, snow, solar radiation, etc.).

### **Data Preparation**

* `03_Lat_Lon.ipynb` – Extracts and prepares ~11,000 unique latitude–longitude pairs for weather API queries. Outputs unique_lat_lon.csv.
* `04_Data_Co.ipynb` – Initial cleaning of DataCo raw file (handling nulls, dropping masked fields, converting dates to datetime, deriving time features).
* `07_aggregate_to_daily.py` – Aggregates raw transactional data into **daily-location-level sales** with pivoted categorical features.
* `06_Weather_features_fetching.ipynb` – Fetches weather features (temperature, humidity, precipitation, etc.) from **Visual Crossing API**.
* `08_Data_Cleaning_Weather.ipynb` – Cleans and imputes missing weather data.
* `09_Feature_Preparation.ipynb` – Creates lag features (1, 7, 30 days), and merges sales + weather datasets.

### **Exploratory Data Analysis (EDA)**

* `05_EDA.ipynb` – Initial exploratory analysis (distributions, seasonal decomposition, correlation heatmaps). It also generates an **automated data profiling report** (`DataCo_report.html`) using `ydata-profiling`.
* `10_EDA_V2.ipynb` – Refined exploratory analysis (category-level trends, weather-sales sensitivity). It also do correlation-based filtering. This is then used for feature selection methods like Randdom forest feature importance and Recursive Feature Elimination (RFE)

### **Feature Importance & Explainability (XAI)**

* `11_shap_validation_V2.py` – Runs **SHAP analysis**, saves top N most informative features for downstream modeling.
* `shap_validation_V2_with_plots.py` – Extended SHAP script, it does the following:
  - Saves top features dataset (`Feature_Selected_SHAP_V2.csv`)  
  - Generates SHAP plots:  
    - `shap_summary.png` – Beeswarm plot showing overall feature impact on predictions  
    - `shap_bar.png` – Bar chart of mean absolute SHAP values  
   - Output is stored in `/MRP/SHAP Plots/` folder in GitHub 

### **Forecasting Models with albation studies**

* `01_RF.ipynb` – Random Forest with MinMax scaling.
* `02_CatBoost.ipynb` – CatBoost regressor with categorical feature handling.
* `03_XGB.ipynb` – XGBoost with hyperparameter tuning.
* `04_LSTM.ipynb` – Stacked LSTM with Keras (time-series reshaping, feature scaling).



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
```
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn shap xgboost catboost tensorflow ydata-profiling

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

Output: `DataCo_report.html` and Final_Selected_Features_new.csv

### 4. Train Forecasting Models

Run each notebook:

* `01_RF.ipynb` → Random Forest
* `02_CatBoost.ipynb` → CatBoost
* `03_XGB.ipynb` → XGBoost
* `04_LSTM.ipynb` → Stacked LSTM

### 5. Feature Importance with SHAP
* Option A. Run without plots
```bash
python 11_shap_validation_V2.py
```
Output: `Feature_Selected_SHAP_V2.csv`

* Option B. Run with plots
```bash
python shap_validation_V2_with_plots.py
```
Outputs:
- Feature_Selected_SHAP_V2.csv and two generated plots.
- One plot is stored in `/MRP/SHAP Plots/shap_summary.png` folder in GitHub 
- Other plot is stored in `/MRP/SHAP Plots/shap_bar.png` folder in GitHub 

📊 Example Plots (Explainability)

* SHAP Beeswarm Plot – shows how each feature drives sales predictions:
* SHAP Bar Plot – average magnitude of each feature’s impact:

---

## 💡 Key Findings

* **Stacked LSTM** consistently outperformed ensemble models (with R² = 0.972), but at higher computational cost.
* **Tree-based methods (XGBoost, CatBoost)** achieved competitive accuracy with much faster runtime.
* **Inclusion of the weather feature tempmax** led to a 0.3% improvement in explained variance for the LSTM model. 
* **Combining lag and weather features** further improved accuracy, yielding a 3.9% gain in explained variance and a 4.64% reduction in sMAPE.

---

## ⚠️ Limitations

* Missing values in weather features required imputation (e.g., solar radiation gaps).
* Transactional data lacked 2018 beyond January, limiting horizon testing.
* Limited data availability (only around 3 years of records).
* Insufficient data for reinforcement learning experiments initially planned.
* Potential bias from short time horizon in capturing long-term seasonality.



---

## 🚀 Future Work

* Extend forecasting with GRU and CNN_LSTM hybrid architectures to enhance temporal feature learning.
* Integrate forecasted sales into reinforcement learning agents for inventory optimization.
* Develop category-specific forecasting for weather-sensitive and high-demand products to capture heterogeneity.


---

## 📜 License

GNU General Public License v3.0

---

