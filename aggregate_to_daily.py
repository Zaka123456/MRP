# aggregate_to_daily_by_location.py
"""
Converts transaction-level data into daily-location level dataset:
- Aggregates numeric columns by date + Latitude + Longitude
- Pivots top-3 values of key categorical variables
- Includes Customer Segment and Order Status
- Output: Data_Co_Daily_By_Location.csv
"""

import pandas as pd
from functools import reduce

def top_k_or_other(df, col, k=3):
    """Replace infrequent categories with 'Other'."""
    top_k = df[col].value_counts().nlargest(k).index
    df[col] = df[col].apply(lambda x: x if x in top_k else "Other")
    return df

def main():
    # === Load Dataset ===
    input_file = "Data_Co_cleaned.csv"
    output_file = "Data_Co_Daily_By_Location.csv"

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # === Show Columns ===
    print("\n--- Columns in the dataset ---")
    print(df.columns.tolist())

    # === Create Date Column ===
    df["date_only"] = pd.to_datetime(df["order date (DateOrders)"], errors='coerce').dt.date

    # === Group by Date + Location ===
    group_cols = ["date_only", "Latitude", "Longitude"]

    agg_rules = {
        "Sales": "sum",
        "Order Item Total": "sum",
        "Order Profit Per Order": "sum",
        "Benefit per order": "sum",
        "Order Item Quantity": "sum",
        "Order Item Discount": "mean",
        "Order Item Discount Rate": "mean",
        "Order Item Profit Ratio": "mean",
        "Days for shipping (real)": "mean",
        "Days for shipment (scheduled)": "mean",
        "Late_delivery_risk": "mean",
        "Order Id": pd.Series.nunique,
        "Customer Id": pd.Series.nunique,
        "Product Card Id": pd.Series.nunique
    }

    print("Aggregating numeric features by date and location...")
    daily_grouped = df.groupby(group_cols).agg(agg_rules).reset_index()

    # === Pivot Key Categorical Columns (Top-3 only) ===
    categorical_cols = [
        "Type",
        "Delivery Status",
        "Market",
        "Order Region",
        "Shipping Mode",
        "Customer Segment",   # NEW
        "Order Status"        # NEW
    ]

    pivot_tables = []

    for col in categorical_cols:
        print(f"Pivoting top-3 values for: {col}")
        df = top_k_or_other(df, col, k=3)

        pivot = pd.pivot_table(
            df,
            index=group_cols,
            columns=col,
            values="Order Id",
            aggfunc="count",
            fill_value=0
        )
        pivot.columns = [f"{col}_{c}" for c in pivot.columns]
        pivot_tables.append(pivot.reset_index())

    # === Merge All ===
    print("Merging numeric and categorical features...")
    final_df = reduce(lambda left, right: pd.merge(left, right, on=group_cols, how="left"),
                      [daily_grouped] + pivot_tables)

    # === Save Output ===
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Final dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
