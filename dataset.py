import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('/Users/aniketsaini/Desktop/bi_project/BI_Mini_Project/data/raw/retail_data.csv')

print("--- FIGURE 1.1: RAW DATASET (FIRST 30 ROWS) ---")
# Passing 30 into head() shows more rows
print(df.head(30).to_string()) 

# Scaling Logic
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

print("\n--- FIGURE 1.2: PREPROCESSED DATASET (SCALED - 30 ROWS) ---")
print(df_scaled[['Recency', 'Frequency', 'Monetary']].head(30).to_string())