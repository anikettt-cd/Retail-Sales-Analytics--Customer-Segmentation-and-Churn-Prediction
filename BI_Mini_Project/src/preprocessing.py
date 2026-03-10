import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(file_path):
    # Loading the dataset
    df = pd.read_csv(file_path)
    
    # 1. Handling Missing Values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # 2. Outlier Removal (Simple IQR method)
    Q1 = df['Monetary'].quantile(0.25)
    Q3 = df['Monetary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Monetary'] < (Q1 - 1.5 * IQR)) | (df['Monetary'] > (Q3 + 1.5 * IQR)))]
    
    # 3. Scaling for BI Models (Clustering/Classification)
    scaler = StandardScaler()
    features = ['Recency', 'Frequency', 'Monetary']
    df_scaled = scaler.fit_transform(df[features])
    
    return df, df_scaled