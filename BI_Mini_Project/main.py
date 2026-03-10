import pandas as pd
from src.preprocessing import clean_data
from src.clustering import run_kmeans
from src.classification import run_classification

def main():
    print("--- Starting BI Project Pipeline ---")
    
    # 1. Preprocessing
    df, scaled_data = clean_data('data/raw/retail_data.csv')
    print("✓ Data Cleaned and Scaled.")

    # 2. Clustering (Customer Segmentation)
    df, centroids = run_kmeans(df, scaled_data)
    print(f"✓ Clustering Complete. Found {len(centroids)} segments.")

    # 3. Classification (Churn Prediction)
    accuracy, report, model = run_classification(df)
    
    print("\n--- Project Metrics ---")
    print(f"Model Accuracy Score: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", report)
    
    # Save the final processed data
    df.to_csv('data/processed/final_bi_report_data.csv', index=False)
    print("\n✓ Processed data saved to data/processed/")
    
    # The Link in main.py
    df_with_segments, centroids = run_kmeans(df, scaled_data)  # Step 1
    accuracy, report, model = run_classification(df_with_segments) # Step 2

if __name__ == "__main__":
    main()