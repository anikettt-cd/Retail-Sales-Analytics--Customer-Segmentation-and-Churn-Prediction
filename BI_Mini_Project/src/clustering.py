import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(df, scaled_data):
    # 1. Run K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(scaled_data)
    
    # 2. Create the Plot for your Report
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Segment', palette='viridis')
    plt.title('BI Insight: Customer Segmentation (Recency vs Monetary)')
    
    # Save to your figures folder
    plt.savefig('reports/figures/cluster_plot.png')
    print("✓ Cluster visualization saved as reports/figures/cluster_plot.png")
    
    # Return the updated dataframe and the centroids
    return df, kmeans.cluster_centers_