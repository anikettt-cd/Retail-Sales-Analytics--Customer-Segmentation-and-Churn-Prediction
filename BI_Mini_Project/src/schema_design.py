def create_star_schema(df):
    # Dimension Table: Customer
    dim_customer = df[['CustomerID', 'Segment']].drop_duplicates()
    
    # Dimension Table: Time (Simplified)
    dim_time = pd.DataFrame({'DateID': range(1, 366), 'Year': 2026})
    
    # Fact Table: Sales
    fact_sales = df[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    
    print("Star Schema Layers Created: Fact_Sales, Dim_Customer, Dim_Time")
    return dim_customer, fact_sales