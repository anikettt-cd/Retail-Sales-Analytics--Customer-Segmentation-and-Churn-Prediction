from mlxtend.frequent_patterns import apriori, association_rules

def run_frequent_patterns(transaction_df):
    # Frequent Itemsets with a minimum support of 50%
    frequent_itemsets = apriori(transaction_df, min_support=0.5, use_colnames=True)
    
    # Generate rules based on Confidence
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]