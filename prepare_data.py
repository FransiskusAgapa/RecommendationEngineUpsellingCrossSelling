# NOTE: Unless data change Run this Once, this file saves processed data once and reuse it (Best Performance)
import pandas as pd

orders = pd.read_csv('InstaCartMarketAnalysis/order_products__prior.csv')
products = pd.read_csv('InstaCartMarketAnalysis/products.csv')
merged_df = orders.merge(products, on='product_id', how='left')

# Save as compressed file clearly for quicker future loading
merged_df.to_pickle('merged_instacart.pkl')
