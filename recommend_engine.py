import streamlit as st # Streamtlit library for interactive web app interface
import pandas as pd # Pandas library for data handling and manipulation
import numpy as np # Numpy library for numerical computation
from scipy.sparse import csr_matrix # Sparse matric implementation, saves memory
from sklearn.neighbors import NearestNeighbors # KNN algorithm for collaborative filtering

# Load datasets and cache data for improved performance
def load_data(sample_size=500000): #  # Loads only a small/simple subset (first 500,000 rows) for safe load, take this out when cloud deploy
    orders = pd.read_csv("InstaCartMarketAnalysis/order_products__prior.csv",nrows=sample_size) # Loads Instacart order data
    products = pd.read_csv("InstaCartMarketAnalysis/products.csv") # Loads Instacart products data 
    merged_df = orders.merge(products,on="product_id") # Merges orders and product datasets for complete information
    return merged_df 

# Create a user-item interaction matrix fro collaborative filtering 
def create_matrix(df, num_orders=10000, num_products=1000):
    # Reduce dataset to manageable size for pivot
    top_orders = df['order_id'].value_counts().head(num_orders).index
    top_products = df['product_id'].value_counts().head(num_products).index

    df_reduced = df[df['order_id'].isin(top_orders) & df['product_id'].isin(top_products)]

    # Generate pivot table safely on reduced dataset
    user_item_matrix = df_reduced.pivot_table(
        index='product_id', columns='order_id', values='add_to_cart_order', fill_value=0
    )

    # Sparse matrix to optimize memory
    matrix = csr_matrix(user_item_matrix.values)
    return matrix, user_item_matrix

# Fit KNN model using cosine similarity for recommending similar products
def fit_model(matrix):
    model_knn = NearestNeighbors(metric="cosine",algorithm="brute") # Initialzie KNN model using cosine similarity metric
    model_knn.fit(matrix) # Trains the KNN model on the provided user-item matrix
    return model_knn 

# Recommend similar products based on trained collaborative filtering model 
def recommend_products(model,data,product_id,n_recommendations=5):
    product_index = data.index.get_loc(product_id) # Finds index location of the selected product 
    distances, indices = model.kneighbors(data.iloc[product_index,:].values.reshape(1,-1),n_neighbors=n_recommendations + 1) # Identify nearest neighbors to the selected product 
    rec_indices = indices.flatten()[1:] # Extracts indices of recommend products excluding the selected product itself
    recommendations = data.iloc[rec_indices].index.tolist() # Retrieves product IDs of recommended products
    return recommendations

# ===== Streamlit App =====
def main():
    st.title("\n🛒 Instacart Recommendation Engine") # Sets title of the streamlit app 
    df = load_data() # Loads datasets for recommendation engine

    st.subheader("\nSample Products") # Subtitle for sample product display
    sample_products = df[["product_id","product_name"]].drop_duplicates().sample(10) # Randomly selects samples products to display to users
    st.table(sample_products) # Displays the selected sample products in table format 

    matrix, user_item_matrix = create_matrix(df) # Creates interaction matrix needed for collaborative filtering
    model_knn = fit_model(matrix) # Fits the KNN model on interaction matrix

    # Local test only use available ID's within specified subset
    available_product_ids = user_item_matrix.index.to_list() # Get available IDs from the redduced datasets
    selected_product = st.selectbox(
        "Choose a product ID to recommend similar products:",
        available_product_ids
    )
    
    # NOTE: Use this variable below on production
    # selected_product = st.number_input("Enter a Product ID to Recommend Similar Products:",min_value=int(df["product_id"].min()),max_value=int(df["product_id"].max()),value=int(sample_products.iloc[0]["product_id"])) # User input for selecting product ID to get recommendation

    if st.button("Recommend"): # Button to trigger recommendation
        recommendations = recommend_products(model_knn,user_item_matrix,selected_product) # Generates recommended products based on user selection
        recommended_products = df[df["product_id"].isin(recommendations)][["product_id","product_name"]].drop_duplicates() # Filters recommended products for displaying product names
        st.success("\n🔖 Recommended Products:") # Filters recommended products for displaying products names
        st.table(recommended_products) # Shows recommended product in table format

if __name__ == "__main__":
    main() # Execute main function
