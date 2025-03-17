import streamlit as st # Streamtlit library for interactive web app interface
import pandas as pd # Pandas library for data handling and manipulation
import numpy as np # Numpy library for numerical computation
from scipy.sparse import csr_matrix # Sparse matric implementation, saves memory
from sklearn.neighbors import NearestNeighbors # KNN algorithm for collaborative filtering
from fpdf import FPDF

# **TODO**
# - Add CSV Submission 
# - Printable File Generator

# Load datasets and cache data for improved performance
# @st.cache_data: Clearly tells Streamlit to load data only once
# persist=True: Keep data cached across app restarts
# show_spinner: Improve UX by reducing visual clutter
@st.cache_data(show_spinner=False)
def load_and_merge(orders_file, products_file, sample_size=500000):
    orders_df = pd.read_csv(orders_file)  # Reads uploaded orders CSV into DataFrame
    products_df = pd.read_csv(products_file)  # Reads products CSV
    merged_df = orders_df.merge(products_df, on="product_id", how="left")
    
    # Optional sampling clearly to improve performance
    if sample_size and sample_size < len(merged_df):
        merged_df = merged_df.sample(n=sample_size)
    return merged_df

# Create a user-item interaction matrix fro collaborative filtering 
@st.cache_resource(show_spinner=False)
def create_matrix_and_model(df, num_orders=10000, num_products=1000):
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

    #Fit KNN model using cosine similarity for recommending similar products
    model_knn = NearestNeighbors(metric="cosine",algorithm="brute") # Initialzie KNN model using cosine similarity metric
    model_knn.fit(matrix) # Trains the KNN model on the provided user-item matrix
    return user_item_matrix, model_knn

# Recommend similar products based on trained collaborative filtering model 
def recommend_products(model,data,product_id,n_recommendations=5):
    product_index = data.index.get_loc(product_id) # Finds index location of the selected product 
    distances, indices = model.kneighbors(data.iloc[product_index,:].values.reshape(1,-1),n_neighbors=n_recommendations + 1) # Identify nearest neighbors to the selected product 
    rec_indices = indices.flatten()[1:] # Extracts indices of recommend products excluding the selected product itself
    recommendations = data.iloc[rec_indices].index.tolist() # Retrieves product IDs of recommended products
    return recommendations

# == PDF File Generator ==
def generate_pdf_report(recommended_products):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "📌 Instacart Recommendation Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    for _, row in recommended_products.iterrows():
        pdf.cell(200, 10, txt=f"Product ID: {row['product_id']} - {row['product_name']}", ln=True)

    pdf_output = pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf.output(dest='S').encode('latin1'),
        file_name="recommendation_report.pdf",
        mime="application/pdf"
    )

# ===== Streamlit App =====
def main():
    st.title("🛒 Instacart Recommendation Engine") # App title 

    st.subheader("📤 Step 1: Upload Required CSV Files")

    uploaded_orders = st.file_uploader("Upload 'order_products__prior.csv' CSV file", type=['csv'], key="orders")
    uploaded_products = st.file_uploader("Upload 'products.csv' CSV file", type=['csv'], key="products")

    if uploaded_orders and uploaded_products:
        valid_upload = True  # check clearly files validity

        # Check orders file is not empty clearly
        uploaded_orders.seek(0)
        if uploaded_orders.read() == b'':
            st.error("⚠️ Error: 'order_products__prior.csv' is empty.")
            valid_upload = False
        uploaded_orders.seek(0)  # Reset pointer for CSV read clearly

        # Check products file clearly
        uploaded_products.seek(0)
        if uploaded_products.read() == b'':
            st.error("⚠️ Error: 'products.csv' is empty.")
            valid_upload = False
        uploaded_products.seek(0)  # Reset pointer clearly again

        if valid_upload:
            df = load_and_merge(uploaded_orders, uploaded_products)  # Cached clearly for speed

            user_item_matrix, model_knn = create_matrix_and_model(df)  # Cached clearly

            st.subheader('Sample Products')
            sample_products = df[['product_id', 'product_name']].drop_duplicates().sample(10)
            st.table(sample_products)

            available_products = df[['product_id', 'product_name']].drop_duplicates()
            product_dict = dict(zip(available_products['product_id'], available_products['product_name']))

            selected_product = st.selectbox(
                "Select a product to recommend similar items:",
                available_products['product_id'].tolist(),
                format_func=lambda pid: f"{pid} - {product_dict.get(pid, 'Unknown Product')}"
            )

            if st.button("Recommend"):
                recommendations = recommend_products(model_knn, user_item_matrix, selected_product)
                recommended_products = df[df['product_id'].isin(recommendations)][['product_id', 'product_name']].drop_duplicates()
                st.success("🔖 Recommended Products:")
                st.table(recommended_products)

                st.session_state.recommendations = recommended_products

            if "recommendations" in st.session_state:
                if st.button("📄 Generate PDF Report"):
                    generate_pdf_report(st.session_state.recommendations)

    else:
        st.info("ℹ️ Please upload both CSV files to continue.")

if __name__ == "__main__":
    main() # Execute main function
