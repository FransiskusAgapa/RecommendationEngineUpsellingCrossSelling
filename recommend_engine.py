import streamlit as st # Streamtlit library for interactive web app interface
import pandas as pd # Pandas library for data handling and manipulation
import numpy as np # Numpy library for numerical computation
from scipy.sparse import csr_matrix # Sparse matric implementation, saves memory
from sklearn.neighbors import NearestNeighbors # KNN algorithm for collaborative filtering
from fpdf import FPDF # PDF file generator
from datetime import datetime # time in pdf file

# Load datasets and cache data for improved performance
# @st.cache_data: Clearly tells Streamlit to load data only once
# persist=True: Keep data cached across app restarts
# show_spinner: Improve UX by reducing visual clutter
@st.cache_data(show_spinner=False)
def load_and_merge(orders_file, products_file, sample_size=100000):
    # Define dtypes for optimization
    orders_dtypes = {
        'order_id': 'int32',
        'product_id': 'int32',
        'add_to_cart_order': 'int16',
        'reordered': 'int8'
    }

    products_dtypes = {
        'product_id': 'int32',
        'product_name': 'category',  # Strings as categories to save memory
        'aisle_id': 'int16',
        'department_id': 'int16'
    }

    # Load CSV with specified dtypes
    orders_df = pd.read_csv(orders_file, dtype=orders_dtypes, usecols=orders_dtypes.keys(),nrows=sample_size)
    products_df = pd.read_csv(products_file, dtype=products_dtypes, usecols=products_dtypes.keys(),nrows=sample_size)

    # debug data
    # st.write("Orders CSV columns:", orders_df.columns.tolist())
    # st.write("Products CSV columns:", products_df.columns.tolist())

    # Merge
    merged_df = orders_df.merge(products_df, on="product_id", how="left")

    #st.write("Merged columns (inside function):", merged_df.columns.tolist())

    # Sampling for speed
    if sample_size and sample_size < len(merged_df):
        merged_df = merged_df.sample(n=sample_size, random_state=42)  # random_state for reproducibility

    return merged_df

# Create a user-item interaction matrix fro collaborative filtering 
@st.cache_resource(show_spinner=False)
def create_matrix_and_model(df):
    # Create interaction matrix (product_id vs order_id)
    user_item_matrix = pd.pivot_table(df, index='product_id', columns='order_id', 
                                    aggfunc='size', fill_value=0)

    # Fit KNN model (optimized)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    model_knn.fit(user_item_matrix)

    return user_item_matrix, model_knn

# Pre-filter products that are not in sample — but for KNN, limit candidates
def filter_top_products(df, n=1000):
    #st.write("Columns inside filter_top_products:", df.columns.tolist())  # Check columns again

    # Filter based on 'product_id' frequency
    top_products = df['product_id'].value_counts().nlargest(n).index
    return df[df['product_id'].isin(top_products)]

def recommend_products(model_knn, user_item_matrix, product_id, n_recommendations=5):
    """
    Function to recommend similar products based on KNN model.
    
    Args:
    - model_knn: Trained KNN model.
    - user_item_matrix: User-item interaction matrix (product_id as index).
    - product_id: Selected product ID to base recommendations on.
    - n_recommendations: Number of products to recommend (default=5).
    
    Returns:
    - List of recommended product IDs.
    """
    # Check if product_id exists in the matrix (important!)
    if product_id not in user_item_matrix.index:
        st.error(f"❌ Product ID {product_id} not found in the dataset. Please choose another product.")
        return []

    # Reshape product_id to use for KNN (expects 2D array)
    product_vector = user_item_matrix.loc[[product_id]]

    # Find nearest neighbors
    distances, indices = model_knn.kneighbors(product_vector, n_neighbors=n_recommendations + 1)  # +1 to exclude itself

    # Flatten indices and get product IDs
    recommended_indices = indices.flatten()
    recommended_ids = user_item_matrix.index[recommended_indices].tolist()

    # Remove the first one if it's the same as input (because nearest neighbor includes itself)
    recommended_ids = [pid for pid in recommended_ids if pid != product_id]

    # Take only n_recommendations final
    return recommended_ids[:n_recommendations]

# === PDF Generator ===
def generate_pdf_report(recommended_products):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Instacart Recommendation Report", ln=True, align="C")
    
    # Add date and time of generation
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Generated on: {current_time}", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)  # Add space

    # Loop through products and print clean ASCII-safe names
    for _, row in recommended_products.iterrows():
        product_id = row['product_id']
        product_name = row['product_name']
        # Ensure product name is ASCII-safe (strip non-ASCII)
        safe_product_name = product_name.encode('ascii', 'ignore').decode('ascii')
        pdf.cell(200, 10, txt=f"Product ID: {product_id} - {safe_product_name}", ln=True)

    # Generate PDF in memory and prepare for download
    pdf_output = pdf.output(dest='S').encode('latin1')

    # Download button in Streamlit
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_output,
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
            with st.spinner('Processing data, please wait...'):
                df = load_and_merge(uploaded_orders, uploaded_products)
                filtered_df = filter_top_products(df)
                user_item_matrix, model_knn = create_matrix_and_model(filtered_df)

                st.subheader('Sample Products')
                sample_products = df[['product_id', 'product_name']].drop_duplicates().sample(10)
                st.table(sample_products)

                # Get unique available products clearly from your sampled dataset
                available_products = df[['product_id', 'product_name']].drop_duplicates()
                product_dict = dict(zip(available_products['product_id'], available_products['product_name']))

                # Clearly display product selection using product ID and name from your sampled data
                selected_product = st.selectbox(
                    "Select a product to recommend similar items:",
                    available_products['product_id'].tolist(),
                    format_func=lambda pid: f"{pid} - {product_dict.get(pid, 'Unknown Product')}"
                )

                if st.button("Recommend"):
                    # Before making recommendations, ensure the product ID is clearly present
                    if selected_product not in user_item_matrix.index:
                        st.error(f"⚠️ Error: The product ID {selected_product} doesn't exist in the current data. Please select another product.")
                    else:
                        recommendations = recommend_products(model_knn, user_item_matrix, selected_product)
                        recommended_products = df[df['product_id'].isin(recommendations)][['product_id', 'product_name']].drop_duplicates()
                        st.success("🔖 Recommended Products:")
                        st.table(recommended_products)

                        # Store recommendations clearly for later PDF generation
                        st.session_state.recommendations = recommended_products
                
                # If recommendations exist, allow generating PDF
                if "recommendations" in st.session_state:
                    if st.button("📄 Generate PDF Report"):
                        generate_pdf_report(st.session_state.recommendations)

    else:
        st.info("ℹ️ Please upload both CSV files to continue.")

if __name__ == "__main__":
    main() # Execute main function
