import streamlit as st
import requests
import os

# Docker Compose and Kubernetes each provide a different service DNS name.
# Cloud Run provides a public HTTPS URL.  Configuration, rather than code
# changes, lets the same UI run in all three places.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="E-commerce Recommendation App",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-commerce Product Recommendation System")

st.write(
    "Select a product and get similar product recommendations based on category, price, and rating."
)
st.caption(f"API endpoint: {API_BASE_URL}")

# Fetch products from FastAPI
try:
    products_response = requests.get(f"{API_BASE_URL}/products")

    if products_response.status_code == 200:
        products = products_response.json()
    else:
        st.error("Unable to fetch products from API.")
        st.stop()

except requests.exceptions.ConnectionError:
    st.error("FastAPI backend is not running. Please start the backend first.")
    st.stop()


# Create dropdown options
product_options = {
    f"{product['product_name']} | ₹{product['price']} | Rating: {product['rating']}": product["product_id"]
    for product in products
}

selected_product_label = st.selectbox(
    "Select a product",
    list(product_options.keys())
)

selected_product_id = product_options[selected_product_label]

if st.button("Recommend Similar Products"):
    request_data = {
        "product_id": selected_product_id
    }

    response = requests.post(
        f"{API_BASE_URL}/recommend",
        json=request_data
    )

    if response.status_code == 200:
        result = response.json()

        selected_product = result["selected_product"]
        recommendations = result["recommendations"]

        st.subheader("Selected Product")

        st.info(
            f"""
            **{selected_product['product_name']}**  
            Category: {selected_product['category']}  
            Price: ₹{selected_product['price']}  
            Rating: {selected_product['rating']}
            """
        )

        st.subheader("Recommended Products")

        for product in recommendations:
            st.success(
                f"""
                **{product['product_name']}**  
                Category: {product['category']}  
                Price: ₹{product['price']}  
                Rating: {product['rating']}
                """
            )

    else:
        st.error("Unable to get recommendations.")
