import streamlit as st
import pandas as pd
import base64
import numpy as np
from ml_model import ProductRecommender ,df
from ml_model import cosine_sim_matrix
recommender = ProductRecommender(df,cosine_sim_matrix)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Samsung Product Recommendation System",
    layout="wide"
)
import base64
# ========== FULL PAGE BG.PNG ==========
def get_bg_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_base64 = get_bg_base64("bg.png")

st.markdown("""
<style>
.block-container {
    padding-top: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
}
</style>
""", unsafe_allow_html=True) 
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)
# =====================================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
img_base64 = get_base64_image("image1.jpeg")


st.markdown("""
<style>
.block-container {
    padding-top: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>
.hero {{
    position: relative;
    width: 100vw;
    height: 680px;
    margin-left: calc(-50vw + 50%);
    margin-right: calc(-50vw + 50%);
    background-image: 
        linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
        url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}}

.hero-content {{
    color: white;
    max-width: 900px;
    padding: 20px;
}}

.hero-content h1 {{
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 10px;
    color: white;
    text-shadow: 0 4px 12px rgba(0,0,0,0.6);
}}

.hero-content p {{
    font-size: 20px;
    color: #f1f1f1;
}}
</style>

<div class="hero">
    <div class="hero-content">
        <h1>Samsung Product Recommendation System</h1>
        <p>Smart recommendations for smart living</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Product cards + details - WHITE FONT */
div[class*="stHorizontalBlock"] h1, h2, h3,
div[data-testid="column"] h1, h2, h3,
div:has(input) h1, h2, h3,
.metric > div > div > div {
    color: white !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.7) !important;
    font-weight: 700 !important;
}

/* Product text white */
div[class*="stHorizontalBlock"] p,
div[data-testid="column"] p,
div:has(input) p,
.metric-container label {
    color: #f8f9ff !important;
    font-weight: 500 !important;
}

/* Button text white */
button span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
# ========== BUTTON TEXT BLACK FIX ==========
st.markdown("""
<style>
/* Buttons ka text BLACK */
div.stButton > button > div,
div.stButton > button > span,
button span,
.stButton > button * {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Button background gradient */
div.stButton > button {
    background: linear-gradient(45deg, #0d6efd, #28a745) !important;
    border-radius: 25px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
}
</style>
""", unsafe_allow_html=True)
# ==========================================

# ---------------- SELECT BOX ----------------
if "page" not in st.session_state:
    st.session_state.page = "list"

if "selected_product_id" not in st.session_state:
    st.session_state.selected_product_id = None

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "show_similar" not in st.session_state:
    st.session_state.show_similar = False



if st.session_state.page == "list":

    st.subheader("🔎 Search Product by Description")
    query = st.text_input("Type what you are looking for (eg: Smart Watch,AC,Gaming Phone)")

    if query:
    
       st.session_state.last_results = recommender.recommend_from_query(query, top_k=6)

    results = st.session_state.last_results
    if results is not None:

        for idx, row in results.iterrows():
            col1, col2, col3 = st.columns([1,3,0.5])

            img = str(row["Image URL"]).strip()
            if img.startswith("//"):
                img = "https:" + img

            with col1:
                if img.startswith("http"):
                    st.image(img, width=200)

            with col2:
                st.markdown(f"### {row['Product Name']}")
                st.write("*Price:*₹", row["Price"])
                st.write("*Rating:*⭐", row["Rating"])
                st.divider()
                st.write("")

            with col3:
                if st.button("View Details", key=idx):
                    st.session_state.selected_product_id = row[" Product ID"]
                    st.session_state.page = "details"
                    st.session_state.show_similar = False
                    st.rerun()
            
   # ========== PRODUCT DETAILS SECTION ==========#

if st.session_state.page == "details":

    product = df[df[" Product ID"] == st.session_state.selected_product_id].iloc[0]

    col1, col2 = st.columns([1,2])

    img = str(product["Image URL"]).strip()
    if img.startswith("//"):
        img = "https:" + img

    with col1:
        st.image(img, width=300)

    with col2:
        st.markdown(f"# {product['Product Name']}")
        st.write("*Price:*₹", product["Price"])
        st.write("*Rating:*⭐", product["Rating"])
        st.write("Category:", product["Category"])
        st.write("RAM:", product["RAM_GB"], "GB")
        st.write("Storage:", product["Storage_GB"], "GB")
        st.write(product["Description"])

    if st.button("⬅ Back to results"):
        st.session_state.page = "list"
        st.rerun()
        
    # SIMILAR PRODUCTS
    if st.button("🔁 View Similar Products"):
       st.session_state.show_similar = True
       st.rerun()
    
    if st.session_state.show_similar:
        st.subheader("🤝 Similar Products")
        similar_products = recommender.recommend(st.session_state.selected_product_id, top_k=6)
        
        for _, row in similar_products.iterrows():
            col1, col2 = st.columns([1.5, 3])
            img = str(row["Image URL"]).strip()
            if img.startswith("//"):
                img = "https:" + img
            
            with col1:
                st.image(img if img.startswith("http") else "https://via.placeholder.com/150?text=No+Img", width=180)
            
            with col2:
                st.markdown(f"**{row['Product Name']}**")
                st.write("*Price:*₹", row["Price"])
                st.write("*Rating:*⭐", row["Rating"])
            st.divider()
            st.write("")
  


               

