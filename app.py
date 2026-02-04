import streamlit as st
import pandas as pd
import base64
import numpy as np
from ml.recommender import ProductRecommender
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Samsung Product Recommendation System",
    layout="wide"
)
import base64

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
    height: 700px;
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



#---------------- LOAD DATA ----------------
df = pd.read_csv("Review_samsung .csv")

@st.cache_data
def load_similarity(data):
    numeric_df = data.select_dtypes(include=[np.number]).fillna(0)
    return cosine_similarity(numeric_df)
similarity_matrix = load_similarity(df)












recommender = ProductRecommender(df ,similarity_matrix)




# ---------------- SELECT BOX ----------------


st.subheader("🔍 Select a Product")

product_names = ["-- Select a Product --"] + list(df["Product Name"].unique())
selected_name = st.selectbox("Select Product by Name", product_names)

if selected_name == "-- Select a Product --":
    st.stop()



# Get Product ID of selected product
selected_product_id = df[df["Product Name"] == selected_name][" Product ID"].values[0]
product = df[df[" Product ID"] == selected_product_id].iloc[0]

st.subheader("📦 Product Details")


col1, col2 = st.columns([1, 2])

img = str(product["Image URL"]).strip()

# Fix broken URLs starting with //
if img.startswith("//"):
    img = "https:" + img

with col1:

# Show image only if valid
    if img.startswith("https"):
       st.image(img, width=250)
    else:
       st.warning("Image not available")




with col2:
    st.markdown(f"### {product['Product Name']}")
    st.write("**Category:**", product["Category"])
    st.write("**Price:** ₹", product["Price"])
    st.write("**Rating:** ⭐", product["Rating"])
    st.write("**Color:**", product["Color"])
    st.write("**RAM:**", product["RAM"], "GB")
    st.write("**Storage:**", product["Storage"], "GB")
    st.write("**Description:**", product["Descriptoin"])




    
# ---------------- BUTTON ----------------
if st.button("🔁 Recommend Similar Products"):

    similar_ids = recommender.recommend(selected_product_id)[" Product ID"]

    similar_products = df[df[" Product ID"].isin(similar_ids)]

    # Same category filter
    selected_category = product["Category"]
    similar_products = similar_products[similar_products["Category"] == selected_category]

    # ❗ FALLBACK if empty (buds/watches case)
    if similar_products.empty:
        similar_products = df[
            (df["Category"] == selected_category) &
            (df[" Product ID"] != selected_product_id)
        ].sort_values(by="Rating", ascending=False).head(3)

    st.subheader("✨ Similar Products")

    for _, row in similar_products.head(6).iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        img = str(row.get("Image URL", "")).strip()
        if img.startswith("//"):
            img = "https:" + img

        with col1:
            if img.startswith("http"):
                st.image(img, width=180)
            else:
                st.write("No Image")

        with col2:
            st.markdown(f"### {row['Product Name']}")
            st.write("**Category:**", row["Category"])
            st.write("**Price:** ₹", row["Price"])
            st.write("**Rating:** ⭐", row["Rating"])
